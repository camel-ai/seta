"""Claude SDK agent implementations for the evolution pipeline.

Provides ClaudeEvolAgent and ClaudeDatapointAgent, using the same SDK
patterns as seed2synth_pipeline/agents/claude_agents.py but adapted for
the evolution workflow (base prompt + strategy adapter, evolution fidelity).
"""

import os
from typing import Any, Dict, List, Optional

from pipeline_base import (
    EvolAgent, DatapointAgent,
    TaskContext, EvolvedTask, EvolutionOption, DatapointResult,
)

try:
    from claude_agent_sdk import (
        ClaudeSDKClient, ClaudeAgentOptions, HookMatcher, HookContext,
    )
except ImportError:
    print("Warning: claude_agent_sdk not found. Claude agents will not work.")
    HookContext = Any


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))


class ClaudeRateLimitError(RuntimeError):
    """Raised when Claude SDK emits an explicit quota/rate-limit event."""


def _message_indicates_rate_limit(message: Any) -> bool:
    try:
        from claude_agent_sdk import RateLimitEvent
        if isinstance(message, RateLimitEvent):
            return message.rate_limit_info.status in (
                "blocked", "rate_limited", "throttled", "denied",
            )
    except Exception:
        pass
    msg = str(message).lower()
    return (
        "error='rate_limit'" in msg
        or 'error="rate_limit"' in msg
        or "you're out of extra usage" in msg
    )


def _resolve_agent_path(file_path: str, agent_cwd: str) -> str:
    """Resolve a file path from the agent against the agent's cwd."""
    if os.path.isabs(file_path):
        return os.path.normpath(file_path)
    return os.path.normpath(os.path.join(agent_cwd, file_path))


def _read_file_safe(path: str) -> str:
    """Read a text file, returning '' on any error."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Security hooks
# ---------------------------------------------------------------------------

def create_validate_write_command(
    allowed_write_dir: str,
    agent_cwd: str,
    protected_files: Optional[List[str]] = None,
):
    abs_allowed = os.path.abspath(allowed_write_dir)
    _cwd = os.path.abspath(agent_cwd)
    protected = protected_files or []

    async def hook(input_data: Dict[str, Any], tool_use_id: str | None, context: HookContext) -> Dict[str, Any]:
        if input_data["tool_name"] in ("Write", "Edit"):
            fp = input_data["tool_input"].get("file_path", "")
            abs_fp = _resolve_agent_path(fp, _cwd)
            if not abs_fp.startswith(abs_allowed):
                return {"hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"Cannot write to {fp} outside {allowed_write_dir}.",
                }}
            for pf in protected:
                if abs_fp.endswith(pf) or os.path.basename(abs_fp) == pf:
                    return {"hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": f"{pf} is protected and cannot be modified.",
                    }}
        return {}

    return hook


def create_validate_read_command(allowed_dirs: List[str], agent_cwd: str):
    abs_allowed = [os.path.abspath(d) for d in allowed_dirs]
    _cwd = os.path.abspath(agent_cwd)

    async def hook(input_data: Dict[str, Any], tool_use_id: str | None, context: HookContext) -> Dict[str, Any]:
        if input_data["tool_name"] == "Read":
            fp = input_data["tool_input"].get("file_path", "")
            abs_fp = _resolve_agent_path(fp, _cwd)
            if not any(abs_fp.startswith(d) for d in abs_allowed):
                return {"hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"Cannot read {fp} outside allowed directories.",
                }}
        return {}

    return hook


def create_validate_bash_command(allowed_dirs: Optional[List[str]] = None):
    import re
    abs_allowed = [os.path.abspath(d) for d in (allowed_dirs or [])]

    async def hook(input_data: Dict[str, Any], tool_use_id: str | None, context: HookContext) -> Dict[str, Any]:
        if input_data["tool_name"] == "Bash":
            cmd = input_data["tool_input"].get("command", "")
            if "cd " in cmd:
                return {"hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "cd commands not allowed. Use absolute paths.",
                }}
            if re.search(r"\bln\s+-s\b|\bln\s+--symbolic\b", cmd):
                return {"hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "Symlinks are strictly forbidden.",
                }}
            if abs_allowed:
                for tok in re.findall(r"/[^\s'\";&|><()\\]+", cmd):
                    abs_tok = os.path.normpath(tok)
                    if not any(abs_tok.startswith(d) for d in abs_allowed):
                        return {"hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": f"Path {tok} outside allowed directories.",
                        }}
        return {}

    return hook


# ---------------------------------------------------------------------------
# Preload helpers
# ---------------------------------------------------------------------------

def _preload_example(guide_dir: str) -> str:
    """Load the hello-world example files as a reference block."""
    example_dir = os.path.join(guide_dir, "example", "hello-world")
    if not os.path.isdir(example_dir):
        return ""

    lines = ["## Reference: example/hello-world/ (structure only — do NOT copy trivial content)\n"]
    for root, _dirs, files in sorted(os.walk(example_dir)):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, example_dir)
            content = _read_file_safe(fpath)
            lines.append(f"### `{rel}`\n```\n{content.rstrip()}\n```\n")
    return "\n".join(lines)


def _preload_guide(
    guide_dir: str,
    evol_task_path: str,
    substitutions: Optional[Dict[str, str]] = None,
) -> str:
    """Load agent.md + example + draft_spec.md + judge_report.md into a single prompt block."""
    sections: List[str] = []

    content = _read_file_safe(os.path.join(guide_dir, "agent.md"))
    if content:
        for placeholder, value in (substitutions or {}).items():
            content = content.replace(placeholder, value)
        sections.append(f"## Reference: agent.md\n\n{content}")

    example = _preload_example(guide_dir)
    if example:
        sections.append(example)

    spec = _read_file_safe(os.path.join(evol_task_path, "draft_spec.md"))
    if spec:
        sections.append(f"## draft_spec.md\n\n{spec}")

    judge = _read_file_safe(os.path.join(evol_task_path, "judge_report.md"))
    if judge:
        sections.append(
            f"## judge_report.md — address every FAIL item before anything else\n\n{judge}"
        )

    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# Strategy adapter registry
# ---------------------------------------------------------------------------

STRATEGY_ADAPTERS: Dict[str, str] = {
    "INCREASE_DIFFICULTY": "evol_strategy_prompts/increase_difficulty_adapter.md",
    "DECREASE_DIFFICULTY": "evol_strategy_prompts/decrease_difficulty_adapter.md",
    "SLIGHT_INCREASE": "evol_strategy_prompts/slight_increase_adapter.md",
    "SLIGHT_DECREASE": "evol_strategy_prompts/slight_decrease_adapter.md",
    "CHANGE_CONTEXT": "evol_strategy_prompts/change_context_adapter.md",
    "INCREASE_DIFFICULTY_AND_CHANGE_CONTEXT": "evol_strategy_prompts/increase_difficulty_and_change_context_adapter.md",
}


# ---------------------------------------------------------------------------
# Evol Agent
# ---------------------------------------------------------------------------

class ClaudeEvolAgent(EvolAgent):
    """Reads an input Harbor task and writes draft_spec.md per variant.

    Uses base prompt + strategy adapter (like Seed2IdeaAgent uses
    base prompt + source adapter).
    """

    async def evolve(
        self,
        task_context: TaskContext,
        evol_opt: EvolutionOption,
        variant_paths: List[str],
        **kwargs,
    ) -> EvolvedTask:
        task_id = task_context.task_id
        input_task_path = os.path.abspath(task_context.metadata["input_task_path"])
        evol_target = evol_opt.parameters.get("target", "INCREASE_DIFFICULTY")

        # --- Build prompt: base + adapter ---
        base_prompt_path = os.path.join(AGENTS_DIR, "evol_agent_base_prompt.md")
        base_prompt = _read_file_safe(base_prompt_path)
        if not base_prompt:
            raise FileNotFoundError(f"Base prompt not found: {base_prompt_path}")

        adapter_rel = STRATEGY_ADAPTERS.get(evol_target)
        if not adapter_rel:
            raise ValueError(
                f"No adapter for evol_target={evol_target}. "
                f"Available: {list(STRATEGY_ADAPTERS)}"
            )
        adapter_path = os.path.join(AGENTS_DIR, adapter_rel)
        adapter_text = _read_file_safe(adapter_path)
        if not adapter_text:
            raise FileNotFoundError(f"Strategy adapter not found: {adapter_path}")

        variant_dirs_text = "\n".join(f"- `{vp}`" for vp in variant_paths)

        prompt = base_prompt.format(
            task_id=task_id,
            input_task_path=input_task_path,
            evol_target=evol_target,
            variant_dirs=variant_dirs_text,
            strategy_instructions=adapter_text,
        )

        # --- Security hooks ---
        # Evol agent can read input task + write to variant dirs only
        all_write_dirs = variant_paths
        all_read_dirs = [input_task_path] + variant_paths

        # Use the first variant path's parent as the agent cwd
        agent_cwd = os.path.dirname(variant_paths[0]) if variant_paths else input_task_path

        validate_write = create_validate_write_command(
            agent_cwd, agent_cwd=agent_cwd,
        )
        validate_read = create_validate_read_command(all_read_dirs, agent_cwd=agent_cwd)
        validate_bash = create_validate_bash_command(all_read_dirs)

        log_path = os.path.join(
            variant_paths[0] if variant_paths else ".",
            "evol_agent_log.txt",
        )

        options = ClaudeAgentOptions(
            model="claude-sonnet-4-6",
            permission_mode="acceptEdits",
            cwd=agent_cwd,
            env=os.environ.copy(),
            allowed_tools=["Bash", "Read", "Write", "WebSearch", "WebFetch"],
            hooks={
                "PreToolUse": [
                    HookMatcher(matcher="Bash", hooks=[validate_bash]),
                    HookMatcher(matcher="Edit|Write", hooks=[validate_write]),
                    HookMatcher(matcher="Read", hooks=[validate_read]),
                ],
            },
        )

        print(f"[{task_id}] Evol Agent ({evol_target}) working in: {agent_cwd}")
        print(f"[{task_id}] Evol Agent log: {log_path}")

        with open(log_path, "w") as log_f:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    log_f.write(str(message))
                    log_f.write("\n")
                    if _message_indicates_rate_limit(message):
                        raise ClaudeRateLimitError(
                            "Claude rate limit/quota reached."
                        )

        return EvolvedTask(
            task_id=task_id,
            variant_paths=variant_paths,
            metadata={"log_file": log_path, "evol_target": evol_target},
        )


# ---------------------------------------------------------------------------
# Datapoint Agent
# ---------------------------------------------------------------------------

class ClaudeDatapointAgent(DatapointAgent):
    """Reads draft_spec.md and builds a complete Harbor task with self-assessment.

    Adapted from seed2synth's ClaudeDatapointAgent with evolution fidelity
    context: passes the original input task path and evol_target so the agent
    can verify evolution fidelity in its self-review.
    """

    async def create(
        self,
        task_id_evol: str,
        evol_task_path: str,
        guide_dir: str,
        *,
        input_task_path: Optional[str] = None,
        evol_target: Optional[str] = None,
        **kwargs,
    ) -> DatapointResult:
        evol_task_path = os.path.abspath(evol_task_path)
        log_file_path = os.path.join(evol_task_path, "datapoint_agent_log.txt")

        # Harbor validation output: <rollout_dir>/validation/<task_id>/
        # Sibling of model rollout dirs (kimi_k2/, azure_gpt54/, ...).
        rollout_dir = kwargs.get("rollout_dir", "")
        if rollout_dir:
            harbor_out = os.path.abspath(
                os.path.join(rollout_dir, "validation", task_id_evol)
            )
        else:
            harbor_out = os.path.abspath(
                os.path.join(evol_task_path, "..", "..", "evol_data_rollouts", "validation", task_id_evol)
            )

        allowed_dirs = [guide_dir, evol_task_path, harbor_out]
        if input_task_path and os.path.isdir(input_task_path):
            allowed_dirs.append(os.path.abspath(input_task_path))

        validate_bash = create_validate_bash_command(allowed_dirs)
        validate_write = create_validate_write_command(
            evol_task_path,
            agent_cwd=evol_task_path,
            protected_files=["draft_spec.md"],
        )
        validate_read = create_validate_read_command(allowed_dirs, agent_cwd=evol_task_path)

        # Preload guide with runtime path substitutions
        substitutions = {
            "<task-dir>": evol_task_path,
            "<harbor-out>": harbor_out,
            "{evol_target}": evol_target or "UNKNOWN",
            "{input_task_path}": input_task_path or "(not provided)",
        }
        preloaded = _preload_guide(guide_dir, evol_task_path, substitutions=substitutions)

        # Build context about the original input task for evolution fidelity
        input_context = ""
        if input_task_path and os.path.isdir(input_task_path):
            input_context = (
                f"\n## Original Input Task (read-only, for evolution fidelity comparison)\n"
                f"Path: `{input_task_path}`\n"
                f"You have read access. Compare your built task against this to verify "
                f"evolution fidelity (criterion 7 in self-review).\n"
            )

        prompt = (
            f"Your working directory (task dir): {evol_task_path}\n"
            f"Evolution strategy applied: **{evol_target or 'UNKNOWN'}**\n\n"
            f"{input_context}\n"
            f"{preloaded}\n\n"
            f"Now build the complete Harbor task. Go!"
        )

        options = ClaudeAgentOptions(
            model="claude-sonnet-4-6",
            permission_mode="acceptEdits",
            cwd=evol_task_path,
            env=os.environ.copy(),
            allowed_tools=["Bash", "Read", "Write", "Edit"],
            hooks={
                "PreToolUse": [
                    HookMatcher(matcher="Bash", hooks=[validate_bash]),
                    HookMatcher(matcher="Edit|Write", hooks=[validate_write]),
                    HookMatcher(matcher="Read", hooks=[validate_read]),
                ],
            },
        )

        print(f"[{task_id_evol}] Datapoint Agent working in: {evol_task_path}")
        print(f"[{task_id_evol}] Datapoint Agent log: {log_file_path}")

        with open(log_file_path, "w") as log_f:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    log_f.write(str(message))
                    log_f.write("\n")
                    if _message_indicates_rate_limit(message):
                        raise ClaudeRateLimitError(
                            "Claude rate limit/quota reached."
                        )

        return DatapointResult(
            task_id_evol=task_id_evol,
            metadata={
                "evol_task_path": evol_task_path,
                "log_file": log_file_path,
                "harbor_out": harbor_out,
            },
        )


# ---------------------------------------------------------------------------
# Trajectory Judge Agent
# ---------------------------------------------------------------------------

class ClaudeTrajectoryJudgeAgent:
    """Analyzes all-fail rollout trajectories to classify failure mode.

    Writes ``traj_judge_report.md`` in the task folder.
    """

    async def judge(
        self,
        task_id: str,
        task_path: str,
        rollout_dir: str,
        failure_summary: str,
    ) -> Dict[str, Any]:
        task_path = os.path.abspath(task_path)
        rollout_dir = os.path.abspath(rollout_dir)

        prompt_path = os.path.join(AGENTS_DIR, "traj_judge_prompt.md")
        prompt_template = _read_file_safe(prompt_path)
        if not prompt_template:
            raise FileNotFoundError(f"Judge prompt not found: {prompt_path}")

        prompt = prompt_template.format(
            task_id=task_id, task_path=task_path,
            rollout_dir=rollout_dir, failure_summary=failure_summary,
        )

        log_path = os.path.join(task_path, "traj_judge_log.txt")
        report_path = os.path.join(task_path, "traj_judge_report.md")

        validate_write = create_validate_write_command(
            task_path, agent_cwd=task_path,
            protected_files=["draft_spec.md", "synth_info.json"],
        )

        # Evidence is pre-embedded in the prompt; the judge only needs Write.
        options = ClaudeAgentOptions(
            model="claude-sonnet-4-6",
            permission_mode="acceptEdits",
            cwd=task_path,
            env=os.environ.copy(),
            allowed_tools=["Write"],
            hooks={
                "PreToolUse": [
                    HookMatcher(matcher="Write|Edit", hooks=[validate_write]),
                ],
            },
        )

        print(f"[{task_id}] Trajectory Judge working in: {task_path}")

        with open(log_path, "w") as log_f:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    log_f.write(str(message))
                    log_f.write("\n")
                    if _message_indicates_rate_limit(message):
                        raise ClaudeRateLimitError("Claude rate limit/quota reached.")

        # Parse verdict
        verdict = "too_hard"
        if os.path.exists(report_path):
            content = _read_file_safe(report_path)
            if "### Verdict: DESIGN_FLAW" in content:
                verdict = "design_flaw"
            elif "### Verdict: TOO_HARD" in content:
                verdict = "too_hard"

        print(f"[{task_id}] Trajectory Judge verdict: {verdict}")
        return {"verdict": verdict, "report_path": report_path}
