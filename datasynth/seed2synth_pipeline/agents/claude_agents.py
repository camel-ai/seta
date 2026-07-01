import glob
import os
import asyncio
from typing import Any, Dict, List, Optional
from pipeline_base import (
    AnalysisAgent, EvolAgent, DirectEvolAgent, Seed2IdeaAgent,
    DatapointAgent, JudgeAgent,
    TaskContext, AnalysisResult, EvolvedTask, DatapointResult, SynthResult,
    EvolutionOption
)
from io_utils import get_next_version, load_cleaned_trajectory
try:
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, HookMatcher, HookContext
except ImportError:
    print("Warning: claude_agent_sdk not found. Claude agents will not work.")
    HookContext = Any  # Fallback type if import fails


class ClaudeRateLimitError(RuntimeError):
    """Raised when Claude SDK emits an explicit quota/rate-limit event."""


def _message_indicates_rate_limit(message: Any) -> bool:
    """Best-effort detection for SDK rate-limit/quota events serialized as messages."""
    # Handle RateLimitEvent directly: only block on explicitly denied statuses.
    # 'allowed_warning' means approaching limit but still allowed — do not block.
    try:
        from claude_agent_sdk import RateLimitEvent
        if isinstance(message, RateLimitEvent):
            return message.rate_limit_info.status in ("blocked", "rate_limited", "throttled", "denied")
    except Exception:
        pass
    msg = str(message).lower()
    return (
        "error='rate_limit'" in msg
        or 'error="rate_limit"' in msg
        or "you're out of extra usage" in msg
    )

# --- Security Hooks ---

def _resolve_agent_path(file_path: str, agent_cwd: str) -> str:
    """Resolve a file path from the agent against the agent's cwd.

    The agent runs with its own cwd (set via SDK), which differs from the
    Python process cwd.  Relative paths from the agent must be resolved
    against the agent's cwd, not the process cwd.
    """
    if os.path.isabs(file_path):
        return os.path.normpath(file_path)
    return os.path.normpath(os.path.join(agent_cwd, file_path))


def create_validate_write_command(allowed_write_dir: str, agent_cwd: str, protected_files: List[str] = None):
    """Factory for write validation hook. Optionally protects specific filenames."""
    abs_allowed = os.path.abspath(allowed_write_dir)
    _agent_cwd = os.path.abspath(agent_cwd)
    protected = protected_files or []

    async def validate_write_command(input_data: Dict[str, Any], tool_use_id: str | None, context: HookContext) -> Dict[str, Any]:
        if input_data['tool_name'] in ['Write', 'Edit']:
            file_path = input_data['tool_input'].get('file_path', '')
            abs_path = _resolve_agent_path(file_path, _agent_cwd)
            if not abs_path.startswith(abs_allowed):
                return {
                    'hookSpecificOutput': {
                        'hookEventName': 'PreToolUse',
                        'permissionDecision': 'deny',
                        'permissionDecisionReason': f'Access denied: Cannot write to {file_path} outside {allowed_write_dir}.'
                    }
                }
            for pf in protected:
                if abs_path.endswith(pf) or os.path.basename(abs_path) == pf:
                    return {
                        'hookSpecificOutput': {
                            'hookEventName': 'PreToolUse',
                            'permissionDecision': 'deny',
                            'permissionDecisionReason': f'Access denied: {pf} is protected and cannot be modified.'
                        }
                    }
        return {}
    return validate_write_command

def create_validate_read_command(allowed_dirs: List[str], agent_cwd: str):
    """Factory for read validation hook. Allows reads from any of the given dirs."""
    abs_allowed = [os.path.abspath(d) for d in allowed_dirs]
    _agent_cwd = os.path.abspath(agent_cwd)

    async def validate_read_command(input_data: Dict[str, Any], tool_use_id: str | None, context: HookContext) -> Dict[str, Any]:
        if input_data['tool_name'] == 'Read':
            file_path = input_data['tool_input'].get('file_path', '')
            abs_path = _resolve_agent_path(file_path, _agent_cwd)
            if not any(abs_path.startswith(d) for d in abs_allowed):
                return {
                    'hookSpecificOutput': {
                        'hookEventName': 'PreToolUse',
                        'permissionDecision': 'deny',
                        'permissionDecisionReason': f'Access denied: Cannot read {file_path} outside allowed directories.'
                    }
                }
        return {}
    return validate_read_command

def create_validate_bash_command(allowed_dirs: List[str] | None = None):
    """Factory for bash validation hook.

    Blocks cd commands.  If allowed_dirs is given, also blocks any bash
    command that references an absolute path outside those directories
    (catches cat/find/ls/head/tail used to snoop other task folders).
    """
    import re
    abs_allowed = [os.path.abspath(d) for d in (allowed_dirs or [])]

    async def validate_bash_command(input_data: Dict[str, Any], tool_use_id: str | None, context: HookContext) -> Dict[str, Any]:
        if input_data['tool_name'] == 'Bash':
            command = input_data['tool_input'].get('command', '')
            if 'cd ' in command:
                return {
                    'hookSpecificOutput': {
                        'hookEventName': 'PreToolUse',
                        'permissionDecision': 'deny',
                        'permissionDecisionReason': 'cd commands are not allowed. Use absolute paths instead.'
                    }
                }
            if re.search(r'\bln\s+-s\b|\bln\s+--symbolic\b', command):
                return {
                    'hookSpecificOutput': {
                        'hookEventName': 'PreToolUse',
                        'permissionDecision': 'deny',
                        'permissionDecisionReason': 'Symlinks are strictly forbidden. Do not use ln -s.'
                    }
                }
            if abs_allowed:
                # Extract all absolute path tokens from the command
                for path_tok in re.findall(r'/[^\s\'";&|><()\\]+', command):
                    abs_tok = os.path.normpath(path_tok)
                    if not any(abs_tok.startswith(d) for d in abs_allowed):
                        return {
                            'hookSpecificOutput': {
                                'hookEventName': 'PreToolUse',
                                'permissionDecision': 'deny',
                                'permissionDecisionReason': (
                                    f'Access denied: path {path_tok} is outside allowed directories. '
                                    f'Only operate within your task directory.'
                                )
                            }
                        }
        return {}
    return validate_bash_command


# --- Analysis Agent ---

class ClaudeAnalysisAgent(AnalysisAgent):
    def __init__(self):
        pass

    async def analyze(self, task_context: TaskContext, evol_opt: EvolutionOption = None, **kwargs) -> AnalysisResult:
        task_id = task_context.task_id

        # Analysis output goes to the seed task folder (not the evolved folder)
        seed_path = task_context.metadata.get("seed_path")
        if not seed_path:
            raise ValueError("seed_path must be set in task_context.metadata")
        os.makedirs(seed_path, exist_ok=True)

        log_file_path = os.path.join(seed_path, "analysis_agent_log.txt")
        report_file = os.path.join(seed_path, "analysis_report.md")

        # Format rollout info
        rollout_info_lines = []
        for i, r in enumerate(task_context.rollouts):
            rollout_info_lines.append(f"--- Rollout {i} ---")
            cleaned_traj = load_cleaned_trajectory(r['trajectory'])
            rollout_info_lines.append(f"Cleaned Trajectory:\n{cleaned_traj}")
            rollout_info_lines.append(f"Test Results: {r['test_results']}")
            rollout_info_lines.append("")
            if i >= 0:  # limit to first rollout for now
                break
        rollout_info = "\n".join(rollout_info_lines)

        # Load prompt template
        prompt_path = os.path.join(os.path.dirname(__file__), "analysis_agent_prompt.md")
        with open(prompt_path, 'r') as f:
            prompt_tmpl = f.read()

        if not evol_opt:
            raise ValueError("Evolution option (evol_opt) must be provided to the analysis agent.")

        evol_context = f"Strategy: {evol_opt.strategy}\nParameters: {evol_opt.parameters}"
        prompt = prompt_tmpl.format(rollout_info=rollout_info, evol_context=evol_context)

        validate_write = create_validate_write_command(seed_path, agent_cwd=seed_path)
        validate_read = create_validate_read_command([seed_path], agent_cwd=seed_path)

        options = ClaudeAgentOptions(
            model="claude-opus-4-6",
            permission_mode='acceptEdits',
            cwd=seed_path,
            env=os.environ.copy(),
            allowed_tools=["Bash", "Read", "Write"],
            hooks={
                'PreToolUse': [
                    HookMatcher(matcher='Write', hooks=[validate_write]),
                    HookMatcher(matcher='Read', hooks=[validate_read]),
                ]
            }
        )

        analysis_content = ""
        print(f"[{task_id}] Analysis working in: {seed_path}")
        print(f"[{task_id}] Analysis log: {log_file_path}")

        with open(log_file_path, 'w') as log_f:
            log_f.write(f"Prompt:\n{prompt}\n\n")
            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    log_f.write(str(message))
                    log_f.write('\n')
                    if _message_indicates_rate_limit(message):
                        raise ClaudeRateLimitError(
                            "Claude rate limit/quota reached (e.g. out of extra usage)."
                        )

        if os.path.exists(report_file):
            with open(report_file, 'r') as f:
                analysis_content = f.read()
        else:
            analysis_content = None

        return AnalysisResult(
            task_id=task_id,
            analysis_content=analysis_content,
            metadata={"seed_path": seed_path, "log_file": log_file_path}
        )


# --- Evolution Agent ---

class ClaudeEvolAgent(EvolAgent):
    def __init__(self):
        pass

    async def evolve(
        self,
        task_context: TaskContext,
        analysis_result: AnalysisResult,
        evol_opt: EvolutionOption = None,
        variant_paths: List[str] = None,
        **kwargs
    ) -> EvolvedTask:
        task_id = task_context.task_id
        seed_path = task_context.metadata.get("seed_path")

        if not evol_opt:
            raise ValueError("Evolution option (evol_opt) must be provided to the evol agent.")
        if not variant_paths:
            raise ValueError("variant_paths must be provided to the evol agent.")

        # Use first variant dir as the cwd for the agent (it can write to all variant dirs)
        evol_cwd = variant_paths[0]
        log_file_path = os.path.join(evol_cwd, "evol_agent_log.txt")

        # Build variant dir listing for the prompt
        variant_dirs_str = "\n".join(f"- {p}" for p in variant_paths)
        n_variants = len(variant_paths)

        # Load prompt template
        prompt_path = os.path.join(os.path.dirname(__file__), "evol_agent_prompt.md")
        with open(prompt_path, 'r') as f:
            prompt_tmpl = f.read()

        evol_context = f"Strategy: {evol_opt.strategy}\nParameters: {evol_opt.parameters}"
        prompt = prompt_tmpl.format(
            task_id=task_id,
            evol_context=evol_context,
            analysis_content=analysis_result.analysis_content or "(no analysis available)",
            variant_dirs=variant_dirs_str,
            n_variants=n_variants,
        )

        # Allow reads from seed_path and all variant dirs
        allowed_read_dirs = [seed_path] + variant_paths
        validate_read = create_validate_read_command(allowed_read_dirs, agent_cwd=evol_cwd)
        # Allow writes to any variant dir
        # We do this by allowing writes to the common parent of all variant paths
        evol_data_base = os.path.dirname(variant_paths[0])  # parent = evol_data_base/<seed_name>/
        validate_write = create_validate_write_command(evol_data_base, agent_cwd=evol_cwd)

        options = ClaudeAgentOptions(
            model="claude-opus-4-6",
            permission_mode='acceptEdits',
            cwd=evol_cwd,
            env=os.environ.copy(),
            allowed_tools=["Bash", "Read", "Write", "WebSearch", "WebFetch"],
            hooks={
                'PreToolUse': [
                    HookMatcher(matcher='Edit|Write', hooks=[validate_write]),
                    HookMatcher(matcher='Read', hooks=[validate_read]),
                ]
            }
        )

        print(f"[{task_id}] Evol Agent working in: {evol_cwd}")
        print(f"[{task_id}] Variants to populate: {variant_paths}")
        print(f"[{task_id}] Evol Agent log: {log_file_path}")

        with open(log_file_path, 'w') as log_f:
            log_f.write(f"Prompt:\n{prompt}\n\n")
            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    log_f.write(str(message))
                    log_f.write('\n')
                    if _message_indicates_rate_limit(message):
                        raise ClaudeRateLimitError(
                            "Claude rate limit/quota reached (e.g. out of extra usage)."
                        )

        return EvolvedTask(
            task_id=task_id,
            variant_paths=variant_paths,
            metadata={"evol_cwd": evol_cwd, "log_file": log_file_path}
        )


# --- Direct Evolution Agent (no analysis step) ---

class ClaudeDirectEvolAgent(DirectEvolAgent):
    def __init__(self):
        pass

    async def evolve(
        self,
        task_context: TaskContext,
        evol_opt: EvolutionOption,
        variant_paths: List[str],
        **kwargs
    ) -> EvolvedTask:
        task_id = task_context.task_id
        seed_path = task_context.metadata.get("seed_path")

        if not evol_opt:
            raise ValueError("Evolution option (evol_opt) must be provided to the direct evol agent.")
        if not variant_paths:
            raise ValueError("variant_paths must be provided to the direct evol agent.")

        evol_cwd = variant_paths[0]
        log_file_path = os.path.join(evol_cwd, "evol_agent_log.txt")

        variant_dirs_str = "\n".join(f"- {p}" for p in variant_paths)
        n_variants = len(variant_paths)

        # Load the direct evol prompt (no analysis_content placeholder)
        prompt_path = os.path.join(os.path.dirname(__file__), "direct_evol_agent_prompt.md")
        with open(prompt_path, 'r') as f:
            prompt_tmpl = f.read()

        evol_context = f"Strategy: {evol_opt.strategy}\nParameters: {evol_opt.parameters}"
        prompt = prompt_tmpl.format(
            task_id=task_id,
            seed_path=seed_path,
            evol_context=evol_context,
            variant_dirs=variant_dirs_str,
            n_variants=n_variants,
        )

        # Allow reads from seed_path and all variant dirs
        allowed_read_dirs = [seed_path] + variant_paths
        validate_read = create_validate_read_command(allowed_read_dirs, agent_cwd=evol_cwd)
        # Allow writes to the common parent of all variant paths
        evol_data_base = os.path.dirname(variant_paths[0])
        validate_write = create_validate_write_command(evol_data_base, agent_cwd=evol_cwd)

        options = ClaudeAgentOptions(
            model="claude-opus-4-6",
            permission_mode='acceptEdits',
            cwd=evol_cwd,
            env=os.environ.copy(),
            allowed_tools=["Bash", "Read", "Write", "WebSearch", "WebFetch"],
            hooks={
                'PreToolUse': [
                    HookMatcher(matcher='Edit|Write', hooks=[validate_write]),
                    HookMatcher(matcher='Read', hooks=[validate_read]),
                ]
            }
        )

        print(f"[{task_id}] Direct Evol Agent working in: {evol_cwd}")
        print(f"[{task_id}] Variants to populate: {variant_paths}")
        print(f"[{task_id}] Direct Evol Agent log: {log_file_path}")

        with open(log_file_path, 'w') as log_f:
            log_f.write(f"Prompt:\n{prompt}\n\n")
            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    log_f.write(str(message))
                    log_f.write('\n')
                    if _message_indicates_rate_limit(message):
                        raise ClaudeRateLimitError(
                            "Claude rate limit/quota reached (e.g. out of extra usage)."
                        )

        return EvolvedTask(
            task_id=task_id,
            variant_paths=variant_paths,
            metadata={"evol_cwd": evol_cwd, "log_file": log_file_path}
        )


# --- Preload helpers ---

def _read_file_safe(path: str) -> str | None:
    """Return file contents or None if missing/unreadable."""
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception:
        return None


_PRELOAD_MAX_CHARS = 80_000   # ~20k tokens; skip individual file if it exceeds this


# ---------------------------------------------------------------------------
# Source-specific seed data adapter functions
#
# Each function receives the seed_data_folder path and returns a prompt block:
#   - Fully inlines small/structured files (JSON Q&A, metadata)
#   - For large or binary files: describes the structure and lists paths so the
#     agent knows exactly what to Read and what schema to expect
#
# Return empty string to fall back to the agent reading files on its own.
# ---------------------------------------------------------------------------

_QA_JSON_FIELDS = ("tags", "title", "category", "question_text_markdown", "answer_text_markdown")


def _extract_qa_fields(raw_json: str) -> str:
    """Parse a Q&A JSON string and return only the fields the idea agent needs."""
    import json as _json
    try:
        data = _json.loads(raw_json)
    except Exception:
        return raw_json   # malformed — return as-is
    extracted = {k: data[k] for k in _QA_JSON_FIELDS if k in data}
    return _json.dumps(extracted, ensure_ascii=False, indent=2)


def _seed_adapter_qa_json(seed_data_folder: str) -> str:
    """Adapter for Q&A JSON sources: unix_linux_se, stackoverflow, nl2bash.

    Extracts only the fields the idea agent needs (tags, title, category,
    question_text_markdown, answer_text_markdown) — skips redundant plain-text
    duplicates and metadata fields.
    """
    main_raw = _read_file_safe(os.path.join(seed_data_folder, "main.json"))
    if not main_raw:
        return ""

    parts = [f"### main.json\n```json\n{_extract_qa_fields(main_raw)}\n```"]
    for related_path in sorted(glob.glob(os.path.join(seed_data_folder, "related_*.json"))):
        raw = _read_file_safe(related_path)
        if raw:
            name = os.path.basename(related_path)
            parts.append(f"### {name}\n```json\n{_extract_qa_fields(raw)}\n```")

    return "## Preloaded Seed Data\n\n" + "\n\n".join(parts)


def _seed_adapter_kaggle(seed_data_folder: str) -> str:
    """Adapter for Kaggle notebook seeds with local dataset support.

    Inlines:
      - kernel-metadata.json (small, ~5KB)
      - datasets/*/manifest.json (dataset metadata, ~1KB each)
      - List of actual data files (CSV, parquet, etc.) in each dataset folder

    Describes (for Read tool):
      - Notebook .ipynb file (can be large, 1-5MB)

    All datasets are pre-downloaded locally — agent loads from datasets/ folder.
    """
    parts: list[str] = []

    # 1. Preload kernel metadata
    meta_content = _read_file_safe(os.path.join(seed_data_folder, "kernel-metadata.json"))
    if meta_content:
        parts.append(f"### kernel-metadata.json (preloaded)\n```json\n{meta_content}\n```")

    # 2. Preload dataset manifests and list available data files
    datasets_dir = os.path.join(seed_data_folder, "datasets")
    if os.path.exists(datasets_dir):
        datasets_info = []
        for dataset_dir in sorted(os.listdir(datasets_dir)):
            dataset_path = os.path.join(datasets_dir, dataset_dir)
            if not os.path.isdir(dataset_path):
                continue

            # Preload manifest
            manifest_file = os.path.join(dataset_path, "manifest.json")
            manifest_content = None
            if os.path.exists(manifest_file):
                manifest_content = _read_file_safe(manifest_file)

            # List actual data files (CSV, parquet, JSON, etc.)
            data_files = []
            for fname in sorted(os.listdir(dataset_path)):
                fpath = os.path.join(dataset_path, fname)
                if os.path.isfile(fpath) and fname != "manifest.json":
                    size_kb = os.path.getsize(fpath) // 1024
                    data_files.append(f"  - `{fname}` ({size_kb} KB)")

            if manifest_content or data_files:
                datasets_info.append({
                    'name': dataset_dir,
                    'manifest': manifest_content,
                    'files': data_files
                })

        if datasets_info:
            parts.append("### Datasets (Preloaded & Available Locally)")
            for ds_info in datasets_info:
                parts.append(f"#### {ds_info['name']}/")
                if ds_info['manifest']:
                    parts.append(f"**Manifest:**\n```json\n{ds_info['manifest']}\n```")
                if ds_info['files']:
                    parts.append("**Available data files:**")
                    parts.append("\n".join(ds_info['files']))

    # 3. Extract notebook structure (cell summaries) to avoid repeated reads
    notebook_structure = []
    for ext in ("*.ipynb",):
        for notebook_path in sorted(glob.glob(os.path.join(seed_data_folder, ext))):
            try:
                import json
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    notebook = json.load(f)

                # Extract cell structure
                cells_summary = []
                for i, cell in enumerate(notebook.get('cells', [])[:20]):  # First 20 cells
                    cell_type = cell.get('cell_type', 'unknown')
                    if cell_type == 'markdown':
                        source_text = ''.join(cell.get('source', [])).strip()
                        # Extract heading if present
                        if source_text.startswith('#'):
                            heading = source_text.split('\n')[0][:60]
                            cells_summary.append(f"  {i+1}. [Markdown] {heading}")
                    elif cell_type == 'code':
                        source_text = ''.join(cell.get('source', []))
                        # Extract first meaningful line or function call
                        first_line = [l.strip() for l in source_text.split('\n') if l.strip() and not l.strip().startswith('#')][0] if source_text.strip() else ''
                        if first_line:
                            preview = first_line[:70]
                            cells_summary.append(f"  {i+1}. [Code] {preview}...")

                if cells_summary:
                    notebook_structure.append(
                        f"### Notebook Structure ({os.path.basename(notebook_path)})\n"
                        + "First 20 cells (read file directly if you need full details):\n"
                        + "\n".join(cells_summary)
                    )
            except Exception as e:
                pass  # Fallback if notebook parsing fails

    # 4. Describe notebook files (agent reads with Read tool if needed)
    notebook_files = []
    for ext in ("*.ipynb",):
        for p in sorted(glob.glob(os.path.join(seed_data_folder, ext))):
            size_kb = os.path.getsize(p) // 1024
            notebook_files.append(
                f"- `{os.path.basename(p)}` ({size_kb} KB) — read with the Read tool if needed"
            )

    # Add notebook structure summary before asking agent to read
    if notebook_structure:
        parts.extend(notebook_structure)

    if notebook_files:
        parts.append(
            "### Notebook File\n"
            + "Full notebook: " + "\n".join(notebook_files) + "\n\n"
            + "Use the Read tool ONLY if you need specific cell details beyond the structure above."
        )

    if not parts:
        return ""

    return (
        "## Seed Data\n\n"
        + "\n\n".join(parts)
        + "\n\n> **Datasets are pre-downloaded and ready to load from local folders.** "
        "kernel-metadata.json and manifests are preloaded above. "
        "Use the Read tool to examine the notebook file if needed."
    )


# Registry: source_type -> adapter function
_SEED_ADAPTER_MAP: dict[str, callable] = {
    "unix_linux_se":  _seed_adapter_qa_json,
    "stackoverflow":  _seed_adapter_qa_json,
    "stack_overflow": _seed_adapter_qa_json,
    "nl2bash":        _seed_adapter_qa_json,
    "kaggle_notebook": _seed_adapter_kaggle,
}


def _preload_seed_data(source_type: str, seed_data_folder: str) -> str:
    """Dispatch to the source-specific adapter and return a prompt block.

    Returns empty string if no adapter is registered for the source.
    """
    adapter = _SEED_ADAPTER_MAP.get(source_type)
    if adapter is None:
        return ""
    return adapter(seed_data_folder)


def _preload_task_files(evol_task_path: str) -> str:
    """Load all Harbor task files for the judge agent into a prompt block.

    Includes draft_spec.md and all standard Harbor task files.
    """
    sections: list[str] = []

    # Spec first so the judge can compare intent vs. implementation
    for fname in ["draft_spec.md", "task.toml", "instruction.md",
                  "environment/Dockerfile", "solution/solve.sh",
                  "tests/test.sh", "tests/test_outputs.py", "weights.json"]:
        content = _read_file_safe(os.path.join(evol_task_path, fname))
        if content:
            ext = fname.rsplit(".", 1)[-1] if "." in fname else "text"
            lang = {"toml": "toml", "md": "markdown", "sh": "bash",
                    "py": "python", "json": "json", "Dockerfile": "dockerfile"}.get(ext, "")
            sections.append(f"### {fname}\n```{lang}\n{content}\n```")

    return "\n\n".join(sections) if sections else "(no task files found)"


def _preload_guide(guide_dir: str, evol_task_path: str,
                   substitutions: dict[str, str] | None = None) -> str:
    """Load agent.md + task-specific spec into a prompt block.

    substitutions: mapping of placeholder → value applied to agent.md
    (e.g. {"<tasks-path>": "/synth_data/unix_linux_se", "<task-id>": "102191"})
    The agent can start building immediately — no Read tool calls needed for reference material.
    """
    sections: list[str] = []

    for fname in ["agent.md"]:
        content = _read_file_safe(os.path.join(guide_dir, fname))
        if content:
            if substitutions:
                for placeholder, value in substitutions.items():
                    content = content.replace(placeholder, value)
            sections.append(f"## Reference: {fname}\n\n{content}")

    spec = _read_file_safe(os.path.join(evol_task_path, "draft_spec.md"))
    if spec:
        sections.append(f"## draft_spec.md\n\n{spec}")

    judge = _read_file_safe(os.path.join(evol_task_path, "judge_report.md"))
    if judge:
        sections.append(
            f"## judge_report.md — address every FAIL item before anything else\n\n{judge}"
        )

    return "\n\n---\n\n".join(sections)


# --- Seed-to-Idea Agent (converts raw external data into draft_spec.md) ---

AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))


class ClaudeSeed2IdeaAgent(Seed2IdeaAgent):
    """Claude SDK implementation of Seed2IdeaAgent.

    Concatenates the source-specific adapter prompt with the shared base prompt,
    injects {seed_data_folder} and {output_path} into the adapter, and runs a
    Claude agent to produce draft_spec.md.
    """

    ADAPTER_MAP = {
        "nl2bash":        "seed2idea_prompts/nl2bash_adapter.md",
        "stackoverflow":  "seed2idea_prompts/stackoverflow_adapter.md",
        "stack_overflow": "seed2idea_prompts/stackoverflow_adapter.md",  # dir name alias
        "unix_linux_se":  "seed2idea_prompts/unix_linux_se_adapter.md",
        "kaggle_notebook": "seed2idea_prompts/kaggle_notebook_adapter.md",
        # v1 source kept for backward compatibility
        "nvd":            "seed2idea_prompts/nvd_idea_agent_prompt.md",
    }
    BASE_PROMPT_PATH = "seed2idea_prompts/idea_agent_base_prompt.md"

    def __init__(self):
        pass

    async def generate(
        self,
        seed_data_folder: str,
        source_type: str,
        output_path: str,
        **kwargs
    ) -> EvolvedTask:
        if source_type not in self.ADAPTER_MAP:
            raise ValueError(
                f"Unsupported source_type '{source_type}'. "
                f"Available: {list(self.ADAPTER_MAP.keys())}"
            )

        task_name = os.path.basename(output_path)
        log_file_path = os.path.join(output_path, "idea_agent_log.txt")

        # Load and render adapter (source-specific, has template vars)
        adapter_path = os.path.join(AGENTS_DIR, self.ADAPTER_MAP[source_type])
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter prompt not found: {adapter_path}")
        with open(adapter_path, 'r') as f:
            adapter_tmpl = f.read()

        # Load base prompt (static, no template vars)
        base_path = os.path.join(AGENTS_DIR, self.BASE_PROMPT_PATH)
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Base prompt not found: {base_path}")
        with open(base_path, 'r') as f:
            base = f.read()

        # Render only the adapter; base is static
        # Use simple string replacement to safely handle arbitrary {braces} in content
        adapter_rendered = adapter_tmpl
        adapter_rendered = adapter_rendered.replace('{seed_data_folder}', seed_data_folder)
        adapter_rendered = adapter_rendered.replace('{output_path}', output_path)

        # Preload seed data via source-specific adapter
        seed_block = _preload_seed_data(source_type, seed_data_folder)
        seed_note = (
            "\n\n> **Seed data above is preloaded. "
            "Only use the Read tool for files explicitly listed as 'read with the Read tool'.**\n"
            if seed_block else ""
        )

        prompt = (
            adapter_rendered
            + (f"\n\n{seed_block}{seed_note}" if seed_block else "")
            + "\n\n---\n\n"
            + base
        )

        # Security hooks — writes restricted to output dir
        validate_write = create_validate_write_command(output_path, agent_cwd=output_path)
        validate_read = create_validate_read_command([output_path, seed_data_folder], agent_cwd=output_path)

        options = ClaudeAgentOptions(
            model="claude-opus-4-6",
            permission_mode='acceptEdits',
            cwd=output_path,
            env=os.environ.copy(),
            allowed_tools=["Bash", "Read", "Write", "WebSearch", "WebFetch"],
            hooks={
                'PreToolUse': [
                    HookMatcher(matcher='Edit|Write', hooks=[validate_write]),
                    HookMatcher(matcher='Read', hooks=[validate_read]),
                ]
            }
        )

        print(f"[{task_name}] Seed2Idea Agent working in: {output_path}")
        print(f"[{task_name}] Source type: {source_type}")
        print(f"[{task_name}] Seed folder: {seed_data_folder}")
        print(f"[{task_name}] Idea Agent log: {log_file_path}")

        os.makedirs(output_path, exist_ok=True)
        with open(log_file_path, 'w') as log_f:
            log_f.write(f"Prompt:\n{prompt}\n\n")
            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    log_f.write(str(message))
                    log_f.write('\n')
                    if _message_indicates_rate_limit(message):
                        raise ClaudeRateLimitError(
                            "Claude rate limit/quota reached (e.g. out of extra usage)."
                        )

        return EvolvedTask(
            task_id=task_name,
            variant_paths=[output_path],
            metadata={
                "output_path": output_path,
                "source_type": source_type,
                "seed_data_folder": seed_data_folder,
                "log_file": log_file_path,
            }
        )


# --- Datapoint Creation Agent ---

class ClaudeDatapointAgent(DatapointAgent):
    def __init__(self):
        pass

    async def create(
        self,
        task_id_evol: str,
        evol_task_path: str,
        guide_dir: str,
        seed_data_folder: str = "",
        **kwargs
    ) -> DatapointResult:
        log_file_path = os.path.join(evol_task_path, "datapoint_agent_log.txt")

        # Harbor run paths
        tasks_path    = os.path.dirname(evol_task_path)   # e.g. .../synth_data/unix_linux_se
        task_id       = os.path.basename(evol_task_path)  # e.g. 102191
        source        = os.path.basename(tasks_path)      # e.g. unix_linux_se
        scraping_base = os.path.dirname(os.path.dirname(tasks_path))  # e.g. .../seed_data_scraping
        harbor_out    = os.path.join(
            scraping_base, "synth_data_rollouts", "validation", source, task_id
        )  # e.g. .../synth_data_rollouts/validation/unix_linux_se/102191

        allowed_dirs = [guide_dir, evol_task_path, harbor_out]
        if seed_data_folder and os.path.isdir(seed_data_folder):
            allowed_dirs.append(seed_data_folder)
        validate_bash = create_validate_bash_command(allowed_dirs)
        validate_write = create_validate_write_command(
            evol_task_path,
            agent_cwd=evol_task_path,
            protected_files=["draft_spec.md"],
        )
        validate_read = create_validate_read_command(allowed_dirs, agent_cwd=evol_task_path)

        # Preload agent.md with runtime paths substituted in-place
        preloaded = _preload_guide(
            guide_dir, evol_task_path,
            substitutions={
                "<task-dir>":   evol_task_path,
                "<harbor-out>": harbor_out,
            },
        )

        # Build seed data context for the datapoint agent
        seed_data_note = ""
        if seed_data_folder and os.path.isdir(seed_data_folder):
            seed_data_note = (
                f"\n## Seed Data (read-only)\n"
                f"Original seed data is available at: `{seed_data_folder}`\n"
                f"You have read access to this directory.\n"
            )
            # List what's available
            datasets_dir = os.path.join(seed_data_folder, "datasets")
            if os.path.isdir(datasets_dir):
                dataset_files = []
                for root, dirs, files in os.walk(datasets_dir):
                    for f in files:
                        rel = os.path.relpath(os.path.join(root, f), seed_data_folder)
                        dataset_files.append(rel)
                if dataset_files:
                    seed_data_note += (
                        f"\n**IMPORTANT: Real dataset files are available below. You MUST use these real files in your task environment "
                        f"(e.g. COPY them in Dockerfile or entrypoint). Do NOT generate synthetic/fake data when real data exists.**\n"
                        f"\nDataset files:\n"
                    )
                    for df in dataset_files[:20]:
                        seed_data_note += f"- `{seed_data_folder}/{df}`\n"
                    if len(dataset_files) > 20:
                        seed_data_note += f"- ... and {len(dataset_files) - 20} more files\n"
            else:
                # No datasets/ subfolder — list top-level files
                top_files = os.listdir(seed_data_folder)
                if top_files:
                    seed_data_note += f"\nContents: {', '.join(top_files)}\n"

        prompt = (
            f"Your working directory (task dir): {evol_task_path}\n"
            f"You may read/write files inside this directory. "
            f"You may also READ files from the seed data folder listed below.\n\n"
            f"{seed_data_note}\n"
            f"{preloaded}\n\n"
            f"Now build the complete Harbor task. Go!"
        )

        options = ClaudeAgentOptions(
            model="claude-opus-4-6",
            permission_mode='acceptEdits',
            cwd=evol_task_path,
            env=os.environ.copy(),
            allowed_tools=["Bash", "Read", "Write", "Edit"],
            hooks={
                'PreToolUse': [
                    HookMatcher(matcher='Bash', hooks=[validate_bash]),
                    HookMatcher(matcher='Edit|Write', hooks=[validate_write]),
                    HookMatcher(matcher='Read', hooks=[validate_read]),
                ]
            }
        )

        print(f"[{task_id_evol}] Datapoint Agent working in: {evol_task_path}")
        print(f"[{task_id_evol}] Datapoint Agent log: {log_file_path}")

        with open(log_file_path, 'w') as log_f:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    log_f.write(str(message))
                    log_f.write('\n')
                    if _message_indicates_rate_limit(message):
                        raise ClaudeRateLimitError(
                            "Claude rate limit/quota reached (e.g. out of extra usage)."
                        )

        return DatapointResult(
            task_id_evol=task_id_evol,
            metadata={"evol_task_path": evol_task_path, "log_file": log_file_path}
        )


# --- Judge Agent ---

class ClaudeJudgeAgent(JudgeAgent):
    def __init__(self):
        pass

    async def judge(
        self,
        task_id_evol: str,
        evol_task_path: str,
        datapoint_result: DatapointResult,
        **kwargs
    ) -> SynthResult:
        judge_report_path = os.path.join(evol_task_path, "judge_report.md")
        log_file_path = os.path.join(evol_task_path, "judge_agent_log.txt")

        # Load judge prompt template
        prompt_path = os.path.join(os.path.dirname(__file__), "judge_agent_prompt.md")
        with open(prompt_path, 'r') as f:
            prompt_tmpl = f.read()

        oracle_status = "oracle_passed" if datapoint_result.oracle_passed else "oracle_failed"
        empty_status = "empty_failed (correct)" if datapoint_result.empty_failed else "empty_passed (BAD - tests too permissive)"

        # Preload all task files — judge writes only judge_report.md, needs no Read calls
        preloaded_files = _preload_task_files(evol_task_path)

        prompt = prompt_tmpl.format(
            task_id_evol=task_id_evol,
            evol_task_path=evol_task_path,
            preloaded_files=preloaded_files,
        ) + f"\n\nValidation run results: {oracle_status}, {empty_status}"

        # Judge can only write judge_report.md
        # judge_report_path is already absolute (built from absolute evol_task_path)
        _abs_report = os.path.abspath(judge_report_path)
        _abs_cwd = os.path.abspath(evol_task_path)
        async def validate_judge_write(input_data: Dict[str, Any], tool_use_id: str | None, context: HookContext) -> Dict[str, Any]:
            if input_data['tool_name'] in ['Write', 'Edit']:
                file_path = input_data['tool_input'].get('file_path', '')
                # Resolve relative paths against the agent's cwd, not the process cwd
                if os.path.isabs(file_path):
                    abs_path = os.path.normpath(file_path)
                else:
                    abs_path = os.path.normpath(os.path.join(_abs_cwd, file_path))
                if abs_path != _abs_report:
                    return {
                        'hookSpecificOutput': {
                            'hookEventName': 'PreToolUse',
                            'permissionDecision': 'deny',
                            'permissionDecisionReason': f'Judge can only write to {_abs_report}, not {file_path}.'
                        }
                    }
            return {}

        options = ClaudeAgentOptions(
            model="claude-opus-4-6",
            permission_mode='acceptEdits',
            cwd=evol_task_path,
            env=os.environ.copy(),
            allowed_tools=["Write"],
            hooks={
                'PreToolUse': [
                    HookMatcher(matcher='Edit|Write', hooks=[validate_judge_write]),
                ]
            }
        )

        print(f"[{task_id_evol}] Judge Agent evaluating: {evol_task_path}")
        print(f"[{task_id_evol}] Judge Agent log: {log_file_path}")

        with open(log_file_path, 'w') as log_f:
            log_f.write(f"Prompt:\n{prompt}\n\n")
            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    log_f.write(str(message))
                    log_f.write('\n')
                    if _message_indicates_rate_limit(message):
                        raise ClaudeRateLimitError(
                            "Claude rate limit/quota reached (e.g. out of extra usage)."
                        )

        # Parse verdict from judge_report.md
        verdict = "FAIL"
        feedback = ""
        issues = []
        if os.path.exists(judge_report_path):
            with open(judge_report_path, 'r') as f:
                report_content = f.read()
            if "## Verdict: PASS" in report_content:
                verdict = "PASS"
            # Extract feedback section
            if "## Feedback for Datapoint Agent" in report_content:
                feedback_idx = report_content.index("## Feedback for Datapoint Agent")
                feedback = report_content[feedback_idx:]
            # Extract FAIL criteria as issues
            for line in report_content.splitlines():
                if ": FAIL" in line:
                    issues.append(line.strip())

        return SynthResult(
            task_id_evol=task_id_evol,
            verdict=verdict,
            feedback=feedback,
            issues=issues,
            metadata={"evol_task_path": evol_task_path, "log_file": log_file_path, "report": judge_report_path}
        )


