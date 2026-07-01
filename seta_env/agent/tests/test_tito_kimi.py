#!/usr/bin/env python3
"""TITO test with Kimi K2.5 — verify KV cache hits via usage stats.

Runs the TITO agent with Kimi model in a Docker container and extracts
per-turn usage info including cached_tokens from prompt_tokens_details.

Usage:
    MOONSHOT_API_KEY=sk-... python seta_env/agent/tests/test_tito_kimi.py

Prerequisites:
    - MOONSHOT_API_KEY env var set
    - Docker daemon running
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Config ───────────────────────────────────────────────────────────────
TASK_DIR = REPO_ROOT / "dataset" / "seta-env-v2" / "stack_overflow__888"
OUTPUT_DIR = REPO_ROOT / "outputs" / "test_tito_kimi"
CAMEL_LOG_DIR = OUTPUT_DIR / "CAMEL_LOG_DIR"
TERMINAL_LOG_DIR = OUTPUT_DIR / "terminal_logs"

CONTAINER_NAME = "tito_kimi_888"
IMAGE_NAME = "tito_kimi_888_img"

MODEL_TYPE = "kimi-k2.5"
MODEL_URL = "https://api.moonshot.ai/v1"
MODEL_CONFIG = {
    "max_tokens": 4096,
    "stream": False,
    "temperature": 1.0,
}
MAX_ITERATIONS = 10


# =====================================================================
# Docker
# =====================================================================

def docker_setup():
    env_dir = TASK_DIR / "environment"
    print("[docker] Building image ...")
    subprocess.run(["docker", "build", "-t", IMAGE_NAME, str(env_dir)],
                   check=True, capture_output=True)
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", CONTAINER_NAME,
                    "--cpus", "1", "--memory", "2g",
                    IMAGE_NAME, "sleep", "infinity"], check=True)
    print(f"[docker] Container '{CONTAINER_NAME}' running.")


# =====================================================================
# Agent factory
# =====================================================================

def create_agent():
    from camel.messages import BaseMessage
    from seta_env.agent.prompt_loader import load_system_message
    from seta_env.agent.tito_train_agent import AgentTrainTITO
    from seta_env.models.tito_chat_model import TITOChatModel
    from seta_env.toolkits.terminal_toolkit_docker import TerminalToolkit

    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        print("[ERROR] MOONSHOT_API_KEY not set")
        sys.exit(1)

    # Model — TITOChatModel with Kimi
    model = TITOChatModel(
        model_type=MODEL_TYPE,
        model_config_dict=MODEL_CONFIG,
        api_key=api_key,
        url=MODEL_URL,
        tito_validate=True,
    )
    model._log_enabled = True
    model._log_dir = str(CAMEL_LOG_DIR)

    # Toolkit
    toolkit = TerminalToolkit(
        timeout=30.0,
        docker_container_name=CONTAINER_NAME,
        working_directory="/opt/pipeline",
        session_logs_dir=str(TERMINAL_LOG_DIR),
    )
    tool_names = [
        "shell_exec", "shell_view", "shell_wait",
        "shell_write_to_process", "shell_kill_process",
        "shell_write_content_to_file",
    ]
    tools = [t for t in toolkit.get_tools()
             if t.get_function_name() in tool_names]

    # Agent
    system_message = load_system_message("sys_prompt_base")
    agent = AgentTrainTITO(
        system_message=BaseMessage.make_assistant_message(
            role_name="Developer Agent",
            content=system_message,
        ),
        model=model,
        tools=tools,
        token_limit=28672 - 4096,
        max_iteration=MAX_ITERATIONS,
        task_name="stack_overflow__888",
        summarize_threshold=None,
    )
    agent.reset()
    return agent, model, toolkit


# =====================================================================
# Cache stats extraction from CAMEL logs
# =====================================================================

def extract_cache_stats(log_dir: Path) -> list[dict]:
    """Read CAMEL logs and extract per-turn usage with cache info."""
    import glob

    agent_dirs = sorted(log_dir.iterdir(), key=os.path.getmtime)
    if not agent_dirs:
        return []
    latest = agent_dirs[-1]
    files = sorted(latest.glob("conv_*.json"))

    turns = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        resp = data.get("response", {})
        usage = resp.get("usage", {})

        # Kimi returns cached_tokens at top level AND in prompt_tokens_details
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        # Try multiple locations for cached_tokens
        cached_tokens = usage.get("cached_tokens", 0)
        ptd = usage.get("prompt_tokens_details")
        if ptd and isinstance(ptd, dict):
            cached_tokens = cached_tokens or ptd.get("cached_tokens", 0)

        # Check for reasoning_content in response
        choice = resp.get("choices", [{}])[0]
        msg = choice.get("message", {})
        has_reasoning = bool(msg.get("reasoning_content"))
        has_tool_calls = bool(msg.get("tool_calls"))

        turns.append({
            "file": f.name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,
            "cache_hit_ratio": cached_tokens / max(1, prompt_tokens),
            "has_reasoning_content": has_reasoning,
            "has_tool_calls": has_tool_calls,
        })

    return turns


# =====================================================================
# Main
# =====================================================================

async def run_test():
    from camel.messages import BaseMessage

    instruction = (TASK_DIR / "instruction.md").read_text()

    agent, model, toolkit = create_agent()
    print(f"[agent] {len(agent.tool_dict)} tools, "
          f"token_limit={agent._token_limit}, "
          f"max_iter={MAX_ITERATIONS}")
    print(f"[model] {MODEL_TYPE} via {MODEL_URL}")

    print("\n" + "=" * 70)
    print("RUNNING TITO AGENT WITH KIMI K2.5")
    print("=" * 70)

    input_msg = BaseMessage.make_user_message(
        role_name="User", content=instruction,
    )
    response = await agent.astep(input_msg)

    # Agent summary
    meta = agent.meta_info_record
    print(f"\n[agent] Done: {meta['iteration_count']} iterations, "
          f"{meta['total_tool_calls']} tool calls, "
          f"reason={meta['termination_reason']}")

    # TITO session summary
    session_msgs = model._session_messages
    n_asst = sum(1 for m in session_msgs if m.get("role") == "assistant")
    reasoning_preserved = sum(
        1 for m in session_msgs
        if m.get("role") == "assistant" and m.get("reasoning_content")
    )
    think_preserved = sum(
        1 for m in session_msgs
        if m.get("role") == "assistant"
        and "<think>" in (m.get("content") or "")
    )
    print(f"[tito]  Session: {len(session_msgs)} msgs ({n_asst} assistant)")
    print(f"[tito]  reasoning_content preserved: {reasoning_preserved}/{n_asst}")
    print(f"[tito]  <think> in content: {think_preserved}/{n_asst}")

    # TITO model cache stats
    print(f"\n[tito]  Model _cache_stats:")
    print(json.dumps(model._cache_stats, indent=2))

    # Extract cache stats from CAMEL logs
    print("\n" + "=" * 70)
    print("PER-TURN CACHE ANALYSIS (from CAMEL request/response logs)")
    print("=" * 70)

    turns = extract_cache_stats(CAMEL_LOG_DIR)

    print(f"\n{'Turn':<5} {'Prompt':<10} {'Cached':<10} {'Hit%':<8} "
          f"{'Compl':<10} {'Reasoning':<10} {'ToolCalls':<10}")
    print("-" * 70)

    total_prompt = 0
    total_cached = 0
    for i, t in enumerate(turns):
        total_prompt += t["prompt_tokens"]
        total_cached += t["cached_tokens"]
        print(f"{i+1:<5} {t['prompt_tokens']:<10} {t['cached_tokens']:<10} "
              f"{t['cache_hit_ratio']:.1%}{'':>3} "
              f"{t['completion_tokens']:<10} "
              f"{'yes' if t['has_reasoning_content'] else 'no':<10} "
              f"{'yes' if t['has_tool_calls'] else 'no':<10}")

    print("-" * 70)
    overall_ratio = total_cached / max(1, total_prompt)
    print(f"{'TOTAL':<5} {total_prompt:<10} {total_cached:<10} "
          f"{overall_ratio:.1%}")

    # Verdict
    print("\n" + "=" * 70)
    if len(turns) > 1:
        # Check if turns 2+ have cache hits
        later_turns = turns[1:]
        avg_cache_ratio = (
            sum(t["cache_hit_ratio"] for t in later_turns) / len(later_turns)
            if later_turns else 0
        )
        if avg_cache_ratio > 0.5:
            print(f"CACHE VERDICT: GOOD — avg {avg_cache_ratio:.1%} cache hit on turns 2+")
        elif avg_cache_ratio > 0:
            print(f"CACHE VERDICT: PARTIAL — avg {avg_cache_ratio:.1%} cache hit on turns 2+")
        else:
            print(f"CACHE VERDICT: NO CACHE HITS on turns 2+ — check TITO setup")
    else:
        print("CACHE VERDICT: only 1 turn, cannot assess caching")
    print("=" * 70)

    # Save results
    results = {
        "meta_info": {k: str(v) for k, v in meta.items()},
        "tito_session": {
            "message_count": len(session_msgs),
            "assistant_count": n_asst,
            "reasoning_preserved": reasoning_preserved,
        },
        "cache_stats": model._cache_stats,
        "per_turn": turns,
        "overall_cache_hit_ratio": overall_ratio,
    }
    results_path = OUTPUT_DIR / "kimi_cache_results.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nResults: {results_path}")


def main():
    for d in [OUTPUT_DIR, CAMEL_LOG_DIR, TERMINAL_LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    docker_setup()
    try:
        asyncio.run(run_test())
    finally:
        print(f"\n[docker] Container left running: "
              f"docker exec -it {CONTAINER_NAME} bash")

    sys.exit(0)


if __name__ == "__main__":
    main()
