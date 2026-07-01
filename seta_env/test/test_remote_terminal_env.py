"""Test: TerminalEnvironment with remote_docker runtime (full step)
Run: python seta_env/test/test_remote_terminal_env.py
Env: NODE_MANAGER_URL, NODE_MANAGER_API_KEY, ANTHROPIC_API_KEY
"""
import asyncio
import os
import uuid
from pathlib import Path

from camel.models import ModelFactory
from camel.types import ModelPlatformType

from seta_env.environments.terminal_env import TerminalEnvironment

_REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_DIR   = _REPO_ROOT / "dataset/seta-env-harbor/0"
NODE_URL   = os.environ["NODE_MANAGER_URL"]
API_KEY    = os.environ["NODE_MANAGER_API_KEY"]
TRIAL_ROOT = _REPO_ROOT / "seta_env/test/output/trials"
SGLANG_URL = os.environ.get("SGLANG_URL", "http://localhost:30000/v1")

AGENT_CONFIG = {
    "system_message": "You are a developer agent. Use shell tools to complete the task.",
    "max_total_tokens": 8000,
    "max_iteration": 5,
    "working_directory": "/workdir",
    "tool_names": ["shell_exec", "shell_write_content_to_file", "shell_view"],
}
def make_model_config() -> dict:
    return {
        "model": ModelFactory.create(
            model_platform=ModelPlatformType.SGLANG,
            model_type="qwen3-8b",
            url=SGLANG_URL,
            api_key="EMPTY",
            model_config_dict={"max_tokens": 4096, "stream": False},
        )
    }
ENV_CONFIG = {"reward_fn": "pass_ratio"}
TASK = {
    "task_name": "0",
    "task_path": str(TASK_DIR),
    "instruction": (TASK_DIR / "instruction.md").read_text(),
}


def make_runtime_config(node_url=None) -> dict:
    return {
        "task_dir":         str(TASK_DIR),
        "trial_root":       str(TRIAL_ROOT),
        "environment_type": "remote_docker",
        "node_manager_url": node_url or NODE_URL,
        "node_api_key":     API_KEY,
    }


async def test_stage1_failure_bad_url():
    """Unreachable node → error captured at 1_reset_env, reward=None."""
    te = TerminalEnvironment(
        AGENT_CONFIG, make_model_config(),
        make_runtime_config(node_url="http://0.0.0.0:9999"),
        ENV_CONFIG,
    )
    run_info, reward = await te.step(TASK, uid=f"fail_{uuid.uuid4().hex[:6]}", traj_i=0)
    assert run_info["error_info"].get("stage") == "1_reset_env", \
        f"Expected error at 1_reset_env, got: {run_info['error_info']}"
    assert reward is None
    print("PASS test_stage1_failure_bad_url")


async def test_happy_path():
    """Full step: reset → agent run → evaluate → reward."""
    te = TerminalEnvironment(AGENT_CONFIG, make_model_config(), make_runtime_config(), ENV_CONFIG)
    run_info, reward = await te.step(TASK, uid=f"happy_{uuid.uuid4().hex[:6]}", traj_i=0)

    assert run_info["error_info"] == {}, f"Unexpected error: {run_info['error_info']}"
    assert "1_reset_env"       in run_info["timings"]
    assert "2_run_agent"       in run_info["timings"]
    assert "3_evaluate"        in run_info["timings"]
    assert "4_calculate_reward" in run_info["timings"]
    assert "5_close"           in run_info["timings"]
    assert reward is not None, "Expected a numeric reward"
    assert 0.0 <= reward <= 1.0, f"Reward out of range: {reward}"
    assert run_info["agent_summary"].get("iteration_count", 0) > 0
    print(f"PASS test_happy_path (reward={reward:.3f})")


async def test_timings_include_remote_latency():
    """1_reset_env and 3_evaluate must take >1s (network round-trips)."""
    te = TerminalEnvironment(AGENT_CONFIG, make_model_config(), make_runtime_config(), ENV_CONFIG)
    run_info, _ = await te.step(TASK, uid=f"timing_{uuid.uuid4().hex[:6]}", traj_i=0)

    reset_elapsed = run_info["timings"]["1_reset_env"]["elapsed"]
    eval_elapsed  = run_info["timings"]["3_evaluate"]["elapsed"]
    assert reset_elapsed > 0.1, f"reset_elapsed too short: {reset_elapsed:.2f}s"
    assert eval_elapsed  > 0.1, f"eval_elapsed too short: {eval_elapsed:.2f}s"
    print(f"PASS test_timings_include_remote_latency "
          f"(reset={reset_elapsed:.1f}s, eval={eval_elapsed:.1f}s)")


async def main():
    TRIAL_ROOT.mkdir(parents=True, exist_ok=True)
    await test_stage1_failure_bad_url()   # fast — no API call
    await test_happy_path()               # slow — full agent run
    await test_timings_include_remote_latency()
    print("\nAll remote terminal env tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
