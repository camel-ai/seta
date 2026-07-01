"""Test: GRPORollout — concurrent GRPO group rollout
Run: python seta_env/test/test_grpo_rollout.py
Env: NODE_MANAGER_URL, NODE_MANAGER_API_KEY
     SCHEDULER_URL  (default: http://localhost:8000)
     SGLANG_URL     (default: http://localhost:30000/v1)
"""
import asyncio
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
from camel.models import ModelFactory
from camel.types import ModelPlatformType

from seta_env.orchestrators.grpo_rollout import GRPORollout

_REPO_ROOT    = Path(__file__).resolve().parents[2]
TASK_DIR      = _REPO_ROOT / "dataset/seta-env-harbor/0"
NODE_URL      = os.environ["NODE_MANAGER_URL"]
API_KEY       = os.environ["NODE_MANAGER_API_KEY"]
SCHEDULER_URL = os.environ.get("SCHEDULER_URL", "http://localhost:8000")
SGLANG_URL    = os.environ.get("SGLANG_URL", "http://localhost:30000/v1")
TRIAL_ROOT    = _REPO_ROOT / "seta_env/test/output/trials"

TASK = {
    "task_name":    "0",
    "dataset_name": "seta-env-harbor",
    "task_path":    str(TASK_DIR),
    "instruction":  (TASK_DIR / "instruction.md").read_text(),
}

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

def make_rollout() -> GRPORollout:
    return GRPORollout(
        agent_config=AGENT_CONFIG,
        model_config=make_model_config(),
        env_config={
            "environment_type": "remote_docker",
            "scheduler_url": SCHEDULER_URL,
            "node_api_key": API_KEY,
            "reward_fn": "pass_ratio",
        },
        trial_root=str(TRIAL_ROOT),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

async def test_exceeds_max_group_size():
    """n_trajs=17 must raise with '16' in the message."""
    rollout = make_rollout()
    try:
        await rollout.run(TASK, n_trajs=17)
        assert False, "Should have raised"
    except Exception as e:
        assert "16" in str(e), f"Expected '16' in error, got: {e}"
    print("PASS test_exceeds_max_group_size")


async def test_n_trajs_1():
    """Single trajectory through the rollout path returns (run_info, reward)."""
    rollout = make_rollout()
    results = await rollout.run(TASK, n_trajs=1)
    assert len(results) == 1
    run_info, reward = results[0]
    assert isinstance(run_info, dict)
    assert run_info["error_info"] == {}, f"Unexpected error: {run_info['error_info']}"
    assert reward is None or isinstance(reward, float)
    print(f"PASS test_n_trajs_1 (reward={reward})")


async def test_slots_released_after_run():
    """Scheduler shows 0 slots in use after a successful run."""
    rollout = make_rollout()
    await rollout.run(TASK, n_trajs=1)
    async with httpx.AsyncClient(base_url=SCHEDULER_URL, timeout=5) as c:
        status = (await c.get("/status")).json()
    used = sum(n["total_slots"] - n["free_slots"] for n in status["nodes"])
    assert used == 0, f"Expected 0 used slots after run, got {used}"
    print("PASS test_slots_released_after_run")


async def test_slots_released_on_failure():
    """Scheduler slots are released even when the rollout fails."""
    rollout = make_rollout()
    bad_task = dict(TASK, task_path="/nonexistent/path/0")
    try:
        await rollout.run(bad_task, n_trajs=2)
    except Exception:
        pass
    async with httpx.AsyncClient(base_url=SCHEDULER_URL, timeout=5) as c:
        status = (await c.get("/status")).json()
    used = sum(n["total_slots"] - n["free_slots"] for n in status["nodes"])
    assert used == 0, f"Slots not released after failure: {used} still in use"
    print("PASS test_slots_released_on_failure")


async def test_n_trajs_4_concurrent():
    """4 concurrent trajectories: all return results, start times overlap."""
    rollout = make_rollout()
    results = await rollout.run(TASK, n_trajs=4)
    assert len(results) == 4

    for run_info, reward in results:
        assert isinstance(run_info, dict)
        assert reward is None or isinstance(reward, float)

    start_times = [r[0]["timings"]["1_reset_env"]["start"] for r in results]
    spread = max(start_times) - min(start_times)
    assert spread < 30.0, f"Start times too spread out ({spread:.1f}s) — not concurrent"

    rewards = [r[1] for r in results]
    print(f"PASS test_n_trajs_4_concurrent (rewards={rewards}, spread={spread:.1f}s)")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    TRIAL_ROOT.mkdir(parents=True, exist_ok=True)

    await test_exceeds_max_group_size()   # fast, no containers
    await test_n_trajs_1()                # 1 trajectory
    await test_slots_released_after_run() # verify cleanup
    await test_slots_released_on_failure()
    await test_n_trajs_4_concurrent()     # 4 parallel trajectories
    print("\nAll GRPO rollout tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
