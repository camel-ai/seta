# Plan 06 — Concurrent GRPO Rollout Orchestrator

## Source
`seta_env/environments/grpo_rollout.py`  (new file)

## Depends On
- Stage 03 (runtime build()) passing
- Stage 04 (Scheduler Service) passing
- Stage 05 (Remote TerminalEnvironment) passing

## What It Does
Orchestrates a full GRPO group rollout for one task:
1. Allocate N slots from the scheduler (all slots on the same node when possible)
2. Build the Docker image once on the assigned node
3. Run N `TerminalEnvironment.step()` calls concurrently (one per slot)
4. Collect `(run_info, reward)` results
5. Release all slots

This is the unit that `CamelRLVRWorkflow.arun_episode()` calls in Stage 07.

## Concurrency Limits
- Max **group size: 16** — enforced by scheduler
- Max **concurrent builds per node: 16** — enforced by node manager semaphore
- Current config: 1 node × 256 slots = 256 concurrent trajectories max
- At large scale (16 nodes × 256 slots = 4096 slots), the scheduler allocates greedily
  from the node with the most free slots, keeping each group co-located when possible

## Class Signature

```python
# seta_env/environments/grpo_rollout.py

class GRPORollout:
    def __init__(
        self,
        scheduler_url: str,           # e.g. "http://localhost:8000"
        node_api_key: str,            # X-API-Key for all node managers
        agent_config: dict,
        model_config: dict,
        env_config: dict,
        trial_root: str,              # local root for trial output dirs
        http_timeout: float = 300.0,
    )

    async def run(
        self,
        task: dict,                   # {"task_name", "task_path", "instruction"}
        n_trajs: int,                 # number of parallel trajectories (≤ 16)
        task_id: str | None = None,   # unique per training step; auto-generated if None
    ) -> list[tuple[dict, float | None]]:
        """
        Full GRPO group rollout. Returns list of (run_info, reward) length n_trajs.
        Always releases scheduler slots even if trajectories fail.
        """
```

## Internal Flow

```python
async def run(self, task, n_trajs, task_id=None):
    task_id = task_id or f"{task['task_name']}_{uuid.uuid4().hex[:8]}"

    # 1. Allocate slots
    assignments = await self._allocate(task_id, n_trajs)
    # assignments: [SlotAssignment(node_url, slot_id), ...]

    try:
        # 2. Setup node: set active dataset (downloads if missing), then build image once
        node_url = assignments[0].node_url
        await self._setup_node(node_url, task["dataset_name"])
        await self._build(node_url, task)

        # 3. Run N trajectories concurrently
        results = await asyncio.gather(*[
            self._run_one(assignments[i], task, traj_i=i)
            for i in range(n_trajs)
        ], return_exceptions=False)

    finally:
        # 4. Always release
        await self._release(task_id)

    return results

async def _setup_node(self, node_url: str, dataset_name: str):
    """POST /setup — sets active dataset on node, downloads if missing."""
    async with self._get_client(node_url) as c:
        r = await c.post("/setup", json={"dataset_name": dataset_name})
        r.raise_for_status()
```

## `_build()` Detail

```python
async def _build(self, node_url: str, task: dict):
    # No environment_dir or task_path needed — node reads from active dataset
    task_obj = Task(Path(task["task_path"]))   # local Task only for env_config
    remote_env = RemoteDockerEnvironment(
        node_manager_url=node_url,
        api_key=self._api_key,
        environment_name=task["task_name"],     # e.g. "0"
        session_id=f"build_{task['task_name']}", # ephemeral — only used for build
        trial_paths=TrialPaths(trial_dir=Path(self._trial_root) / "_build"),
        task_env_config=task_obj.config.environment,
    )
    runtime = DockerHarborRuntime(environment=remote_env)
    await runtime.build()
    # build() posts POST /build {"task_name": "0"} to node — no file transfer
```

## `_run_one()` Detail

```python
async def _run_one(self, assignment: SlotAssignment, task: dict, traj_i: int):
    task_obj = Task(Path(task["task_path"]))   # local, for env_config only
    session_id = f"{task['task_name']}_t{traj_i}_{uuid.uuid4().hex[:6]}"
    trial_paths = TrialPaths(trial_dir=Path(self._trial_root) / session_id)
    trial_paths.mkdir()

    remote_env = RemoteDockerEnvironment(
        node_manager_url=assignment.node_url,
        api_key=self._api_key,
        environment_name=task["task_name"],   # node resolves against active dataset
        session_id=session_id,
        trial_paths=trial_paths,
        task_env_config=task_obj.config.environment,
    )
    runtime_config = {"environment": remote_env}
    te = TerminalEnvironment(
        agent_config=self._agent_config,
        model_config=self._model_config,
        runtime_config=runtime_config,
        env_config=self._env_config,
    )
    return await te.step(task, uid=session_id, traj_i=traj_i)
```

## httpx Connection Pool

`GRPORollout` creates one shared `httpx.AsyncClient` per node URL with:
```python
limits = httpx.Limits(max_connections=256, max_keepalive_connections=64)
```
All `RemoteDockerEnvironment` instances for the same node share this client pool
(passed via constructor), avoiding connection exhaustion at 16 concurrent trajectories.

---

## Test Script
`seta_env/test/test_grpo_rollout.py`

Run: `python seta_env/test/test_grpo_rollout.py`

## Dependencies
- Scheduler Service on `localhost:8000` (started externally or by script)
- Node Manager on `95.133.253.67:8001`
- `SCHEDULER_URL`, `NODE_MANAGER_URL`, `NODE_MANAGER_API_KEY`, `ANTHROPIC_API_KEY` env vars
- Task: `<REPO_ROOT>/dataset/seta-env-harbor/0`

## Script Structure

```python
"""Test: GRPORollout — concurrent GRPO group rollout
Run: python test_grpo_rollout.py
Env: SCHEDULER_URL, NODE_MANAGER_URL, NODE_MANAGER_API_KEY, ANTHROPIC_API_KEY
"""
import asyncio, os, subprocess, time, uuid
from pathlib import Path
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from seta_env.environments.grpo_rollout import GRPORollout

TASK_DIR   = Path("<REPO_ROOT>/dataset/seta-env-harbor/0")
SCHEDULER  = os.environ.get("SCHEDULER_URL", "http://localhost:8000")
NODE_URL   = os.environ["NODE_MANAGER_URL"]
API_KEY    = os.environ["NODE_MANAGER_API_KEY"]
TRIAL_ROOT = Path("<REPO_ROOT>/seta_env/test/output/trials")

TASK = {
    "task_name": "seta-env-0",
    "task_path": str(TASK_DIR),
    "instruction": (TASK_DIR / "instruction.md").read_text(),
}

def make_rollout() -> GRPORollout:
    model = ModelFactory.create(
        model_platform=ModelPlatformType.ANTHROPIC,
        model_type="claude-haiku-4-5-20251001",
    )
    return GRPORollout(
        scheduler_url=SCHEDULER,
        node_api_key=API_KEY,
        agent_config={
            "system_message": "You are a developer agent.",
            "max_total_tokens": 8000,
            "max_iteration": 3,
            "working_directory": "/workdir",
            "tool_names": ["shell_exec", "shell_write_content_to_file"],
        },
        model_config={"model": model},
        env_config={"reward_fn": "pass_ratio"},
        trial_root=str(TRIAL_ROOT),
    )


async def test_n_trajs_1():
    rollout = make_rollout()
    results = await rollout.run(TASK, n_trajs=1)
    assert len(results) == 1
    run_info, reward = results[0]
    assert isinstance(run_info, dict)
    assert reward is None or isinstance(reward, float)
    print(f"PASS test_n_trajs_1 (reward={reward})")


async def test_n_trajs_4_concurrent():
    rollout = make_rollout()
    results = await rollout.run(TASK, n_trajs=4)
    assert len(results) == 4

    # check start times overlap — all 4 containers started within 30s of each other
    start_times = [r[0]["timings"]["1_reset_env"]["start"] for r in results]
    assert max(start_times) - min(start_times) < 30.0

    rewards = [r[1] for r in results]
    assert all(r is None or isinstance(r, float) for r in rewards)
    print(f"PASS test_n_trajs_4_concurrent (rewards={rewards})")


async def test_slots_released_after_run():
    import httpx
    rollout = make_rollout()
    await rollout.run(TASK, n_trajs=2)
    async with httpx.AsyncClient(base_url=SCHEDULER, timeout=5) as c:
        status = (await c.get("/status")).json()
    total_used = sum(n["total_slots"] - n["free_slots"] for n in status["nodes"])
    assert total_used == 0
    print("PASS test_slots_released_after_run")


async def test_n_trajs_exceeds_max():
    rollout = make_rollout()
    try:
        await rollout.run(TASK, n_trajs=17)
        assert False, "Should have raised"
    except Exception as e:
        assert "16" in str(e)
    print("PASS test_n_trajs_exceeds_max")


async def test_slots_released_on_failure():
    import httpx
    rollout = make_rollout()
    # use an unreachable node URL via bad task path to trigger failure
    bad_task = dict(TASK, task_path="/nonexistent/path")
    try:
        await rollout.run(bad_task, n_trajs=2)
    except Exception:
        pass
    async with httpx.AsyncClient(base_url=SCHEDULER, timeout=5) as c:
        status = (await c.get("/status")).json()
    total_used = sum(n["total_slots"] - n["free_slots"] for n in status["nodes"])
    assert total_used == 0, f"Slots not released: {total_used} still in use"
    print("PASS test_slots_released_on_failure")


async def main():
    TRIAL_ROOT.mkdir(parents=True, exist_ok=True)
    await test_n_trajs_exceeds_max()         # fast, no API
    await test_n_trajs_1()                   # 1 trajectory
    await test_slots_released_after_run()    # verify cleanup
    await test_n_trajs_4_concurrent()        # 4 concurrent trajectories
    await test_slots_released_on_failure()   # failure cleanup
    print("\nAll GRPO rollout tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
```

## Test Cases

| Test | Checks |
|---|---|
| `test_n_trajs_exceeds_max` | `n_trajs=17` raises with "16" in message |
| `test_n_trajs_1` | single trajectory via rollout path returns `(run_info, reward)` |
| `test_slots_released_after_run` | scheduler `/status` shows 0 slots in use after run |
| `test_n_trajs_4_concurrent` | 4 results, all start times within 30s, rewards are float or None |
| `test_slots_released_on_failure` | even on exception, scheduler releases all slots |

## Setup Notes

- Scheduler must be running before the script. Start with:
  `uvicorn seta_env.environments.scheduler_service:app --host 127.0.0.1 --port 8000`
- `n_trajs=4` test takes 120–600s and uses Claude API tokens.
- `n_trajs=16` test (not in default run) requires ≥16 slots in `nodes.yaml`.
- Always check node manager after each test to verify no dangling containers.
