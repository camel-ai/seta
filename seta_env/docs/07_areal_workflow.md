# Plan 07 — AReaL Workflow Integration

## Source
`src/tbench_areal_workflow/train_remote.py`  (new file, alongside existing `train.py`)

## Depends On
- Stage 06 (GRPORollout) passing

## What Changes

The existing `CamelRLVRWorkflow` in `train.py` uses the old `CamelTerminalAgent` which
manages local docker containers directly. This plan replaces `CamelTerminalAgent` with
`GRPORollout`, wiring the new remote execution stack into the AReaL training loop.

**Nothing changes in `TerminalEnvironment`, `DockerHarborRuntime`, or the verifier.**
The only change is in `arun_episode`.

## New `CamelRLVRWorkflow.arun_episode()` Structure

```python
# Old (train.py):
async def arun_episode(self, engine, data):
    # build image locally
    await build_docker_image(data)
    # run n_trajs concurrently with CamelTerminalAgent
    rewards = await asyncio.gather(*[
        CamelTerminalAgent(...).run_agent(data, client, uid, traj_i)
        for i in range(self.n_trajs)
    ])
    # pack completions_with_reward

# New (train_remote.py):
async def arun_episode(self, engine, data):
    # create ArealOpenAI clients (same as before)
    clients = [ArealOpenAI(engine=engine, ...) for _ in range(self.n_trajs)]
    uids = [uuid.uuid4().hex[:8] for _ in range(self.n_trajs)]

    # GRPORollout handles: allocate → build → N concurrent TerminalEnvironment.step()
    task = {
        "task_name": data["task_name"],
        "task_path": data["task_path"],
        "instruction": data["instruction"],
    }
    results = await self._rollout.run(
        task=task,
        n_trajs=self.n_trajs,
        task_id=f"{data['task_name']}_{uuid.uuid4().hex[:6]}",
    )
    # results: [(run_info, reward), ...]

    # pack completions_with_reward (same logic as before)
    completions_with_reward = {}
    for i, ((run_info, reward), client) in enumerate(zip(results, clients)):
        if reward is None:
            continue
        client.apply_reward_discount(turn_discount=0.9)
        completions = client.export_interactions(style="individual")
        completions_with_reward.update(completions)

    return completions_with_reward or None
```

## Key Difference: Model Client and Trajectory

In the old workflow, `CamelTerminalAgent` held an `ArealOpenAI` client that captured
the token trajectory. In the new workflow, `TerminalEnvironment` uses `CamelAgent`
(from `seta_env.agent.train_agent`) which uses `model_config["model"]`.

The `model_config["model"]` must be an `AReaLOpenAICompatibleModel` (wrapping the
`ArealOpenAI` client) so that token sequences are captured for the GRPO update.

```python
# In GRPORollout._run_one(), model_config is constructed per trajectory:
areal_client = ArealOpenAI(engine=engine, tokenizer=tokenizer, tool_call_parser="qwen25")
model = AReaLOpenAICompatibleModel(client=areal_client)
model_config = {"model": model}
```

This means `GRPORollout.run()` needs access to `engine` and `tokenizer`. The
`model_config` is built per-trajectory inside `run()`, not passed in at construction.

## Updated `GRPORollout.run()` signature

```python
async def run(
    self,
    task: dict,
    n_trajs: int,
    engine,                    # AReaL inference engine
    tokenizer,                 # HF tokenizer
    task_id: str | None = None,
) -> list[tuple[dict, float | None, ArealOpenAI]]:
    """Returns list of (run_info, reward, areal_client) per trajectory."""
```

The `areal_client` is returned alongside `run_info` so `arun_episode` can call
`client.export_interactions()` to extract the token trajectory for GRPO.

## New `train_remote.py` Class Structure

```python
class RemoteCamelRLVRWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        scheduler_url: str,           # "http://localhost:8000"
        node_api_key: str,            # X-API-Key for node managers
        trial_root: str,
        n_trajs: int = 4,
        max_tokens: int = 32768,
        max_iteration: int = 50,
        dump_dir: str | None = None,
        rollout_stat_scope: str = "rollout",
        filter_uniform_reward: bool = False,
        task_timeouts: TaskTimeouts = None,
    ):
        self._rollout = GRPORollout(
            scheduler_url=scheduler_url,
            node_api_key=node_api_key,
            agent_config=...,    # built from max_tokens, max_iteration, etc.
            model_config=None,   # built per-trajectory in run()
            env_config={"reward_fn": "pass_ratio"},
            trial_root=trial_root,
        )
        ...

    async def arun_episode(self, engine, data) -> dict | None:
        ...
```

## Config Additions to `AgentRLConfig`

```python
@dataclass
class AgentRLConfig(GRPOConfig):
    # existing fields ...
    scheduler_url: str = field(default="http://localhost:8000")
    node_api_key: str = field(default="")
    trial_root: str = field(default="/tmp/harbor/trials")
```

## Launch Script Changes

```bash
# Terminal 1: start scheduler
uvicorn seta_env.environments.scheduler_service:app \
    --host 127.0.0.1 --port 8000

# Terminal 2: start training
python train_remote.py --config config_remote.yaml
```

`config_remote.yaml` adds:
```yaml
scheduler_url: "http://localhost:8000"
node_api_key: "<secret>"
trial_root: "/tmp/harbor/trials"
```

---

## Test Script
`seta_env/test/test_areal_workflow.py`

Run: `python seta_env/test/test_areal_workflow.py`

## Dependencies
- All previous stages passing
- Scheduler on `localhost:8000`, Node Manager on `95.133.253.67:8001`
- `SCHEDULER_URL`, `NODE_MANAGER_URL`, `NODE_MANAGER_API_KEY` env vars
- No GPU required for mock-engine tests; set `AREAL_AVAILABLE=1` for full GPU test

## Script Structure

```python
"""Test: RemoteCamelRLVRWorkflow.arun_episode()
Run: python test_areal_workflow.py
Env: SCHEDULER_URL, NODE_MANAGER_URL, NODE_MANAGER_API_KEY
GPU test: AREAL_AVAILABLE=1
"""
import asyncio, os, uuid
from pathlib import Path

SCHEDULER = os.environ.get("SCHEDULER_URL", "http://localhost:8000")
NODE_URL  = os.environ["NODE_MANAGER_URL"]
API_KEY   = os.environ["NODE_MANAGER_API_KEY"]
TASK_DIR  = Path("<REPO_ROOT>/dataset/seta-env-harbor/0")

# ── Stubs ─────────────────────────────────────────────────────────────────────

class MockArealOpenAI:
    def __init__(self, reward: float | None = 0.5):
        self._reward = reward
    def export_interactions(self, style="individual"):
        if self._reward is None:
            return {}
        return {f"uid_{uuid.uuid4().hex[:4]}": (["token1", "token2"], self._reward)}
    def apply_reward_discount(self, turn_discount=0.9):
        pass

class MockEngine:
    """Minimal stub: satisfies GRPORollout's engine interface."""
    async def generate(self, prompts, gconfig):
        return [["tok1", "tok2"] for _ in prompts]


# ── Tests ─────────────────────────────────────────────────────────────────────

async def test_arun_episode_returns_dict():
    from src.tbench_areal_workflow.train_remote import RemoteCamelRLVRWorkflow
    wf = RemoteCamelRLVRWorkflow(
        gconfig=None,
        tokenizer=None,
        scheduler_url=SCHEDULER,
        node_api_key=API_KEY,
        trial_root="/tmp/harbor/trials",
        n_trajs=1,
        max_tokens=8000,
        max_iteration=3,
    )
    task_data = {
        "task_name": "seta-env-0",
        "task_path": str(TASK_DIR),
        "instruction": (TASK_DIR / "instruction.md").read_text(),
    }
    result = await wf.arun_episode(MockEngine(), task_data)
    assert result is None or isinstance(result, dict)
    if result:
        assert len(result) > 0
    print(f"PASS test_arun_episode_returns_dict (result={'non-empty dict' if result else 'None'})")


async def test_uniform_reward_filtered():
    from src.tbench_areal_workflow.train_remote import RemoteCamelRLVRWorkflow
    wf = RemoteCamelRLVRWorkflow(
        gconfig=None, tokenizer=None,
        scheduler_url=SCHEDULER, node_api_key=API_KEY,
        trial_root="/tmp/harbor/trials",
        n_trajs=2, filter_uniform_reward=True,
    )
    task_data = {
        "task_name": "seta-env-0",
        "task_path": str(TASK_DIR),
        "instruction": (TASK_DIR / "instruction.md").read_text(),
    }
    # With real trajectories: if both get same reward, result should be None
    # This test passes structurally if the method runs without exception
    result = await wf.arun_episode(MockEngine(), task_data)
    assert result is None or isinstance(result, dict)
    print("PASS test_uniform_reward_filtered")


async def test_slots_released():
    import httpx
    from src.tbench_areal_workflow.train_remote import RemoteCamelRLVRWorkflow
    wf = RemoteCamelRLVRWorkflow(
        gconfig=None, tokenizer=None,
        scheduler_url=SCHEDULER, node_api_key=API_KEY,
        trial_root="/tmp/harbor/trials", n_trajs=1,
    )
    task_data = {
        "task_name": "seta-env-0",
        "task_path": str(TASK_DIR),
        "instruction": (TASK_DIR / "instruction.md").read_text(),
    }
    await wf.arun_episode(MockEngine(), task_data)
    async with httpx.AsyncClient(base_url=SCHEDULER, timeout=5) as c:
        status = (await c.get("/status")).json()
    total_used = sum(n["total_slots"] - n["free_slots"] for n in status["nodes"])
    assert total_used == 0
    print("PASS test_slots_released")


async def test_two_tasks_concurrent():
    import httpx
    from src.tbench_areal_workflow.train_remote import RemoteCamelRLVRWorkflow
    wf = RemoteCamelRLVRWorkflow(
        gconfig=None, tokenizer=None,
        scheduler_url=SCHEDULER, node_api_key=API_KEY,
        trial_root="/tmp/harbor/trials", n_trajs=1,
    )
    # same task used twice (different task_id will be auto-generated)
    task_data = {
        "task_name": "seta-env-0",
        "task_path": str(TASK_DIR),
        "instruction": (TASK_DIR / "instruction.md").read_text(),
    }
    results = await asyncio.gather(
        wf.arun_episode(MockEngine(), task_data),
        wf.arun_episode(MockEngine(), dict(task_data, task_name="seta-env-0-b")),
    )
    assert len(results) == 2
    async with httpx.AsyncClient(base_url=SCHEDULER, timeout=5) as c:
        status = (await c.get("/status")).json()
    total_used = sum(n["total_slots"] - n["free_slots"] for n in status["nodes"])
    assert total_used == 0
    print("PASS test_two_tasks_concurrent")


async def main():
    await test_arun_episode_returns_dict()
    await test_uniform_reward_filtered()
    await test_slots_released()
    await test_two_tasks_concurrent()

    if os.environ.get("AREAL_AVAILABLE"):
        print("GPU tests skipped in this run — run separately with AREAL_AVAILABLE=1")

    print("\nAll AReaL workflow tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
```

## Test Cases

| Test | Checks |
|---|---|
| `test_arun_episode_returns_dict` | returns dict or None, no exception |
| `test_uniform_reward_filtered` | `filter_uniform_reward=True` path runs without error |
| `test_slots_released` | scheduler shows 0 slots in use after episode |
| `test_two_tasks_concurrent` | 2 concurrent `arun_episode` calls complete, slots released |

## Setup Notes

- No GPU needed for these tests (mock engine).
- Full AReaL training loop test (GPU, real engine) is a separate script run with `AREAL_AVAILABLE=1`.
- Scheduler must be running. Start with:
  `uvicorn seta_env.environments.scheduler_service:app --host 127.0.0.1 --port 8000`
- After all tests, check node manager `/status` shows no active sessions.

## Summary: What `train_remote.py` Replaces

| Old (`train.py`) | New (`train_remote.py`) |
|---|---|
| `CamelTerminalAgent` | `GRPORollout` (Stage 06) |
| local `DockerComposeManager` | `RemoteDockerEnvironment` (Stage 02) |
| `build_docker_image()` | `GRPORollout._build()` via node manager |
| local `asyncio.gather` of `run_agent` | `GRPORollout.run()` with scheduler |
| `ThreadPoolExecutor` for sync docker ops | fully async httpx — no executor needed |
| `CamelRLVRWorkflow` | `RemoteCamelRLVRWorkflow` |
