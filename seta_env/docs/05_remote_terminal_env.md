# Plan 05 — Remote TerminalEnvironment (Single Trajectory)

## Source
`seta_env/runtimes/docker_harbor_runtime.py`  (modify existing)
`seta_env/environments/terminal_env.py`       (no changes needed)

## Depends On
- Stage 02 (RemoteDockerEnvironment) passing
- Stage 03 (runtime build() method) passing

## What This Stage Does

Extends `DockerHarborRuntime.__init__` to accept `environment_type="remote_docker"`,
which constructs a `RemoteDockerEnvironment` internally — the same pattern already
used for local `"docker"`.

Callers use the same runtime interface regardless of whether execution is local or remote:

```python
# Local docker (unchanged)
runtime = DockerHarborRuntime(
    task_dir=str(TASK_DIR),
    trial_root=str(TRIAL_ROOT),
    session_id=session_id,
    environment_type="docker",
)

# Remote docker (new)
runtime = DockerHarborRuntime(
    task_dir=str(TASK_DIR),
    trial_root=str(TRIAL_ROOT),
    session_id=session_id,
    environment_type="remote_docker",
    node_manager_url="http://95.133.253.67:8001",
    node_api_key=API_KEY,
)
```

`TerminalEnvironment` receives the runtime via `runtime_config` as usual:

```python
runtime_config = {
    "task_dir":           str(TASK_DIR),
    "trial_root":         str(TRIAL_ROOT),
    "session_id":         session_id,
    "environment_type":   "remote_docker",
    "node_manager_url":   NODE_URL,
    "node_api_key":       API_KEY,
}
te = TerminalEnvironment(agent_config, model_config, runtime_config, env_config)
run_info, reward = await te.step(task, uid=uid, traj_i=0)
```

## Changes to `DockerHarborRuntime.__init__`

Add `"remote_docker"` to the allowed `environment_type` values, and create
`RemoteDockerEnvironment` in the same branch:

```python
assert environment_type in [
    EnvironmentType.DOCKER.value,
    EnvironmentType.DAYTONA.value,
    EnvironmentType.MODAL.value,
    "remote_docker",
], f"Unsupported environment type: {environment_type}"

if environment_type == "remote_docker":
    node_manager_url = kwargs.pop("node_manager_url")
    node_api_key     = kwargs.pop("node_api_key")
    self.harbor_env = RemoteDockerEnvironment(
        node_manager_url=node_manager_url,
        api_key=node_api_key,
        environment_name=self._task.name,
        session_id=session_id,
        trial_paths=self._trial_paths,
        task_env_config=self._task.config.environment,
    )
else:
    self.harbor_env = EnvironmentFactory.create_environment(
        type=environment_type,
        environment_dir=self._task.paths.environment_dir,
        environment_name=self._task.name,
        session_id=session_id,
        trial_paths=self._trial_paths,
        task_env_config=self._task.config.environment,
        logger=self._logger,
    )
```

## Sequence Inside `TerminalEnvironment.step()`

```
1_reset_env:
    DockerHarborRuntime(environment_type="remote_docker", ...)
    runtime.reset()         → remote_env.start()
                              → node_manager POST /compose/up
    runtime.get_tools()     → TerminalToolkit wrapping remote exec
    Verifier(task, trial_paths, remote_env)

2_run_agent:
    agent.astep(instruction)
    → tool calls → shell_exec(cmd)
                 → remote_env.exec(cmd)
                 → node_manager POST /exec

3_evaluate:
    verifier.verify()
    → remote_env.upload_dir(tests_dir, "/tests")
    → remote_env.exec("bash /tests/test.sh ...")
    → remote_env.download_dir("/logs/verifier", trial_paths.verifier_dir)
    → reads trial_paths.reward_text_path locally

4_calculate_reward:
    reward_factory(evaluation_results)

5_close:
    runtime.stop(delete=True)
    → remote_env.stop()
    → node_manager POST /compose/down
```

---

## Test Script
`seta_env/test/test_remote_terminal_env.py`

Run: `python seta_env/test/test_remote_terminal_env.py`

## Dependencies
- Node Manager on `95.133.253.67:8001`
- `NODE_MANAGER_URL`, `NODE_MANAGER_API_KEY`, `ANTHROPIC_API_KEY` env vars set
- Task: `<REPO_ROOT>/dataset/seta-env-harbor/0`

## Script Structure

```python
"""Test: TerminalEnvironment with remote_docker runtime (full step)
Run: python test_remote_terminal_env.py
Env: NODE_MANAGER_URL, NODE_MANAGER_API_KEY, ANTHROPIC_API_KEY
"""
import asyncio, os, uuid
from pathlib import Path
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from seta_env.environments.terminal_env import TerminalEnvironment

TASK_DIR   = Path("<REPO_ROOT>/dataset/seta-env-harbor/0")
NODE_URL   = os.environ["NODE_MANAGER_URL"]
API_KEY    = os.environ["NODE_MANAGER_API_KEY"]
TRIAL_ROOT = Path("<REPO_ROOT>/seta_env/test/output/trials")

AGENT_CONFIG = {
    "system_message": "You are a developer agent. Use shell tools to complete the task.",
    "max_total_tokens": 8000,
    "max_iteration": 5,
    "working_directory": "/workdir",
    "tool_names": ["shell_exec", "shell_write_content_to_file", "shell_view"],
}
MODEL_CONFIG = {
    "model": ModelFactory.create(
        model_platform=ModelPlatformType.ANTHROPIC,
        model_type="claude-haiku-4-5-20251001",
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
        "session_id":       f"rte_{uuid.uuid4().hex[:8]}",
        "environment_type": "remote_docker",
        "node_manager_url": node_url or NODE_URL,
        "node_api_key":     API_KEY,
    }


async def test_happy_path():
    te = TerminalEnvironment(AGENT_CONFIG, MODEL_CONFIG, make_runtime_config(), ENV_CONFIG)
    run_info, reward = await te.step(TASK, uid="uid_001", traj_i=0)
    assert run_info["error_info"] == {}
    assert len(run_info["timings"]) == 5
    assert reward is not None
    assert 0.0 <= reward <= 1.0
    print(f"PASS test_happy_path (reward={reward:.2f})")


async def test_stage1_failure_bad_url():
    te = TerminalEnvironment(
        AGENT_CONFIG, MODEL_CONFIG,
        make_runtime_config(node_url="http://0.0.0.0:9999"),
        ENV_CONFIG,
    )
    run_info, reward = await te.step(TASK, uid="uid_fail", traj_i=0)
    assert run_info["error_info"].get("stage") == "1_reset_env"
    assert reward is None
    print("PASS test_stage1_failure_bad_url")


async def test_timings_include_remote_latency():
    te = TerminalEnvironment(AGENT_CONFIG, MODEL_CONFIG, make_runtime_config(), ENV_CONFIG)
    run_info, _ = await te.step(TASK, uid="uid_timing", traj_i=0)
    reset_elapsed = run_info["timings"]["1_reset_env"]["elapsed"]
    eval_elapsed  = run_info["timings"]["3_evaluate"]["elapsed"]
    assert reset_elapsed > 1.0, f"reset_elapsed={reset_elapsed}"
    assert eval_elapsed  > 1.0, f"eval_elapsed={eval_elapsed}"
    print("PASS test_timings_include_remote_latency")


async def main():
    TRIAL_ROOT.mkdir(parents=True, exist_ok=True)
    await test_stage1_failure_bad_url()   # fast, no API call
    await test_happy_path()               # slow — full agent run
    await test_timings_include_remote_latency()
    print("\nAll remote terminal env tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
```

## Test Cases

| Test | Checks |
|---|---|
| `test_stage1_failure_bad_url` | unreachable node → `error_info.stage="1_reset_env"`, `reward=None` |
| `test_happy_path` | full `step()` completes, `error_info={}`, reward in [0,1] |
| `test_timings_include_remote_latency` | `1_reset_env.elapsed > 1s`, `3_evaluate.elapsed > 1s` |

## Setup Notes

- Run `test_stage1_failure_bad_url` first (no API call, instant).
- Full step takes 60–300s. Requires `ANTHROPIC_API_KEY`.
- Use `claude-haiku-4-5-20251001` to minimize cost.
- No dangling containers: `stop(delete=True)` is called inside `TerminalEnvironment`'s `5_close` stage.
- The runtime creates its own session_id-scoped trial_paths internally.
