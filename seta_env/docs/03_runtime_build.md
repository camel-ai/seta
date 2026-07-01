# Plan 03 — DockerHarborRuntime build() Method

## Source
`seta_env/runtimes/docker_harbor_runtime.py`  (modify existing)

## Depends On
- Stage 02 (RemoteDockerEnvironment) passing

## What Changes
Add a `build()` method to `DockerHarborRuntime` that calls `harbor_env.build()` if
the underlying environment supports it (i.e., `RemoteDockerEnvironment`), and skips
silently for local `DockerEnvironment` which builds inside `start()`.

This lets the training workflow do:
```python
# Phase 1: build once (before any trajectories start)
await runtime.build()

# Phase 2: start containers (no rebuild needed)
await runtime.reset(force_build=False)
```

## New Method Signature

```python
# seta_env/runtimes/docker_harbor_runtime.py

async def build(self) -> None:
    """
    Pre-build the Docker image without starting the container.

    For RemoteDockerEnvironment: uploads context and runs docker build on the
    remote node. Should be called once per task before reset() to avoid
    concurrent build conflicts when multiple trajectories start simultaneously.

    For local DockerEnvironment: no-op (build happens inside reset()).
    For other environments (Daytona, Modal): no-op.
    """
    if hasattr(self.harbor_env, "build"):
        await self.harbor_env.build()
```

## Why Separate build() from reset()

Without this, calling `reset()` concurrently for N trajectories of the same task
would trigger N concurrent `docker build` commands for the same image — causing
race conditions and wasted work. With `build()` called once first:

```
# Bad (N=4 trajectories):
asyncio.gather(reset(), reset(), reset(), reset())
  → 4 concurrent builds of same image → race condition

# Good:
await runtime.build()           # once
asyncio.gather(reset(), reset(), reset(), reset())
  → 4 concurrent compose-up with pre-built image → safe
```

## Modified `reset()` behavior

`reset()` already accepts `force_build: bool`. After this change:
- `force_build=False` (default): `compose up` only, uses pre-built image
- `force_build=True`: rebuilds + starts (for local docker or forced remote rebuild)

No changes needed to `reset()` itself — the separation is enforced at the caller level.

---

## Test Script
`seta_env/test/test_runtime_build.py`

Run: `python seta_env/test/test_runtime_build.py`

## Dependencies
- Node Manager running on `95.133.253.67:8001`
- `NODE_MANAGER_URL`, `NODE_MANAGER_API_KEY` env vars
- Task: `<REPO_ROOT>/dataset/seta-env-harbor/0`

## Script Structure

```python
"""Test: DockerHarborRuntime.build()
Run: python test_runtime_build.py
Env: NODE_MANAGER_URL, NODE_MANAGER_API_KEY
"""
import asyncio, os, uuid
from pathlib import Path
from seta_env.runtimes.docker_harbor_runtime import DockerHarborRuntime
from harbor.environments.docker.remote_docker_environment import RemoteDockerEnvironment
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths

TASK_DIR   = Path("<REPO_ROOT>/dataset/seta-env-harbor/0")
NODE_URL   = os.environ["NODE_MANAGER_URL"]
API_KEY    = os.environ["NODE_MANAGER_API_KEY"]
TRIAL_ROOT = Path("<REPO_ROOT>/seta_env/test/output/trials")

def make_remote_runtime(session_id=None) -> DockerHarborRuntime:
    task = Task(TASK_DIR)
    sid = session_id or f"rb_{uuid.uuid4().hex[:8]}"
    trial_paths = TrialPaths(trial_dir=TRIAL_ROOT / sid)
    trial_paths.mkdir()
    remote_env = RemoteDockerEnvironment(
        node_manager_url=NODE_URL,
        api_key=API_KEY,
        environment_dir=task.paths.environment_dir,
        environment_name=task.name,
        session_id=sid,
        trial_paths=trial_paths,
        task_env_config=task.config.environment,
    )
    return DockerHarborRuntime(environment=remote_env)


async def test_build_delegates():
    rt = make_remote_runtime()
    await rt.build()   # no exception = delegates to harbor_env.build()
    print("PASS test_build_delegates")


async def test_build_then_reset():
    rt = make_remote_runtime()
    try:
        await rt.build()
        await rt.reset()
        r = await rt.harbor_env.exec("echo ok")
        assert r.return_code == 0
    finally:
        await rt.stop(delete=True)
    print("PASS test_build_then_reset")


async def test_build_idempotent():
    rt = make_remote_runtime()
    try:
        await rt.build()
        await rt.build()  # second call — docker cache hit, no error
    finally:
        await rt.stop(delete=False)
    print("PASS test_build_idempotent")


async def test_local_build_noop():
    rt = DockerHarborRuntime(
        task_dir=str(TASK_DIR),
        trial_root=str(TRIAL_ROOT),
        session_id=f"local_{uuid.uuid4().hex[:8]}",
        environment_type="docker",
    )
    await rt.build()   # must complete without error (no-op for local)
    try:
        await rt.reset()
        r = await rt.harbor_env.exec("echo local_ok")
        assert r.return_code == 0
    finally:
        await rt.stop(delete=True)
    print("PASS test_local_build_noop")


async def test_concurrent_resets_after_one_build():
    # Build once, then start 4 containers simultaneously from the same image
    build_rt = make_remote_runtime()
    await build_rt.build()

    runtimes = [make_remote_runtime() for _ in range(4)]
    try:
        await asyncio.gather(*[rt.reset(force_build=False) for rt in runtimes])
        results = await asyncio.gather(*[rt.harbor_env.exec("echo ok") for rt in runtimes])
        for r in results:
            assert r.return_code == 0
    finally:
        await asyncio.gather(*[rt.stop(delete=True) for rt in runtimes])
    print("PASS test_concurrent_resets_after_one_build (4 containers)")


async def main():
    TRIAL_ROOT.mkdir(parents=True, exist_ok=True)
    await test_build_delegates()
    await test_build_then_reset()
    await test_build_idempotent()
    await test_local_build_noop()
    await test_concurrent_resets_after_one_build()
    print("\nAll runtime build tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
```

## Test Cases

| Test | Checks |
|---|---|
| `test_build_delegates` | `build()` on remote runtime completes without exception |
| `test_build_then_reset` | build → reset → exec works end-to-end |
| `test_build_idempotent` | second `build()` call succeeds (docker cache) |
| `test_local_build_noop` | `build()` on local docker runtime is a no-op; `reset()` still works |
| `test_concurrent_resets_after_one_build` | 4 concurrent `reset(force_build=False)` succeed after single build |

## Setup Notes

- Always `stop(delete=True)` in finally blocks.
- `build()` is slow (60–300s). Run `test_build_delegates` first to warm the cache.
- Local docker tests require local Docker daemon.
