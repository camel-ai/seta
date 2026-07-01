"""Test: DockerHarborRuntime.build()
Run: python seta_env/test/test_runtime_build.py
Env: NODE_MANAGER_URL, NODE_MANAGER_API_KEY
"""
import asyncio
import os
import uuid
from pathlib import Path

from seta_env.runtimes.docker_harbor_runtime import DockerHarborRuntime
from harbor.environments.docker.remote_docker_environment import RemoteDockerEnvironment
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths

_REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_DIR   = _REPO_ROOT / "dataset/seta-env-harbor/0"
NODE_URL   = os.environ["NODE_MANAGER_URL"]
API_KEY    = os.environ["NODE_MANAGER_API_KEY"]
TRIAL_ROOT = _REPO_ROOT / "seta_env/test/output/trials"


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
    await test_concurrent_resets_after_one_build()
    print("\nAll runtime build tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
