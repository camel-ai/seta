"""Test: RemoteDockerEnvironment
Run: python seta_env/test/test_remote_docker_environment.py
Env: NODE_MANAGER_URL, NODE_MANAGER_API_KEY
"""
import asyncio
import os
import uuid
from pathlib import Path

from harbor.environments.docker.remote_docker_environment import RemoteDockerEnvironment
from harbor.models.trial.paths import TrialPaths
from harbor.models.task.task import Task
from harbor.verifier.verifier import Verifier

_REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_DIR   = _REPO_ROOT / "dataset/seta-env-harbor/0"
NODE_URL   = os.environ["NODE_MANAGER_URL"]
API_KEY    = os.environ["NODE_MANAGER_API_KEY"]
TRIAL_ROOT = _REPO_ROOT / "seta_env/test/output/trials"
TASK_NAME  = "0"


def make_env() -> RemoteDockerEnvironment:
    task = Task(TASK_DIR)
    session_id = f"rde_{uuid.uuid4().hex[:8]}"
    trial_paths = TrialPaths(trial_dir=TRIAL_ROOT / session_id)
    trial_paths.mkdir()
    return RemoteDockerEnvironment(
        node_manager_url=NODE_URL,
        api_key=API_KEY,
        environment_name=TASK_NAME,
        session_id=session_id,
        trial_paths=trial_paths,
        task_env_config=task.config.environment,
    )


async def test_is_mounted():
    e = make_env()
    assert e.is_mounted is False
    print("PASS test_is_mounted")


async def test_build_and_start():
    e = make_env()
    try:
        await e.build()
        await e.start()
        r = await e.exec("echo alive")
        assert r.return_code == 0
        assert "alive" in (r.stdout or "")
    finally:
        await e.stop(delete=True)
    print("PASS test_build_and_start")


async def test_exec_variants():
    e = make_env()
    try:
        await e.build()
        await e.start()

        r = await e.exec("echo hello")
        assert r.return_code == 0 and "hello" in r.stdout

        r = await e.exec("pwd", cwd="/workdir")
        assert "/workdir" in r.stdout

        r = await e.exec("echo $FOO", env={"FOO": "bar"})
        assert "bar" in r.stdout

        r = await e.exec("exit 1")
        assert r.return_code != 0
    finally:
        await e.stop(delete=True)
    print("PASS test_exec_variants")


async def test_upload_and_download_reward():
    e = make_env()
    try:
        await e.build()
        await e.start()
        await e.upload_dir(TASK_DIR / "tests", "/tests")
        r = await e.exec("ls /tests")
        assert r.return_code == 0

        await e.exec("mkdir -p /logs/verifier && echo 0.5 > /logs/verifier/reward.txt")
        await e.download_dir("/logs/verifier", e.trial_paths.verifier_dir)
        reward_file = e.trial_paths.verifier_dir / "reward.txt"
        assert reward_file.exists(), f"reward.txt not found in {e.trial_paths.verifier_dir}"
        assert "0.5" in reward_file.read_text()
    finally:
        await e.stop(delete=True)
    print("PASS test_upload_and_download_reward")


async def test_verifier_compat():
    task = Task(TASK_DIR)
    e = make_env()
    try:
        await e.build()
        await e.start()
        verifier = Verifier(task=task, trial_paths=e.trial_paths, environment=e)
        result = await verifier.verify()
        assert len(result.rewards) > 0
        assert e.trial_paths.reward_text_path.exists()
    finally:
        await e.stop(delete=True)
    print("PASS test_verifier_compat")


async def main():
    TRIAL_ROOT.mkdir(parents=True, exist_ok=True)
    await test_is_mounted()
    await test_build_and_start()
    await test_exec_variants()
    await test_upload_and_download_reward()
    await test_verifier_compat()
    print("\nAll RemoteDockerEnvironment tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
