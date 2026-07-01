# Plan 02 — RemoteDockerEnvironment

## Source
`harbor/environments/docker/remote_docker_environment.py`  (new file)

## Depends On
- Stage 01 (Node Manager) passing

## What It Does
Implements `BaseEnvironment` by delegating all operations to a remote Node Manager
via `httpx.AsyncClient`. Replaces the local `subprocess`-based `DockerEnvironment`
for cloud node execution.

The key difference from `DockerEnvironment`:
- `is_mounted = False` — no volume mounts accessible locally; verifier must call
  `download_dir` to fetch `reward.txt` from the remote container
- All docker operations go through HTTP to the node manager

## Class Signature

```python
# harbor/environments/docker/remote_docker_environment.py

class RemoteDockerEnvironment(BaseEnvironment):
    def __init__(
        self,
        node_manager_url: str,         # e.g. "http://95.133.253.67:8001"
        api_key: str,                  # X-API-Key header value
        environment_name: str,         # task name/id, e.g. "0" — resolved by node against active dataset
        session_id: str,               # unique per trajectory, used as compose project
        trial_paths: TrialPaths,       # local trial output paths
        task_env_config: EnvironmentConfig,
        http_timeout: float = 300.0,   # per-request timeout for build/exec
        logger: logging.Logger | None = None,
        *args, **kwargs,
    )

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.DOCKER   # reuses DOCKER type

    @property
    def is_mounted(self) -> bool:
        return False                    # must download verifier dir after run

    async def build(self) -> None:
        """Upload context tar + run docker build on remote node. Call before start()."""

    async def start(self, force_build: bool = False) -> None:
        """Optionally build, then compose up on remote node."""

    async def stop(self, delete: bool = False) -> None:
        """compose down on remote node."""

    async def upload_file(self, source_path: Path | str, target_path: str): ...
    async def upload_dir(self, source_dir: Path | str, target_dir: str): ...
    async def download_file(self, source_path: str, target_path: Path | str): ...
    async def download_dir(self, source_dir: str, target_dir: Path | str): ...

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> ExecResult: ...
```

## Internal State

```python
self._client: httpx.AsyncClient       # connection-pooled, created in __init__
self._compose_yaml: str               # loaded from _DOCKER_COMPOSE_BUILD_PATH
self._image_name: str                 # f"hb__{environment_name}"
self._remote_trial_verifier_dir: str  # set after compose_up response
self._remote_trial_agent_dir: str     # set after compose_up response
```

## Compose Request

`compose_up` sends only the fields the node manager cannot infer itself:

```python
{
    "session_id": self._session_id,
    "task_name":  self._environment_name,   # e.g. "0"
    "env_vars": {
        "MAIN_IMAGE_NAME": f"hb__{self._environment_name}",
        "CPUS":            str(task_env_config.cpus),
        "MEMORY":          f"{task_env_config.memory_mb}M",
    }
}
# Node manager resolves TEST_DIR, HOST_VERIFIER_LOGS_PATH, HOST_AGENT_LOGS_PATH
# from DATASET_ROOT/<active_dataset>/<task_name>/ and /tmp/harbor/trials/<session_id>/
```

## `build()` Implementation Detail

```python
async def build(self) -> None:
    # No file upload — node resolves task from its active dataset
    response = await self._client.post("/build", json={
        "task_name": self._environment_name,   # e.g. "0"
    })
    if not response.json()["success"]:
        raise RuntimeError(f"Build failed: {response.json()['logs']}")
```

## `download_dir()` Implementation Detail

Called by `Verifier.verify()` with `source_dir = str(EnvironmentPaths.verifier_dir)`.

```python
async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
    # GET /download/<session_id>?source_path=<source_dir>
    # streams tarball response
    # extracts to target_dir locally
    response = await self._client.get(f"/download/{self._session_id}",
                                      params={"source_path": source_dir})
    _extract_tar(response.content, Path(target_dir))
```

This is what allows `reward.txt` to appear at `trial_paths.verifier_dir / "reward.txt"`
on the local machine, which the Verifier then reads.

## `exec()` Implementation Detail

```python
async def exec(self, command, cwd=None, env=None, timeout_sec=None) -> ExecResult:
    resp = await self._client.post("/exec", json={
        "session_id": self.session_id,
        "command": command,
        "cwd": cwd,
        "env": env,
        "timeout_sec": timeout_sec,
    })
    data = resp.json()
    return ExecResult(stdout=data["stdout"], stderr=data["stderr"],
                      return_code=data["return_code"])
```

---

## Test Script
`seta_env/test/test_remote_docker_environment.py`

Run: `python seta_env/test/test_remote_docker_environment.py`

## Dependencies
- Node Manager running on `95.133.253.67:8001`
- `NODE_MANAGER_URL`, `NODE_MANAGER_API_KEY` env vars set
- Task: `<REPO_ROOT>/dataset/seta-env-harbor/0`

## Script Structure

```python
"""Test: RemoteDockerEnvironment
Run: python test_remote_docker_environment.py
Env: NODE_MANAGER_URL, NODE_MANAGER_API_KEY
"""
import asyncio, os, sys, uuid, tempfile
from pathlib import Path
from harbor.environments.docker.remote_docker_environment import RemoteDockerEnvironment
from harbor.models.trial.paths import TrialPaths
from harbor.models.task.task import Task
from harbor.verifier.verifier import Verifier

TASK_DIR   = Path("<REPO_ROOT>/dataset/seta-env-harbor/0")
NODE_URL   = os.environ["NODE_MANAGER_URL"]
API_KEY    = os.environ["NODE_MANAGER_API_KEY"]
TRIAL_ROOT = Path("<REPO_ROOT>/seta_env/test/output/trials")
TASK_NAME  = "0"   # resolved by node against its active dataset

# NOTE: node must have active dataset set before running these tests:
#   curl -X POST http://<node>:8001/setup -H "X-API-Key: <key>" \
#        -d '{"dataset_name": "seta-env-harbor"}'

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
        await e.download_dir("/logs/verifier", e._trial_paths.verifier_dir)
        reward_file = e._trial_paths.verifier_dir / "reward.txt"
        assert reward_file.exists()
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
        verifier = Verifier(task=task, trial_paths=e._trial_paths, environment=e)
        result = await verifier.verify()
        assert len(result.rewards) > 0
        assert e._trial_paths.reward_text_path.exists()
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
```

## Test Cases

| Test | Checks |
|---|---|
| `test_is_mounted` | `env.is_mounted is False` |
| `test_build_and_start` | build → start → `exec("echo alive")` returns `return_code=0` |
| `test_exec_variants` | simple exec, cwd, env vars, failing command |
| `test_upload_and_download_reward` | upload tests dir, write reward.txt, download, read locally |
| `test_verifier_compat` | `Verifier.verify()` succeeds, `reward_text_path` exists |

## Setup Notes

- Each test creates its own `session_id` and `trial_paths`; `stop(delete=True)` in finally block.
- `build()` is slow (60–300s). Run once manually first to warm the cache before running the full suite.
- `TRIAL_ROOT` is created automatically if missing.
