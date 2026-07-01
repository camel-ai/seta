# Plan 01 — Node Manager

## Source
`seta_env/environments/node_manager.py`

## What It Does
FastAPI service deployed on each cloud node. Wraps local `docker compose` commands
and exposes them over HTTP with API key auth. The training machine calls this service
to ensure datasets are present, build images, start/stop containers, exec commands,
and transfer verifier output back.

**Key principle:** datasets live on the node. `/build` reads from the local dataset
root — no context tar upload needed. This avoids large file transfers per task.

## Concurrency Constraints
- Max **16 concurrent builds** per node — enforced by `asyncio.Semaphore(16)`.
- No limit on concurrent `compose/up` — isolated by `session_id`.
- Dataset download is serialized per dataset name (one download at a time per dataset).

## Dataset Setup

Each node has a `DATASET_ROOT` directory (e.g. `/data/harbor/dataset`). Datasets are
sub-directories inside it, matching the names from the local machine:

```
/data/harbor/dataset/
├── seta-env-harbor/          # task 0, 1, 2, ...
├── terminal-bench-2.0/
├── terminal-bench-core_migrated/
└── ...
```

Datasets are downloaded on-demand via `POST /dataset/ensure`, which runs the
equivalent of `download_data.sh` for the requested dataset. The download script
is copied to the node alongside `node_manager.py`.

## Deployment (one-time via SSH)

```bash
# On 95.133.253.67:
pip install fastapi "uvicorn[standard]" httpx aiofiles python-multipart

# Copy node_manager.py + datasets.yaml to the server, then:
export NODE_MANAGER_API_KEY=<shared-secret>
export DATASET_ROOT=/data/harbor/dataset
mkdir -p $DATASET_ROOT
uvicorn node_manager:app --host 0.0.0.0 --port 8001
```

As a systemd service (`/etc/systemd/system/node-manager.service`):
```ini
[Unit]
Description=Node Manager
After=network.target docker.service

[Service]
ExecStart=uvicorn node_manager:app --host 0.0.0.0 --port 8001
Environment=NODE_MANAGER_API_KEY=<secret>
Environment=DATASET_ROOT=/data/harbor/dataset
Restart=always

[Install]
WantedBy=multi-user.target
```

## Active Dataset State

The node manager holds an **active dataset name** in memory (e.g. `"seta-env-harbor"`).
It is set via `POST /setup`. All subsequent `/build` and `/compose/up` requests use only
`task_name` — the node resolves the full path as `$DATASET_ROOT/<active_dataset>/<task_name>/`.

To switch datasets during training: call `POST /setup` again with the new dataset name.
The service re-downloads if needed, then updates the active dataset.

## API Key Auth
Every request must include header `X-API-Key: <secret>`.
Missing or wrong key → HTTP 403.
Configured via env var `NODE_MANAGER_API_KEY` on startup.

## Endpoints

### `GET /health`
Returns `{"status": "ok", "node": "<hostname>", "active_dataset": "seta-env-harbor",
"datasets": ["seta-env-harbor", ...]}`.
`datasets` lists sub-directories present under `DATASET_ROOT`. No auth required.

---

### `POST /setup`
Set the active dataset. Downloads it to the node if not already present.
This is the only endpoint that takes a `dataset_name`.

**Request JSON:**
```json
{
  "dataset_name": "seta-env-harbor"
}
```

**Behavior:**
1. Check if `$DATASET_ROOT/<dataset_name>/` exists and is non-empty.
2. If missing → acquire a per-dataset download lock, run the download, release lock.
3. Set `active_dataset = dataset_name` in server state.

Download uses `datasets.yaml` (copied to node alongside `node_manager.py`):
```yaml
datasets:
  seta-env-harbor:
    repo: "https://github.com/Michaelsqj/tbench_data_converted.git"
    subfolder: "seta-env-harbor"   # move only this subfolder; null = whole repo
  terminal-bench-2.0:
    repo: "https://github.com/..."
    subfolder: null
```

**Response:**
```json
{"dataset_name": "seta-env-harbor", "already_present": true, "success": true}
```

---

### `POST /build`
Build a Docker image for a task using the active dataset.

**Request JSON:**
```json
{"task_name": "0"}
```

**Behavior:**
1. Resolve context dir: `$DATASET_ROOT/<active_dataset>/<task_name>/environment/`
2. Resolve compose file: `<context_dir>/docker-compose-build.yaml` (or Harbor template)
3. Acquire build semaphore (max 16 concurrent)
4. Run `docker compose -p build-<task_name> -f <compose_file> build`
5. Release semaphore

**Response:**
```json
{"success": true, "image_name": "hb__0", "logs": "..."}
```
On failure: `{"success": false, "logs": "..."}` with HTTP 500.

---

### `POST /compose/up`
Start a container for one trajectory.

**Request JSON:**
```json
{
  "session_id":  "task0_traj2_abc123",
  "task_name":   "0",
  "env_vars":    {"MAIN_IMAGE_NAME": "hb__0", "CPUS": "1", "MEMORY": "2048M", ...}
}
```

**Behavior:**
1. Resolve task dir: `$DATASET_ROOT/<active_dataset>/<task_name>/`
2. Create trial dirs: `/tmp/harbor/trials/<session_id>/verifier` and `.../agent`
3. Use compose file from: `<task_dir>/environment/docker-compose-prebuilt.yaml`
   (or Harbor template)
4. Set `HOST_VERIFIER_LOGS_PATH`, `HOST_AGENT_LOGS_PATH`, `TEST_DIR` (→ local
   `<task_dir>/tests/`) from dataset root
5. Run `docker compose -p <session_id> up -d`

**Response:**
```json
{"success": true, "session_id": "task0_traj2_abc123"}
```

---

### `POST /compose/down`
Stop and optionally remove a container.

**Request JSON:**
```json
{"session_id": "task0_traj2_abc123", "delete": false}
```

**Behavior:**
- `delete=false`: `docker compose down`
- `delete=true`: `docker compose down --rmi all --volumes --remove-orphans`

**Response:**
```json
{"success": true}
```

---

### `POST /exec`
Run a command inside the running container.

**Request JSON:**
```json
{
  "session_id": "task0_traj2_abc123",
  "command":    "echo hello",
  "cwd":        "/workdir",
  "env":        {"FOO": "bar"},
  "timeout_sec": 30
}
```

**Behavior:**
Runs `docker compose -p <session_id> exec [-w cwd] [-e key=val] main bash -c <command>`.

**Response:**
```json
{"stdout": "hello\n", "stderr": "", "return_code": 0}
```

---

### `POST /upload`
Upload a file or directory into the running container.

**Request** (multipart/form-data):
```
session_id:  str
target_path: str          # path inside container, e.g. "/tests"
content_tar: UploadFile   # tarball of the directory/file to upload
```

**Behavior:**
1. Extract tarball to `/tmp/harbor/uploads/<session_id>/<uuid>/`
2. Run `docker compose -p <session_id> cp <extracted_path> main:<target_path>`

**Response:**
```json
{"success": true}
```

---

### `GET /download/{session_id}`
Download the verifier output directory from the container.

**Query param:** `source_path` (default: `/logs/verifier`) — path inside container.

**Behavior:**
1. Run `docker compose -p <session_id> cp main:<source_path> /tmp/harbor/downloads/<session_id>/`
2. Return as a `.tar.gz` stream

**Response:** `application/octet-stream` tarball.

---

## Implementation Skeleton

```python
# seta_env/environments/node_manager.py

import asyncio, os, shutil, subprocess, yaml
from pathlib import Path
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Node Manager")
API_KEY      = os.environ.get("NODE_MANAGER_API_KEY", "")
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "/data/harbor/dataset"))
HARBOR_ROOT  = Path("/tmp/harbor")
BUILD_SEMAPHORE = asyncio.Semaphore(16)

# per-dataset download locks: prevents two concurrent downloads of same dataset
_DATASET_LOCKS: dict[str, asyncio.Lock] = {}

def _dataset_lock(name: str) -> asyncio.Lock:
    if name not in _DATASET_LOCKS:
        _DATASET_LOCKS[name] = asyncio.Lock()
    return _DATASET_LOCKS[name]

def _load_datasets_config() -> dict:
    cfg_path = Path(__file__).parent / "datasets.yaml"
    return yaml.safe_load(cfg_path.read_text())["datasets"]

def check_auth(x_api_key: str):
    if x_api_key != API_KEY:
        raise HTTPException(403, "Invalid API key")

async def run_subprocess(cmd: list[str], cwd: Path | None = None,
                         env: dict | None = None,
                         timeout: int | None = None) -> tuple[int, str]:
    """Run command, capture combined stdout+stderr, return (rc, output)."""
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=cwd, env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    return proc.returncode, stdout.decode()


class SetupRequest(BaseModel):
    dataset_name: str

class BuildRequest(BaseModel):
    task_name: str

class ComposeUpRequest(BaseModel):
    session_id: str
    task_name: str
    env_vars: dict[str, str] = {}

class ComposeDownRequest(BaseModel):
    session_id: str
    delete: bool = False

class ExecRequest(BaseModel):
    session_id: str
    command: str
    cwd: str | None = None
    env: dict[str, str] | None = None
    timeout_sec: int | None = None


_active_dataset: str | None = None   # set by POST /setup

@app.get("/health")
async def health():
    datasets = [d.name for d in DATASET_ROOT.iterdir() if d.is_dir()] if DATASET_ROOT.exists() else []
    return {"status": "ok", "node": os.uname().nodename,
            "active_dataset": _active_dataset, "datasets": datasets}

@app.post("/setup")
async def setup(req: SetupRequest, x_api_key: str = Header(...)): ...
# downloads dataset if missing, sets _active_dataset

@app.post("/build")
async def build(req: BuildRequest, x_api_key: str = Header(...)): ...

@app.post("/compose/up")
async def compose_up(req: ComposeUpRequest, x_api_key: str = Header(...)): ...

@app.post("/compose/down")
async def compose_down(req: ComposeDownRequest, x_api_key: str = Header(...)): ...

@app.post("/exec")
async def exec_cmd(req: ExecRequest, x_api_key: str = Header(...)): ...

@app.post("/upload")
async def upload(...): ...

@app.get("/download/{session_id}")
async def download(session_id: str, source_path: str = "/logs/verifier",
                   x_api_key: str = Header(...)): ...
```

## `datasets.yaml` (copied to node alongside `node_manager.py`)

```yaml
datasets:
  seta-env-harbor:
    repo: "https://github.com/Michaelsqj/tbench_data_converted.git"
    subfolder: "seta-env-harbor"   # move only this subfolder; null = move entire repo
  terminal-bench-2.0:
    repo: "https://github.com/..."
    subfolder: null
  terminal-bench-core_migrated:
    repo: "https://github.com/..."
    subfolder: null
```

The `subfolder` field handles repos that contain multiple datasets — only the
relevant subdirectory is moved to `DATASET_ROOT`.

---

## Test Script
`seta_env/test/test_node_manager.py`

Run: `python seta_env/test/test_node_manager.py`

## Dependencies
- Node manager running on `95.133.253.67:8001`
- `NODE_MANAGER_URL=http://95.133.253.67:8001` env var set
- `NODE_MANAGER_API_KEY=<secret>` env var set
- Task: `<REPO_ROOT>/dataset/seta-env-harbor/0`

## Test Cases

### Health check

| Scenario | Call | Expected |
|---|---|---|
| No auth needed | `GET /health` | `{"status": "ok"}` HTTP 200 |
| Auth rejected | `POST /build` with wrong key | HTTP 403 |

### Build

| Scenario | Steps | Expected |
|---|---|---|
| Happy path | Upload context tar for task 0, valid compose_yaml | `success=true`, `image_name="hb__0"` |
| Same task twice | Build task 0 twice | Second build uses cache, faster; both succeed |
| Concurrent builds ≤ 16 | Fire 16 build requests simultaneously | All succeed, semaphore allows all |
| Concurrent builds > 16 | Fire 20 build requests simultaneously | All eventually succeed; 4 wait behind semaphore |
| Bad Dockerfile | Upload context with invalid Dockerfile | `success=false`, logs contain error |

### Compose up / down

| Scenario | Steps | Expected |
|---|---|---|
| Happy path | `compose_up` with session_id="test_s1" after build | `success=true`; container running |
| Exec after up | `exec` `echo hello` in session_id="test_s1" | `stdout="hello\n"`, `return_code=0` |
| Down without delete | `compose_down(delete=false)` | Container stopped, image retained |
| Down with delete | `compose_down(delete=true)` | Container + image removed |
| Session not found | `exec` on unknown session_id | Non-zero return_code or HTTP 500 |

### File transfer

| Scenario | Steps | Expected |
|---|---|---|
| Upload dir | Upload `/tests` dir of task 0 to container `/tests` | Exec `ls /tests` shows files |
| Download verifier dir | After exec writes reward.txt, download `/logs/verifier` | Returned tarball contains `reward.txt` |
| reward.txt content | Extract tarball, read `reward.txt` | Valid float string |

## Script Structure

```python
"""Test: Node Manager HTTP API
Run: python test_node_manager.py
Env: NODE_MANAGER_URL, NODE_MANAGER_API_KEY
"""
import asyncio, os, sys, io, tarfile, uuid
import httpx
from pathlib import Path

NODE_URL = os.environ["NODE_MANAGER_URL"]
API_KEY  = os.environ["NODE_MANAGER_API_KEY"]
TASK_DIR = Path("<REPO_ROOT>/dataset/seta-env-harbor/0")

def make_client(timeout=300) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=NODE_URL, headers={"X-API-Key": API_KEY}, timeout=timeout)

def make_context_tar() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(TASK_DIR / "environment", arcname=".")
    buf.seek(0)
    return buf.read()

def sid(prefix="nm") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


async def test_health():
    async with httpx.AsyncClient(base_url=NODE_URL, timeout=10) as c:
        r = await c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
    print("PASS test_health")


async def test_auth_rejected():
    async with httpx.AsyncClient(base_url=NODE_URL, timeout=10) as c:
        r = await c.post("/build", data={"task_name": "x"}, headers={"X-API-Key": "wrong"})
        assert r.status_code == 403
    print("PASS test_auth_rejected")


async def test_setup():
    """Set active dataset (downloads if missing)."""
    async with make_client(timeout=600) as c:
        r = await c.post("/setup", json={"dataset_name": "seta-env-harbor"})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert "already_present" in data
    print(f"PASS test_setup (already_present={data.get('already_present')})")


async def test_health_shows_active_dataset():
    async with httpx.AsyncClient(base_url=NODE_URL, timeout=10) as c:
        r = await c.get("/health")
        body = r.json()
        assert body["active_dataset"] == "seta-env-harbor"
        assert "seta-env-harbor" in body.get("datasets", [])
    print("PASS test_health_shows_active_dataset")


async def test_build_happy_path():
    """Requires active dataset already set via test_setup."""
    async with make_client(timeout=600) as c:
        r = await c.post("/build", json={"task_name": "0"})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["image_name"] == "hb__0"
    print("PASS test_build_happy_path")


async def test_build_idempotent():
    async with make_client(timeout=600) as c:
        for _ in range(2):
            r = await c.post("/build", json={"task_name": "0"})
            assert r.json()["success"] is True
    print("PASS test_build_idempotent")


async def test_compose_lifecycle():
    session = sid("lifecycle")
    async with make_client(timeout=300) as c:
        # up — no dataset_name needed, node uses active dataset
        r = await c.post("/compose/up", json={
            "session_id": session,
            "task_name": "0",
            "env_vars": {"MAIN_IMAGE_NAME": "hb__0"},
        })
        assert r.json()["success"] is True

        # exec
        r = await c.post("/exec", json={"session_id": session, "command": "echo hello"})
        data = r.json()
        assert data["return_code"] == 0
        assert "hello" in data["stdout"]

        # down
        r = await c.post("/compose/down", json={"session_id": session, "delete": True})
        assert r.json()["success"] is True
    print("PASS test_compose_lifecycle")


async def test_upload_and_download():
    session = sid("fileio")
    async with make_client(timeout=300) as c:
        await c.post("/compose/up", json={
            "session_id": session,
            "task_name": "0",
            "env_vars": {"MAIN_IMAGE_NAME": "hb__0"},
        })
        try:
            # upload tests dir
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                tar.add(TASK_DIR / "tests", arcname=".")
            buf.seek(0)
            r = await c.post("/upload", data={"session_id": session, "target_path": "/tests"},
                             files={"content_tar": ("t.tar.gz", buf.read(), "application/gzip")})
            assert r.json()["success"] is True

            # write reward.txt and download
            await c.post("/exec", json={"session_id": session,
                "command": "mkdir -p /logs/verifier && echo 0.75 > /logs/verifier/reward.txt"})
            r = await c.get(f"/download/{session}", params={"source_path": "/logs/verifier"})
            assert r.status_code == 200
            # untar and check
            buf2 = io.BytesIO(r.content)
            with tarfile.open(fileobj=buf2, mode="r:gz") as tar:
                names = tar.getnames()
            assert any("reward.txt" in n for n in names)
        finally:
            await c.post("/compose/down", json={"session_id": session, "delete": True})
    print("PASS test_upload_and_download")


async def test_concurrent_builds():
    async def build_one():
        async with make_client(timeout=600) as c:
            r = await c.post("/build", json={"dataset_name": "seta-env-harbor", "task_name": "0"})
            return r.json()["success"]
    results = await asyncio.gather(*[build_one() for _ in range(4)])
    assert all(results)
    print("PASS test_concurrent_builds (4 concurrent)")


async def main():
    await test_health()
    await test_auth_rejected()
    await test_setup()                       # set active dataset, download if missing
    await test_health_shows_active_dataset() # verify active_dataset in /health
    await test_build_happy_path()
    await test_build_idempotent()
    await test_compose_lifecycle()
    await test_upload_and_download()
    await test_concurrent_builds()
    print("\nAll node manager tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
```

## Setup Notes

- Always call `compose_down(delete=True)` in teardown to avoid leftover containers.
- Build tests are slow (60–300s). Run `test_build_happy_path` first to warm up cache.
- Concurrent build test uses 4 rather than 16 to stay within CI time budgets.
- Script exits non-zero on `AssertionError` — suitable for shell scripting.
