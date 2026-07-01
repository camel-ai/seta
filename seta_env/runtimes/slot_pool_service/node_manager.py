"""Node Manager — FastAPI service deployed on each cloud node.

Wraps local docker compose operations and exposes them over HTTP with API key auth.
Datasets live on the node; /setup downloads them on demand and sets the active dataset.

Start:
    export NODE_MANAGER_API_KEY=<secret>
    export DATASET_ROOT=/data/harbor/dataset
    uvicorn node_manager:app --host 0.0.0.0 --port 8001
"""

import asyncio
import io
import json
import logging
import os
import shutil
import tarfile
import tempfile
import time

import yaml
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("node_manager")

# ── Config ────────────────────────────────────────────────────────────────────

app = FastAPI(title="Node Manager")

API_KEY      = os.environ.get("NODE_MANAGER_API_KEY", "")
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "/data/harbor/dataset"))
HARBOR_ROOT  = Path(os.environ.get("HARBOR_ROOT", "/tmp/harbor"))

_HERE           = Path(__file__).parent
_BUILD_COMPOSE  = _HERE / "docker-compose-build.yaml"
_PREBUILT_COMPOSE = _HERE / "docker-compose-prebuilt.yaml"

BUILD_SEMAPHORE = asyncio.Semaphore(16)

# per-dataset download locks — one lock per dataset name
_DATASET_LOCKS: dict[str, asyncio.Lock] = {}

# in-memory state
_active_dataset: str | None = None
_sessions: dict[str, dict] = {}     # session_id -> session state

# GC config: kill containers with harbor.managed=true that aren't tracked,
# and sessions running longer than SESSION_TTL_SEC
GC_INTERVAL_SEC = 300          # run GC every 5 minutes
SESSION_TTL_SEC = 7200         # 2 hours — force-kill sessions that never cleaned up

# Exec idempotency cache: request_id -> {result, expires_at}
_exec_cache: dict[str, dict] = {}
EXEC_CACHE_TTL = 300           # keep cached results for 5 minutes

# Async build jobs: job_id -> {status, result, started_at}
_build_jobs: dict[str, dict] = {}
BUILD_JOB_TTL = 600            # keep completed job results for 10 minutes


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitize(name: str) -> str:
    """Lowercase + replace chars invalid in docker compose project names."""
    return name.lower().replace("_", "-").replace(".", "-")


def _dataset_lock(name: str) -> asyncio.Lock:
    if name not in _DATASET_LOCKS:
        _DATASET_LOCKS[name] = asyncio.Lock()
    return _DATASET_LOCKS[name]


def _load_datasets_config() -> dict:
    cfg_path = _HERE / "datasets.yaml"
    return yaml.safe_load(cfg_path.read_text())["datasets"]


def _check_auth(x_api_key: str) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(403, "Invalid API key")


async def _run(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict | None = None,
    timeout: int | None = None,
) -> tuple[int, str]:
    """Run a subprocess, capture combined stdout+stderr, return (rc, output)."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise RuntimeError(f"Command timed out after {timeout}s: {' '.join(cmd)}")
    return proc.returncode, stdout.decode(errors="replace")


async def _run_split(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict | None = None,
    timeout: int | None = None,
) -> tuple[int, str, str]:
    """Run a subprocess with separate stdout/stderr, return (rc, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise RuntimeError(f"Command timed out after {timeout}s: {' '.join(cmd)}")
    return (
        proc.returncode,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


# ── GC ────────────────────────────────────────────────────────────────────────

async def _gc_once() -> dict:
    """Kill compose projects not tracked in _sessions and expired sessions.

    Returns a summary dict for logging / the /gc endpoint.
    """
    killed_orphans: list[str] = []
    killed_expired: list[str] = []
    errors: list[str] = []

    # 1. Find all currently running compose projects on this host
    try:
        rc, out = await _run(["docker", "compose", "ls", "--format", "json"], timeout=30)
        all_projects: set[str] = set()
        if rc == 0 and out.strip():
            for entry in json.loads(out):
                name = entry.get("Name", "")
                status = entry.get("Status", "")
                # Only consider running projects (not those already stopped/exited)
                if name and "running" in status.lower():
                    all_projects.add(name)
    except Exception as e:
        errors.append(f"docker compose ls failed: {e}")
        all_projects = set()

    # 2. Kill orphans — running but not in _sessions
    known_projects = {s["project"] for s in _sessions.values()}
    for project in all_projects - known_projects:
        try:
            rc, logs = await _run(
                ["docker", "compose", "-p", project, "down", "--timeout", "10"],
                timeout=30,
            )
            if rc == 0:
                killed_orphans.append(project)
            else:
                errors.append(f"orphan down failed for {project!r}: {logs[:200]}")
        except Exception as e:
            errors.append(f"orphan down error for {project!r}: {e}")

    # 3. Kill sessions that exceeded TTL (cleanup was never called)
    now = time.time()
    expired = [
        sid for sid, s in list(_sessions.items())
        if now - s.get("started_at", now) > SESSION_TTL_SEC
    ]
    for sid in expired:
        session = _sessions.get(sid)
        if session is None:
            continue
        try:
            rc, logs = await _run(
                ["docker", "compose", "-p", session["project"],
                 "-f", session["compose_file"], "down", "--timeout", "10"],
                env=session["env"],
                timeout=30,
            )
            _sessions.pop(sid, None)
            if rc == 0:
                killed_expired.append(sid)
            else:
                errors.append(f"expired session down failed for {sid!r}: {logs[:200]}")
        except Exception as e:
            _sessions.pop(sid, None)
            errors.append(f"expired session error for {sid!r}: {e}")

    # 4. Purge exec cache entries.
    #    Completed entries have "expires_at"; in-flight entries have "task".
    #    _run_exec_bg promotes in-flight → completed when done, so normally
    #    only completed entries accumulate here.
    expired_cache = [
        k for k, v in _exec_cache.items()
        if "expires_at" in v and now > v["expires_at"]
    ]
    for k in expired_cache:
        del _exec_cache[k]

    # 5. Purge completed/errored build jobs older than BUILD_JOB_TTL
    expired_builds = [
        k for k, v in _build_jobs.items()
        if v["status"] in ("done", "error")
        and now - v.get("started_at", now) > BUILD_JOB_TTL
    ]
    for k in expired_builds:
        del _build_jobs[k]

    summary = {
        "killed_orphans": killed_orphans,
        "killed_expired": killed_expired,
        "purged_exec_cache": len(expired_cache),
        "purged_build_jobs": len(expired_builds),
        "errors": errors,
    }
    if killed_orphans or killed_expired or expired_cache or expired_builds or errors:
        logger.warning("GC run: %s", summary)
    return summary


async def _gc_loop() -> None:
    """Background task: runs GC every GC_INTERVAL_SEC."""
    while True:
        await asyncio.sleep(GC_INTERVAL_SEC)
        try:
            await _gc_once()
        except Exception as e:
            logger.error("GC loop error: %s", e)


@app.on_event("startup")
async def _start_gc():
    asyncio.create_task(_gc_loop())


# ── Request / Response Models ─────────────────────────────────────────────────

class SetupRequest(BaseModel):
    dataset_name: str
    hf_token: str = ""

class BuildRequest(BaseModel):
    task_name: str

class ComposeUpRequest(BaseModel):
    session_id: str
    task_name:  str
    env_vars:   dict[str, str] = {}

class ComposeDownRequest(BaseModel):
    session_id: str
    delete: bool = False

class ExecRequest(BaseModel):
    session_id:  str
    command:     str
    cwd:         str | None = None
    env:         dict[str, str] | None = None
    timeout_sec: int | None = None
    request_id:  str | None = None   # idempotency key — same request_id returns cached result


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """No auth required. Returns node status and active dataset."""
    datasets = (
        [d.name for d in DATASET_ROOT.iterdir() if d.is_dir()]
        if DATASET_ROOT.exists() else []
    )
    return {
        "status": "ok",
        "node": os.uname().nodename,
        "active_dataset": _active_dataset,
        "datasets": datasets,
    }


@app.post("/setup")
async def setup(req: SetupRequest, x_api_key: str = Header(...)):
    """Set the active dataset, downloading it if not already present."""
    global _active_dataset
    _check_auth(x_api_key)

    dest = DATASET_ROOT / req.dataset_name
    already_present = dest.exists() and any(dest.iterdir())

    if not already_present:
        datasets_cfg = _load_datasets_config()
        if req.dataset_name not in datasets_cfg:
            raise HTTPException(400, f"Unknown dataset: {req.dataset_name!r}. "
                                     f"Add it to datasets.yaml.")
        cfg  = datasets_cfg[req.dataset_name]
        repo = cfg.get("repo")
        if not repo:
            raise HTTPException(400, f"Dataset {req.dataset_name!r} has no 'repo' in datasets.yaml.")
        subfolder = cfg.get("subfolder")

        async with _dataset_lock(req.dataset_name):
            # double-check after acquiring lock (another coroutine may have downloaded)
            if not (dest.exists() and any(dest.iterdir())):
                DATASET_ROOT.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Inject HF_TOKEN for huggingface.co repos
                    clone_url = repo
                    hf_token = req.hf_token or os.environ.get("HF_TOKEN", "")
                    if hf_token and "huggingface.co" in repo:
                        clone_url = repo.replace("https://", f"https://user:{hf_token}@")

                    rc, out = await _run(
                        ["git", "clone", "--depth=1", clone_url, f"{tmpdir}/repo"],
                        timeout=600,
                    )
                    if rc != 0:
                        raise HTTPException(500, f"git clone failed:\n{out}")

                    # Pull LFS objects (no-op if repo doesn't use git-lfs)
                    rc_lfs, out_lfs = await _run(
                        ["git", "lfs", "pull"],
                        cwd=Path(f"{tmpdir}/repo"),
                        timeout=600,
                    )
                    if rc_lfs != 0:
                        logger.warning("git lfs pull failed (non-fatal): %s", out_lfs[:200])

                    if subfolder:
                        shutil.move(f"{tmpdir}/repo/{subfolder}", str(dest))
                    else:
                        shutil.move(f"{tmpdir}/repo", str(dest))

    _active_dataset = req.dataset_name
    return {
        "dataset_name": req.dataset_name,
        "already_present": already_present,
        "success": True,
    }


async def _run_build(job_id: str, task_name: str, image_name: str,
                     project: str, build_compose: str, env: dict) -> None:
    """Background task that runs the actual docker build."""
    t0 = time.time()
    try:
        async with BUILD_SEMAPHORE:
            rc, logs = await _run(
                ["docker", "compose", "-p", project, "-f", build_compose, "build"],
                env=env,
                timeout=600,
            )
        elapsed = time.time() - t0
        if rc != 0:
            logger.error("build FAILED | task=%s rc=%d elapsed=%.1fs logs=%s",
                         task_name, rc, elapsed, logs[-200:])
            _build_jobs[job_id]["status"] = "error"
            _build_jobs[job_id]["result"] = {"success": False, "image_name": image_name, "logs": logs}
        else:
            logger.info("build OK | task=%s elapsed=%.1fs", task_name, elapsed)
            _build_jobs[job_id]["status"] = "done"
            _build_jobs[job_id]["result"] = {"success": True, "image_name": image_name, "logs": logs}
    except Exception as e:
        logger.error("build exception | task=%s error=%s", task_name, e)
        _build_jobs[job_id]["status"] = "error"
        _build_jobs[job_id]["result"] = {"success": False, "image_name": image_name, "logs": str(e)}


@app.post("/build")
async def build(req: BuildRequest, x_api_key: str = Header(...)):
    """Enqueue a docker image build and return a job_id for polling."""
    _check_auth(x_api_key)

    if _active_dataset is None:
        raise HTTPException(400, "No active dataset set. Call POST /setup first.")

    task_dir = DATASET_ROOT / _active_dataset / req.task_name
    if not task_dir.exists():
        raise HTTPException(400, f"Task directory not found: {task_dir}")

    image_name = f"hb__{req.task_name}"
    project    = f"build-{_sanitize(req.task_name)}"

    # Use task-specific compose file if present, else fall back to harbor template.
    task_compose = task_dir / "environment" / "docker-compose.yaml"
    build_compose = str(task_compose if task_compose.exists() else _BUILD_COMPOSE)

    env = os.environ.copy()
    env.update({
        "CONTEXT_DIR":            str(task_dir / "environment"),
        "MAIN_IMAGE_NAME":        image_name,
        "TEST_DIR":               "/tests",
        "HOST_VERIFIER_LOGS_PATH": "/tmp",
        "HOST_AGENT_LOGS_PATH":    "/tmp",
        "ENV_VERIFIER_LOGS_PATH":  "/logs/verifier",
        "ENV_AGENT_LOGS_PATH":     "/logs/agent",
        "CPUS":                   "1",
        "MEMORY":                 "2048M",
    })

    job_id = f"{_sanitize(req.task_name)}-{int(time.time())}"
    _build_jobs[job_id] = {"status": "running", "result": None, "started_at": time.time()}

    logger.info("build enqueued | task=%s job_id=%s", req.task_name, job_id)
    asyncio.create_task(_run_build(job_id, req.task_name, image_name, project, build_compose, env))

    return {"job_id": job_id, "status": "running"}


@app.get("/build/{job_id}")
async def build_status(job_id: str, x_api_key: str = Header(...)):
    """Poll the status of an enqueued build job."""
    _check_auth(x_api_key)
    job = _build_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown build job: {job_id!r}")
    return {"job_id": job_id, "status": job["status"], **(job["result"] or {})}


@app.post("/compose/up")
async def compose_up(req: ComposeUpRequest, x_api_key: str = Header(...)):
    """Start a container for one trajectory session."""
    _check_auth(x_api_key)

    if _active_dataset is None:
        raise HTTPException(400, "No active dataset set. Call POST /setup first.")

    task_dir    = DATASET_ROOT / _active_dataset / req.task_name
    trial_dir   = HARBOR_ROOT / "trials" / req.session_id
    verifier_dir = trial_dir / "verifier"
    agent_dir    = trial_dir / "agent"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    agent_dir.mkdir(parents=True, exist_ok=True)

    project = _sanitize(req.session_id)

    # Use task-specific compose file if present, else fall back to harbor prebuilt template.
    task_compose = task_dir / "environment" / "docker-compose.yaml"
    up_compose = task_compose if task_compose.exists() else _PREBUILT_COMPOSE

    env = os.environ.copy()
    env.update({
        "PREBUILT_IMAGE_NAME":    f"hb__{req.task_name}",
        "TEST_DIR":               "/tests",
        "HOST_VERIFIER_LOGS_PATH": str(verifier_dir),
        "HOST_AGENT_LOGS_PATH":    str(agent_dir),
        "ENV_VERIFIER_LOGS_PATH":  "/logs/verifier",
        "ENV_AGENT_LOGS_PATH":     "/logs/agent",
        "CPUS":                   "1",
        "MEMORY":                 "2048M",
    })
    # Client overrides (CPUS, MEMORY, etc.)
    env.update(req.env_vars)

    logger.info("compose up | session=%s task=%s", req.session_id, req.task_name)
    rc, logs = await _run(
        ["docker", "compose", "-p", project, "-f", str(up_compose), "up", "-d"],
        env=env,
        timeout=120,
    )

    if rc != 0:
        logger.error("compose up FAILED | session=%s rc=%d logs=%s",
                     req.session_id, rc, logs[-200:])
        return {"success": False, "session_id": req.session_id, "logs": logs}

    logger.info("compose up OK | session=%s project=%s", req.session_id, project)
    _sessions[req.session_id] = {
        "task_name":    req.task_name,
        "project":      project,
        "trial_dir":    trial_dir,
        "compose_file": str(up_compose),
        "env":          env,
        "started_at":   time.time(),
    }

    return {
        "success":              True,
        "session_id":           req.session_id,
        "remote_verifier_dir":  str(verifier_dir),
        "remote_agent_dir":     str(agent_dir),
    }


@app.post("/compose/down")
async def compose_down(req: ComposeDownRequest, x_api_key: str = Header(...)):
    """Stop (and optionally remove) a container."""
    _check_auth(x_api_key)

    session = _sessions.get(req.session_id)
    if session is None:
        raise HTTPException(404, f"Unknown session: {req.session_id!r}")

    project      = session["project"]
    compose_file = session["compose_file"]
    env          = session["env"]

    cmd = ["docker", "compose", "-p", project, "-f", compose_file, "down", "--timeout", "2"]
    if req.delete:
        # Remove container + volumes but keep the built image.
        # The image (e.g. hb__0) is shared across sessions — don't delete it here.
        cmd += ["--volumes", "--remove-orphans"]

    logger.info("compose down | session=%s delete=%s", req.session_id, req.delete)
    try:
        rc, logs = await _run(cmd, env=env, timeout=120)
        if rc != 0:
            logger.warning("compose down non-zero | session=%s rc=%d", req.session_id, rc)
        return {"success": rc == 0, "logs": logs}
    except RuntimeError:
        # compose down timed out — force-kill all containers in the project
        logger.warning("compose down timeout | session=%s — force killing", req.session_id)
        try:
            await _run(
                ["docker", "compose", "-p", project, "kill"],
                env=env, timeout=15,
            )
        except Exception:
            pass
        raise
    finally:
        _sessions.pop(req.session_id, None)


async def _run_exec_bg(cmd: list[str], env: dict, timeout_sec: int | None,
                       session_id: str, command: str, request_id: str) -> dict:
    """Background exec task — survives client disconnect, always returns a result dict."""
    t0 = time.time()
    try:
        rc, output = await _run(cmd, env=env, timeout=timeout_sec)
    except Exception as e:
        logger.error("exec error | session=%s cmd=%.80s error=%s", session_id, command, e)
        result = {"stdout": "", "stderr": str(e), "return_code": -1}
    else:
        elapsed = time.time() - t0
        if rc != 0:
            logger.warning("exec non-zero | session=%s rc=%d elapsed=%.1fs cmd=%.80s output=%.200s",
                           session_id, rc, elapsed, command, output)
        else:
            logger.debug("exec OK | session=%s rc=%d elapsed=%.1fs cmd=%.80s",
                         session_id, rc, elapsed, command)
        result = {"stdout": output, "stderr": None, "return_code": rc}
    # Promote to completed cache entry with expiry
    _exec_cache[request_id] = {
        "result": result,
        "expires_at": time.time() + EXEC_CACHE_TTL,
    }
    return result


@app.post("/exec")
async def exec_cmd(req: ExecRequest, x_api_key: str = Header(...)):
    """Run a command inside the running container."""
    _check_auth(x_api_key)

    if req.request_id:
        cached = _exec_cache.get(req.request_id)
        if cached is not None:
            # Completed — return cached result
            if "result" in cached:
                logger.info("exec cache hit | request_id=%s session=%s",
                            req.request_id, req.session_id)
                return cached["result"]
            # Still running — await the existing task (don't start a duplicate)
            logger.info("exec in-flight, re-attaching | request_id=%s session=%s",
                        req.request_id, req.session_id)
            return await asyncio.shield(cached["task"])

    session = _sessions.get(req.session_id)
    if session is None:
        raise HTTPException(404, f"Unknown session: {req.session_id!r}")

    project      = session["project"]
    compose_file = session["compose_file"]
    env          = session["env"]

    cmd = ["docker", "compose", "-p", project, "-f", compose_file, "exec", "-T"]
    if req.cwd:
        cmd += ["-w", req.cwd]
    if req.env:
        for k, v in req.env.items():
            cmd += ["-e", f"{k}={v}"]
    cmd += ["main", "bash", "-c", req.command]

    if req.request_id:
        # Run as background task so it survives HTTP client disconnect.
        # Register immediately so retries find the in-flight task.
        task = asyncio.create_task(_run_exec_bg(
            cmd, env, req.timeout_sec, req.session_id, req.command, req.request_id,
        ))
        _exec_cache[req.request_id] = {"task": task, "started_at": time.time()}
        # shield: if this handler is cancelled (client disconnect), the task keeps running
        return await asyncio.shield(task)
    else:
        # No request_id — run inline (backward compat for simple calls)
        t0 = time.time()
        try:
            rc, output = await _run(cmd, env=env, timeout=req.timeout_sec)
        except Exception as e:
            logger.error("exec error | session=%s cmd=%.80s error=%s",
                         req.session_id, req.command, e)
            return {"stdout": "", "stderr": str(e), "return_code": -1}
        elapsed = time.time() - t0
        if rc != 0:
            logger.warning("exec non-zero | session=%s rc=%d elapsed=%.1fs cmd=%.80s output=%.200s",
                           req.session_id, rc, elapsed, req.command, output)
        else:
            logger.debug("exec OK | session=%s rc=%d elapsed=%.1fs cmd=%.80s",
                         req.session_id, rc, elapsed, req.command)
        return {"stdout": output, "stderr": None, "return_code": rc}


@app.post("/upload")
async def upload(
    session_id:  str        = Form(...),
    target_path: str        = Form(...),
    content_tar: UploadFile = File(...),
    x_api_key:   str        = Header(...),
):
    """Upload a tarball and extract it into the container at target_path."""
    _check_auth(x_api_key)

    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"Unknown session: {session_id!r}")

    project      = session["project"]
    compose_file = session["compose_file"]
    env          = session["env"]

    tar_bytes = await content_tar.read()
    with tempfile.TemporaryDirectory() as tmpdir:
        buf = io.BytesIO(tar_bytes)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            tar.extractall(tmpdir)

        # "tmpdir/." copies contents (not the dir itself) into target_path
        rc, logs = await _run(
            ["docker", "compose", "-p", project, "-f", compose_file,
             "cp", f"{tmpdir}/.", f"main:{target_path}"],
            env=env,
            timeout=60,
        )

    return {"success": rc == 0, "logs": logs}


@app.get("/download/{session_id}")
async def download(
    session_id:  str,
    source_path: str = "/logs/verifier",
    x_api_key:   str = Header(...),
):
    """Stream a tarball of source_path from the container."""
    _check_auth(x_api_key)

    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"Unknown session: {session_id!r}")

    project      = session["project"]
    compose_file = session["compose_file"]
    env          = session["env"]

    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / "dl"
        dest.mkdir()

        rc, logs = await _run(
            ["docker", "compose", "-p", project, "-f", compose_file,
             "cp", f"main:{source_path}", str(dest)],
            env=env,
            timeout=60,
        )
        if rc != 0:
            raise HTTPException(500, f"docker compose cp failed:\n{logs}")

        buf = io.BytesIO()
        # Find the downloaded directory (docker cp creates a subdir named after source basename)
        downloaded = list(dest.iterdir())
        root = downloaded[0] if len(downloaded) == 1 and downloaded[0].is_dir() else dest

        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for f in root.rglob("*"):
                if f.is_file():
                    tar.add(f, arcname=f.relative_to(root))
        buf.seek(0)
        content = buf.read()

    return StreamingResponse(
        io.BytesIO(content),
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=download.tar.gz"},
    )


@app.get("/containers")
async def list_containers(x_api_key: str = Header(...)):
    """List tracked sessions and all running compose projects on this node."""
    _check_auth(x_api_key)

    # Tracked sessions
    now = time.time()
    sessions = [
        {
            "session_id":  sid,
            "task_name":   s["task_name"],
            "project":     s["project"],
            "age_sec":     int(now - s.get("started_at", now)),
        }
        for sid, s in _sessions.items()
    ]

    # All running compose projects
    try:
        rc, out = await _run(["docker", "compose", "ls", "--format", "json"], timeout=30)
        running = json.loads(out) if rc == 0 and out.strip() else []
    except Exception:
        running = []

    known_projects = {s["project"] for s in _sessions.values()}
    orphans = [p["Name"] for p in running if p.get("Name") not in known_projects]

    return {
        "sessions":        sessions,
        "running_projects": running,
        "orphan_projects": orphans,
    }


@app.post("/gc")
async def gc(x_api_key: str = Header(...)):
    """Manually trigger GC: kill orphaned containers and expired sessions."""
    _check_auth(x_api_key)
    return await _gc_once()


@app.post("/cleanup")
async def cleanup(x_api_key: str = Header(...)):
    """Hard cleanup: stop and remove ALL containers, prune all networks.

    Use this to fully reset the node between eval runs.
    Clears _sessions state as well.
    """
    _check_auth(x_api_key)
    global _sessions
    errors: list[str] = []

    # Stop all containers
    rc, container_ids = await _run(["docker", "ps", "-a", "-q"], timeout=30)
    container_ids = container_ids.strip()
    if container_ids:
        ids = container_ids.split()
        rc, out = await _run(["docker", "stop"] + ids, timeout=120)
        if rc != 0:
            errors.append(f"docker stop: {out[:200]}")
        rc, out = await _run(["docker", "rm"] + ids, timeout=60)
        if rc != 0:
            errors.append(f"docker rm: {out[:200]}")

    # Prune networks
    rc, out = await _run(["docker", "network", "prune", "-f"], timeout=30)
    if rc != 0:
        errors.append(f"docker network prune: {out[:200]}")

    stopped = len(container_ids.split()) if container_ids else 0
    _sessions.clear()

    return {"stopped_containers": stopped, "errors": errors}
