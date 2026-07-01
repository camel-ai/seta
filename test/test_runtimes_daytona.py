"""
Plan 03 — DockerHarborRuntime tests (Daytona backend).
Requires: DAYTONA_API_KEY and DAYTONA_API_URL set in environment.

Note on Modal: harbor's ModalEnvironment uses standard modal client auth
(MODAL_TOKEN_ID + MODAL_TOKEN_SECRET). MODAL_API_KEY is not used by modal's
client library, so Modal tests are not included here.

Tasks used:
  - nginx-request-logging (terminal-bench-2.0) — uses prebuilt docker image,
    Daytona pulls it instead of building from Dockerfile, so reset is fast.

Trial output: <repo_root>/test/output/trials

All reset() calls use reset_timeout=20s — prebuilt image should start fast.
"""
import asyncio
import os
import uuid
import pytest

from pathlib import Path
from seta_env.runtimes.docker_harbor_runtime import DockerHarborRuntime

_REPO_ROOT = Path(__file__).resolve().parents[1]
TRIAL_ROOT = str(_REPO_ROOT / "test/output/trials")
# Uses a prebuilt Docker image — Daytona pulls it instead of building, much faster
TASK_DIR = str(_REPO_ROOT / "dataset/terminal-bench-2.0/nginx-request-logging")

HAS_DAYTONA = bool(os.environ.get("DAYTONA_API_KEY"))
pytestmark = pytest.mark.skipif(not HAS_DAYTONA, reason="DAYTONA_API_KEY not set")

DAYTONA_RESET_TIMEOUT = 20  # seconds — prebuilt image, should start fast


def sid(prefix="daytona"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Initialization (no network)
# ---------------------------------------------------------------------------

def test_daytona_init_ok():
    rt = DockerHarborRuntime(
        task_dir=TASK_DIR,
        trial_root=TRIAL_ROOT,
        session_id=sid(),
        environment_type="daytona",
    )
    assert rt.harbor_env is not None
    assert rt._file_handler is not None
    assert rt._trial_paths.log_path.exists()


# ---------------------------------------------------------------------------
# Daytona lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_daytona_reset_and_exec():
    rt = DockerHarborRuntime(
        task_dir=TASK_DIR,
        trial_root=TRIAL_ROOT,
        session_id=sid("exec"),
        environment_type="daytona",
    )
    try:
        await rt.reset(reset_timeout=DAYTONA_RESET_TIMEOUT)
        result = await rt.harbor_env.exec("echo daytona_alive")
        assert result.return_code == 0
        assert "daytona_alive" in (result.stdout or "")
    finally:
        await rt.stop(delete=True)

@pytest.mark.asyncio
async def test_daytona_stop_delete():
    rt = DockerHarborRuntime(
        task_dir=TASK_DIR,
        trial_root=TRIAL_ROOT,
        session_id=sid("del"),
        environment_type="daytona",
    )
    await rt.reset()
    await rt.stop(delete=True)
    assert rt._file_handler is None

@pytest.mark.asyncio
async def test_daytona_context_manager():
    async with DockerHarborRuntime(
        task_dir=TASK_DIR,
        trial_root=TRIAL_ROOT,
        session_id=sid("ctx"),
        environment_type="daytona",
    ) as rt:
        await rt.reset(reset_timeout=DAYTONA_RESET_TIMEOUT)
        result = await rt.harbor_env.exec("echo daytona_ctx_ok")
        assert "daytona_ctx_ok" in (result.stdout or "")
    assert rt._file_handler is None

@pytest.mark.asyncio
async def test_daytona_get_tools():
    rt = DockerHarborRuntime(
        task_dir=TASK_DIR,
        trial_root=TRIAL_ROOT,
        session_id=sid("tools"),
        environment_type="daytona",
    )
    try:
        await rt.reset(reset_timeout=DAYTONA_RESET_TIMEOUT)
        await rt.get_tools()
        names = [t.get_function_name() for t in rt.tools]
        assert "shell_exec" in names
        assert "shell_view" in names
        assert "shell_write_content_to_file" in names
    finally:
        await rt.stop(delete=True)


@pytest.mark.asyncio
async def test_reset_timeout_raises():
    """reset_timeout=0.001 must raise asyncio.TimeoutError, never hang."""
    rt = DockerHarborRuntime(
        task_dir=TASK_DIR,
        trial_root=TRIAL_ROOT,
        session_id=sid("timeout"),
        environment_type="daytona",
    )
    with pytest.raises(asyncio.TimeoutError):
        await rt.reset(reset_timeout=0.001)
    # cleanup: sandbox may not have started, stop() should be safe regardless
    await rt.stop(delete=True)
