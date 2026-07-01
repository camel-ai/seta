"""
Plan 03 — DockerHarborRuntime tests (Docker backend).
Uses: <repo_root>/dataset/terminal-bench-core_migrated/analyze-access-logs
Trial output: <repo_root>/test/output/trials
"""
import uuid
import pytest
from pathlib import Path

from seta_env.runtimes.docker_harbor_runtime import DockerHarborRuntime

_REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = str(_REPO_ROOT / "dataset/terminal-bench-core_migrated/analyze-access-logs")
TRIAL_ROOT = str(_REPO_ROOT / "test/output/trials")


def make_session_id(prefix="test"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Initialization — validation
# ---------------------------------------------------------------------------

def test_invalid_environment_type():
    with pytest.raises(AssertionError, match="Unsupported environment type"):
        DockerHarborRuntime(
            task_dir=TASK_DIR,
            trial_root=TRIAL_ROOT,
            session_id=make_session_id(),
            environment_type="k8s",
        )

def test_docker_init_ok():
    rt = DockerHarborRuntime(
        task_dir=TASK_DIR,
        trial_root=TRIAL_ROOT,
        session_id=make_session_id(),
        environment_type="docker",
    )
    assert rt.harbor_env is not None
    assert rt.session_id is not None
    assert rt._trial_paths is not None

def test_logger_initialized():
    rt = DockerHarborRuntime(
        task_dir=TASK_DIR,
        trial_root=TRIAL_ROOT,
        session_id=make_session_id(),
        environment_type="docker",
    )
    assert rt._file_handler is not None
    assert rt._trial_paths.log_path.exists()

def test_trial_dir_created():
    sid = make_session_id()
    rt = DockerHarborRuntime(
        task_dir=TASK_DIR,
        trial_root=TRIAL_ROOT,
        session_id=sid,
        environment_type="docker",
    )
    assert rt.trial_dir == Path(TRIAL_ROOT) / sid
    assert rt.trial_dir.exists()

def test_pre_built_environment():
    sid = make_session_id("prebuild")
    rt1 = DockerHarborRuntime(
        task_dir=TASK_DIR,
        trial_root=TRIAL_ROOT,
        session_id=sid,
        environment_type="docker",
    )
    harbor_env = rt1.harbor_env

    rt2 = DockerHarborRuntime(environment=harbor_env)
    assert rt2.harbor_env is harbor_env
    assert rt2.session_id == harbor_env.session_id

def test_getattr_proxy():
    rt = DockerHarborRuntime(
        task_dir=TASK_DIR,
        trial_root=TRIAL_ROOT,
        session_id=make_session_id(),
        environment_type="docker",
    )
    assert rt.session_id == rt.harbor_env.session_id


# ---------------------------------------------------------------------------
# Docker runtime lifecycle (requires Docker daemon)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_docker_reset_and_stop():
    rt = DockerHarborRuntime(
        task_dir=TASK_DIR,
        trial_root=TRIAL_ROOT,
        session_id=make_session_id("lifecycle"),
        environment_type="docker",
    )
    try:
        await rt.reset()
        result = await rt.harbor_env.exec("echo alive")
        assert result.return_code == 0
        assert "alive" in (result.stdout or "")
    finally:
        await rt.stop(delete=True)

@pytest.mark.asyncio
async def test_docker_stop_without_delete():
    rt = DockerHarborRuntime(
        task_dir=TASK_DIR,
        trial_root=TRIAL_ROOT,
        session_id=make_session_id("nodelete"),
        environment_type="docker",
    )
    await rt.reset()
    await rt.stop(delete=False)
    assert rt._file_handler is None

@pytest.mark.asyncio
async def test_docker_logger_closed_after_stop():
    rt = DockerHarborRuntime(
        task_dir=TASK_DIR,
        trial_root=TRIAL_ROOT,
        session_id=make_session_id("logclose"),
        environment_type="docker",
    )
    await rt.reset()
    assert rt._file_handler is not None
    await rt.stop(delete=True)
    assert rt._file_handler is None

@pytest.mark.asyncio
async def test_docker_context_manager():
    sid = make_session_id("ctx")
    async with DockerHarborRuntime(
        task_dir=TASK_DIR,
        trial_root=TRIAL_ROOT,
        session_id=sid,
        environment_type="docker",
    ) as rt:
        await rt.reset()
        result = await rt.harbor_env.exec("echo ctx_ok")
        assert "ctx_ok" in (result.stdout or "")
    assert rt._file_handler is None

@pytest.mark.asyncio
async def test_get_tools_returns_function_tools():
    rt = DockerHarborRuntime(
        task_dir=TASK_DIR,
        trial_root=TRIAL_ROOT,
        session_id=make_session_id("tools"),
        environment_type="docker",
    )
    try:
        await rt.reset()
        await rt.get_tools()
        assert len(rt.tools) > 0
        tool_names = [t.get_function_name() for t in rt.tools]
        assert "shell_exec" in tool_names
        assert "shell_view" in tool_names
        assert "shell_write_content_to_file" in tool_names
    finally:
        await rt.stop(delete=True)
