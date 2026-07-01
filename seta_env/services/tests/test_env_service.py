"""Integration tests for env_service FastAPI app.

Tests the HTTP layer, request validation, build gating, and slot semaphore.
Uses mocked TerminalEnvironment to avoid needing Docker.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_state():
    """Reset global state between tests."""
    import seta_env.services.env_service as svc
    svc._build_gate = svc.BuildGate()
    svc._slot_semaphore = asyncio.Semaphore(svc.MAX_SLOTS)
    svc._active_count = 0
    svc.API_KEY = ""  # disable auth for tests
    yield


@pytest.fixture
def client():
    from seta_env.services.env_service import app
    # Use TestClient which handles lifespan
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["max_slots"] == 16
    assert data["active_steps"] == 0


def test_step_missing_task_dir(client, tmp_path):
    """Step with nonexistent dataset returns error, not 500."""
    resp = client.post("/step", json={
        "task": {"task_name": "nonexistent"},
        "uid": "test_1",
        "agent_config": {},
        "llm_config": {},
        "runtime_config": {},
        "env_config": {},
        "dataset_name": "fake_dataset",
        "task_name": "nonexistent",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["error"] is not None
    assert "not found" in data["error"]


def test_step_with_mock_terminal_env(client, tmp_path):
    """Step with mocked TerminalEnvironment returns run_info and reward."""
    # Create a fake task dir so path check passes
    task_dir = tmp_path / "test_dataset" / "task_0"
    task_dir.mkdir(parents=True)

    mock_run_info = {"task_name": "task_0", "reward": 0.5, "timings": {}}
    mock_reward = 0.5

    with patch("seta_env.services.env_service.DATASET_ROOT", tmp_path), \
         patch("seta_env.services.env_service.DockerHarborRuntime") as MockRT, \
         patch("seta_env.services.env_service.TerminalEnvironment") as MockTE:

        # Mock build
        mock_rt = MagicMock()
        mock_rt.build = AsyncMock()
        mock_rt.stop = AsyncMock()
        MockRT.return_value = mock_rt

        # Mock step
        mock_te = MagicMock()
        mock_te.step = AsyncMock(return_value=(mock_run_info, mock_reward))
        MockTE.return_value = mock_te

        resp = client.post("/step", json={
            "task": {"task_name": "task_0"},
            "uid": "test_session_1",
            "agent_config": {"agent": "train_agent", "prompt": "test", "max_total_tokens": 1000,
                             "max_iteration": 5, "tool_names": []},
            "llm_config": {"model_platform": "sglang", "url": "http://fake:8000/v1"},
            "runtime_config": {"trial_root": str(tmp_path / "trials")},
            "env_config": {"reward_fn": "pass_ratio"},
            "dataset_name": "test_dataset",
            "task_name": "task_0",
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["error"] is None
    assert data["run_info"]["task_name"] == "task_0"
    assert data["reward"] == 0.5


def test_build_gate_called_once_for_same_task(client, tmp_path):
    """Multiple step requests for same task_name → build_fn called once."""
    task_dir = tmp_path / "ds" / "task_X"
    task_dir.mkdir(parents=True)

    build_count = 0

    with patch("seta_env.services.env_service.DATASET_ROOT", tmp_path), \
         patch("seta_env.services.env_service.DockerHarborRuntime") as MockRT, \
         patch("seta_env.services.env_service.TerminalEnvironment") as MockTE:

        original_build = AsyncMock()

        async def counting_build():
            nonlocal build_count
            build_count += 1
            await original_build()

        mock_rt = MagicMock()
        mock_rt.build = counting_build
        mock_rt.stop = AsyncMock()
        MockRT.return_value = mock_rt

        mock_te = MagicMock()
        mock_te.step = AsyncMock(return_value=({"task_name": "task_X"}, 1.0))
        MockTE.return_value = mock_te

        # Send 3 requests for same task
        for i in range(3):
            resp = client.post("/step", json={
                "task": {"task_name": "task_X"},
                "uid": f"sess_{i}",
                "agent_config": {"agent": "train_agent", "prompt": "t",
                                 "max_total_tokens": 1000, "max_iteration": 5,
                                 "tool_names": []},
                "llm_config": {},
                "runtime_config": {"trial_root": str(tmp_path / "trials")},
                "env_config": {},
                "dataset_name": "ds",
                "task_name": "task_X",
            })
            assert resp.status_code == 200
            assert resp.json()["error"] is None

    assert build_count == 1, f"Expected 1 build, got {build_count}"


def test_auth_rejection(client):
    """Request with wrong API key is rejected."""
    import seta_env.services.env_service as svc
    svc.API_KEY = "secret"

    resp = client.post("/step", json={
        "task": {}, "uid": "x", "agent_config": {}, "llm_config": {},
    }, headers={"X-API-Key": "wrong"})
    assert resp.status_code == 403

    svc.API_KEY = ""  # reset


def test_cleanup(client):
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc

        resp = client.post("/cleanup")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
