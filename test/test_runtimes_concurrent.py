"""
Plan 03 — DockerHarborRuntime concurrency tests.

Tests:
  - Multiple sessions of the same task in parallel (Docker)
  - Multiple different tasks, multiple sessions each (Docker)
  - Multiple sessions of the same task in parallel (Daytona)
  - Multiple different tasks, multiple sessions each (Daytona)

Tasks used:
  - analyze-access-logs  (easy, data-science)
  - ancient-puzzle       (easy, file-operations)
  - assign-seats         (easy, algorithms)

Trial output: <repo_root>/test/output/trials
"""
import asyncio
import os
import uuid
import pytest

DAYTONA_RESET_TIMEOUT = 600  # seconds per sandbox start — matches task build_timeout_sec

from seta_env.runtimes.docker_harbor_runtime import DockerHarborRuntime

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = str(_REPO_ROOT / "dataset/terminal-bench-2.0")
TRIAL_ROOT = str(_REPO_ROOT / "test/output/trials")

# All three use prebuilt Docker images — no build step, no image-name race condition
TASKS = {
    "nginx":    f"{DATASET}/nginx-request-logging",
    "logs":     f"{DATASET}/log-summary-date-ranges",
    "sanitize": f"{DATASET}/sanitize-git-repo",
}

HAS_DAYTONA = bool(os.environ.get("DAYTONA_API_KEY"))


def sid(prefix="c"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ===========================================================================
# Docker — multiple sessions, same task
# ===========================================================================

@pytest.mark.asyncio
async def test_docker_3_sessions_same_task():
    """3 parallel sessions of nginx-request-logging (prebuilt image) — no interference."""
    runtimes = [
        DockerHarborRuntime(
            task_dir=TASKS["nginx"],
            trial_root=TRIAL_ROOT,
            session_id=sid(f"same_{i}"),
            environment_type="docker",
        )
        for i in range(3)
    ]
    try:
        await asyncio.gather(*[rt.reset(reset_timeout=DAYTONA_RESET_TIMEOUT) for rt in runtimes])

        results = await asyncio.gather(*[
            rt.harbor_env.exec(f"echo session_{i}")
            for i, rt in enumerate(runtimes)
        ])
        for i, result in enumerate(results):
            assert result.return_code == 0
            assert f"session_{i}" in (result.stdout or "")
    finally:
        await asyncio.gather(*[rt.stop(delete=True) for rt in runtimes])


# ===========================================================================
# Docker — multiple tasks, multiple sessions each
# ===========================================================================

@pytest.mark.asyncio
async def test_docker_3_tasks_2_sessions_each():
    """
    3 tasks × 2 sessions each = 6 parallel Docker containers.

    All tasks use prebuilt Docker images — no build step, no image-name export race.
    Sessions start concurrently without any warm-up needed.
    """
    task_keys = ["nginx", "logs", "sanitize"]

    combos = [
        (task_key, session_i)
        for task_key in task_keys
        for session_i in range(2)
    ]
    runtimes = [
        DockerHarborRuntime(
            task_dir=TASKS[task_key],
            trial_root=TRIAL_ROOT,
            session_id=sid(f"{task_key}_{session_i}"),
            environment_type="docker",
        )
        for task_key, session_i in combos
    ]

    try:
        await asyncio.gather(*[rt.reset() for rt in runtimes])

        markers = [f"mark_{task_key}_{session_i}" for task_key, session_i in combos]
        results = await asyncio.gather(*[
            rt.harbor_env.exec(f"echo {marker}")
            for rt, marker in zip(runtimes, markers)
        ])

        for result, marker in zip(results, markers):
            assert result.return_code == 0, f"Non-zero exit for {marker}"
            assert marker in (result.stdout or ""), (
                f"Expected '{marker}' in stdout, got: {result.stdout!r}"
            )
    finally:
        await asyncio.gather(*[rt.stop(delete=True) for rt in runtimes])


# ===========================================================================
# Daytona — multiple sessions, same task
# ===========================================================================

@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_DAYTONA, reason="DAYTONA_API_KEY not set")
async def test_daytona_2_sessions_same_task():
    """2 parallel Daytona sessions of the same task."""
    runtimes = [
        DockerHarborRuntime(
            task_dir=TASKS["nginx"],
            trial_root=TRIAL_ROOT,
            session_id=sid(f"dt_same_{i}"),
            environment_type="daytona",
        )
        for i in range(2)
    ]
    try:
        await asyncio.gather(*[rt.reset(reset_timeout=DAYTONA_RESET_TIMEOUT) for rt in runtimes])

        results = await asyncio.gather(*[
            rt.harbor_env.exec(f"echo dt_session_{i}")
            for i, rt in enumerate(runtimes)
        ])
        for i, result in enumerate(results):
            assert result.return_code == 0
            assert f"dt_session_{i}" in (result.stdout or "")
    finally:
        await asyncio.gather(*[rt.stop(delete=True) for rt in runtimes])


# ===========================================================================
# Daytona — multiple tasks, multiple sessions each
# ===========================================================================

@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_DAYTONA, reason="DAYTONA_API_KEY not set")
async def test_daytona_2_tasks_2_sessions_each():
    """2 Daytona tasks × 2 sessions each = 4 parallel workspaces."""
    combos = [
        (task_key, session_i)
        for task_key in ["nginx", "logs"]
        for session_i in range(2)
    ]

    runtimes = [
        DockerHarborRuntime(
            task_dir=TASKS[task_key],
            trial_root=TRIAL_ROOT,
            session_id=sid(f"dt_{task_key}_{session_i}"),
            environment_type="daytona",
        )
        for task_key, session_i in combos
    ]

    try:
        await asyncio.gather(*[rt.reset(reset_timeout=DAYTONA_RESET_TIMEOUT) for rt in runtimes])

        markers = [f"dt_{task_key}_{session_i}" for task_key, session_i in combos]
        results = await asyncio.gather(*[
            rt.harbor_env.exec(f"echo {marker}")
            for rt, marker in zip(runtimes, markers)
        ])

        for result, marker in zip(results, markers):
            assert result.return_code == 0
            assert marker in (result.stdout or "")
    finally:
        await asyncio.gather(*[rt.stop(delete=True) for rt in runtimes])
