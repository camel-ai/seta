"""Unit tests for BuildGate — the per-task_name single-flight build pattern."""

import asyncio
import sys
from pathlib import Path

import pytest

# Add project root to path so we can import env_service
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from seta_env.services.env_service import BuildGate


@pytest.mark.asyncio
async def test_single_flight_same_task():
    """10 concurrent requests for same task → only 1 build_fn call."""
    gate = BuildGate()
    build_count = 0

    async def build_fn():
        nonlocal build_count
        build_count += 1
        await asyncio.sleep(0.1)  # simulate build time

    await asyncio.gather(*[
        gate.ensure_built("task_a", build_fn) for _ in range(10)
    ])
    assert build_count == 1, f"Expected 1 build, got {build_count}"


@pytest.mark.asyncio
async def test_parallel_different_tasks():
    """2 different task_names → 2 parallel builds."""
    gate = BuildGate()
    build_count = 0
    concurrent = 0
    max_concurrent = 0

    async def build_fn():
        nonlocal build_count, concurrent, max_concurrent
        build_count += 1
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0.1)
        concurrent -= 1

    await asyncio.gather(
        gate.ensure_built("task_a", build_fn),
        gate.ensure_built("task_b", build_fn),
    )
    assert build_count == 2, f"Expected 2 builds, got {build_count}"
    assert max_concurrent == 2, f"Expected 2 concurrent builds, got {max_concurrent}"


@pytest.mark.asyncio
async def test_build_failure_propagates():
    """Build fails → all waiters get RuntimeError."""
    gate = BuildGate()

    async def failing_build():
        await asyncio.sleep(0.05)
        raise RuntimeError("docker build failed")

    results = await asyncio.gather(*[
        gate.ensure_built("task_fail", failing_build) for _ in range(5)
    ], return_exceptions=True)

    assert all(isinstance(r, RuntimeError) for r in results), \
        f"Expected all RuntimeError, got: {[type(r).__name__ for r in results]}"
    assert all("docker build failed" in str(r) for r in results)


@pytest.mark.asyncio
async def test_cached_after_build():
    """After a successful build, subsequent calls return immediately (no rebuild)."""
    gate = BuildGate()
    build_count = 0

    async def build_fn():
        nonlocal build_count
        build_count += 1
        await asyncio.sleep(0.05)

    # First batch
    await asyncio.gather(*[
        gate.ensure_built("task_cached", build_fn) for _ in range(3)
    ])
    assert build_count == 1

    # Second batch — should NOT rebuild
    await asyncio.gather(*[
        gate.ensure_built("task_cached", build_fn) for _ in range(3)
    ])
    assert build_count == 1, f"Expected no rebuild, but build_count={build_count}"


@pytest.mark.asyncio
async def test_clear_ttl():
    """clear() removes entries older than TTL."""
    gate = BuildGate()

    async def build_fn():
        pass

    await gate.ensure_built("old_task", build_fn)
    assert "old_task" in gate._registry

    # Clear with 0 TTL → removes everything
    cleared = gate.clear(older_than=0)
    assert cleared == 1
    assert "old_task" not in gate._registry


@pytest.mark.asyncio
async def test_stats():
    """stats property reports correct counts."""
    gate = BuildGate()

    async def build_fn():
        pass

    async def fail_fn():
        raise RuntimeError("fail")

    await gate.ensure_built("ok_task", build_fn)
    try:
        await gate.ensure_built("bad_task", fail_fn)
    except RuntimeError:
        pass

    stats = gate.stats
    assert stats["built"] == 1
    assert stats["failed"] == 1
    assert stats["building"] == 0


@pytest.mark.asyncio
async def test_mixed_tasks_isolation():
    """Failed build for task_a does NOT affect task_b."""
    gate = BuildGate()

    async def fail_fn():
        raise RuntimeError("fail")

    async def ok_fn():
        pass

    results = await asyncio.gather(
        gate.ensure_built("fail_task", fail_fn),
        gate.ensure_built("ok_task", ok_fn),
        return_exceptions=True,
    )

    assert isinstance(results[0], RuntimeError)
    assert results[1] is None  # success returns None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
