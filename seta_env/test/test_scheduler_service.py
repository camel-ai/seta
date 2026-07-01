"""Test: Scheduler Service (unit + HTTP)
Run: python seta_env/test/test_scheduler_service.py
"""
import asyncio
import subprocess
import sys
import time

import httpx
from fastapi import HTTPException

from seta_env.runtimes.slot_pool_service.scheduler_service import (
    Scheduler,
    NodeConfig,
)


# ── Unit tests (no HTTP) ──────────────────────────────────────────────────────

async def test_allocate_same_node():
    s = Scheduler([NodeConfig(url="http://node-a:8001", slots=256)])
    assignments = await s.allocate_group("t1", 4)
    assert len(assignments) == 4
    assert all(a.node_url == "http://node-a:8001" for a in assignments)
    await s.release_group("t1")
    print("PASS test_allocate_same_node")


async def test_max_group_size():
    s = Scheduler([NodeConfig(url="http://node-a:8001", slots=256)])
    try:
        await s.allocate_group("t1", 17)
        assert False, "Should have raised"
    except (HTTPException, ValueError) as e:
        assert "16" in str(e)
    print("PASS test_max_group_size")


async def test_no_free_slots():
    s = Scheduler([NodeConfig(url="http://node-a:8001", slots=4)])
    await s.allocate_group("t1", 4)
    try:
        await s.allocate_group("t2", 1)
        assert False, "Should have raised"
    except (HTTPException, Exception):
        pass
    await s.release_group("t1")
    print("PASS test_no_free_slots")


async def test_duplicate_task_id():
    s = Scheduler([NodeConfig(url="http://node-a:8001", slots=256)])
    await s.allocate_group("t1", 2)
    try:
        await s.allocate_group("t1", 2)
        assert False, "Should have raised"
    except (HTTPException, ValueError):
        pass
    await s.release_group("t1")
    print("PASS test_duplicate_task_id")


async def test_release_and_reallocate():
    s = Scheduler([NodeConfig(url="http://node-a:8001", slots=256)])
    await s.allocate_group("t1", 16)
    released = await s.release_group("t1")
    assert released == 16
    assignments = await s.allocate_group("t1", 16)
    assert len(assignments) == 16
    await s.release_group("t1")
    print("PASS test_release_and_reallocate")


async def test_concurrent_allocation_lock():
    # 256 slots, 16 groups of 16 = 256 total — all succeed
    s = Scheduler([NodeConfig(url="http://node-a:8001", slots=256)])
    results = await asyncio.gather(
        *[s.allocate_group(f"t{i}", 16) for i in range(16)],
        return_exceptions=True,
    )
    successes = [r for r in results if isinstance(r, list)]
    failures  = [r for r in results if isinstance(r, Exception)]
    assert len(successes) == 16, f"Expected 16 successes, got {len(successes)}: {failures}"
    assert len(failures)  == 0,  f"Expected 0 failures, got {len(failures)}: {failures}"
    for assignments in successes:
        assert len(assignments) == 16   # no partial allocation
    # now exhausted — one more should fail
    extra = await asyncio.gather(s.allocate_group("overflow", 1), return_exceptions=True)
    assert isinstance(extra[0], Exception), f"Expected exception, got {extra[0]}"
    print("PASS test_concurrent_allocation_lock")


async def test_multi_node_overflow():
    # Node A has 2 slots, Node B has 2 slots; need n=4 → must take 2 from each
    s = Scheduler([
        NodeConfig(url="http://node-a:8001", slots=2),
        NodeConfig(url="http://node-b:8001", slots=2),
    ])
    assignments = await s.allocate_group("cross", 4)
    assert len(assignments) == 4
    node_a_slots = [a for a in assignments if a.node_url == "http://node-a:8001"]
    node_b_slots = [a for a in assignments if a.node_url == "http://node-b:8001"]
    assert len(node_a_slots) == 2, f"Expected 2 on node-a, got {len(node_a_slots)}"
    assert len(node_b_slots) == 2, f"Expected 2 on node-b, got {len(node_b_slots)}"
    await s.release_group("cross")
    print("PASS test_multi_node_overflow")


async def test_proportional_balance():
    """Greedy fill by free_ratio keeps utilization proportional across nodes.

    Node A: 256 slots, Node B: 64 slots (ratio 4:1).
    After N groups allocated, each node's utilization rate should be ~equal
    (within one group-size tolerance).
    """
    s = Scheduler([
        NodeConfig(url="http://node-a:8001", slots=256),
        NodeConfig(url="http://node-b:8001", slots=64),
    ])
    # Allocate 10 groups of 16 = 160 slots (50% of total 320)
    n_groups = 10
    for i in range(n_groups):
        await s.allocate_group(f"g{i}", 16)

    status = s.status()
    nodes = status["nodes"]
    ratios = {
        n["url"]: (n["total_slots"] - n["free_slots"]) / n["total_slots"]
        for n in nodes
    }
    r_a = ratios["http://node-a:8001"]
    r_b = ratios["http://node-b:8001"]

    # Utilisation rates should be equal within one group-worth of tolerance
    tol = 16 / 64   # one group on the smaller node = max step size
    assert abs(r_a - r_b) <= tol, \
        f"Utilisation imbalance: node-a={r_a:.2%}, node-b={r_b:.2%}, tol={tol:.2%}"

    for i in range(n_groups):
        await s.release_group(f"g{i}")
    print(f"PASS test_proportional_balance (node-a={r_a:.1%}, node-b={r_b:.1%})")


# ── HTTP tests (starts uvicorn) ───────────────────────────────────────────────

def start_scheduler() -> subprocess.Popen:
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "seta_env.runtimes.slot_pool_service.scheduler_service:app",
            "--host", "127.0.0.1", "--port", "8000",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2.0)   # wait for startup
    return proc


async def test_http_lifecycle():
    async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=10) as c:
        r = await c.get("/health")
        assert r.status_code == 200, r.text

        r = await c.post("/allocate_group", json={"task_id": "x", "n_slots": 2})
        assert r.status_code == 200, r.text
        data = r.json()
        assert len(data["assignments"]) == 2

        r = await c.get("/status")
        assert r.status_code == 200

        r = await c.post("/release_group", json={"task_id": "x"})
        assert r.status_code == 200
        assert r.json()["released_slots"] == 2

        r = await c.get("/status")
        status = r.json()
        total_used = sum(n["total_slots"] - n["free_slots"] for n in status["nodes"])
        assert total_used == 0, f"Expected 0 used slots, got {total_used}"
    print("PASS test_http_lifecycle")


async def test_http_balance():
    """Verify proportional load balance across two differently-sized nodes.

    nodes.yaml: node-1 (256 slots) + node-2 (64 slots) = 320 total.
    Allocates half capacity (10 groups × 16 = 160 slots) and checks that
    each node's utilisation rate is roughly equal (within one group tolerance).
    Then fills completely, verifies 503, and checks full recovery.
    """
    async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=30) as c:
        status = (await c.get("/status")).json()
        nodes = status["nodes"]
        assert len(nodes) == 2, f"Expected 2 nodes, got {len(nodes)}"

        total_slots = sum(n["total_slots"] for n in nodes)   # 320

        # ── Phase 1: half-fill and check proportional balance ──────────────
        half_groups = total_slots // 32   # 10 groups × 16 = 160 slots (50%)
        half_ids = [f"half_{i}" for i in range(half_groups)]
        for tid in half_ids:
            r = await c.post("/allocate_group", json={"task_id": tid, "n_slots": 16})
            assert r.status_code == 200, f"half-fill failed for {tid}: {r.text}"

        status = (await c.get("/status")).json()
        ratios = {
            n["url"]: (n["total_slots"] - n["free_slots"]) / n["total_slots"]
            for n in status["nodes"]
        }
        min_node = min(status["nodes"], key=lambda n: n["total_slots"])
        tol = 16 / min_node["total_slots"]   # one group on smallest node
        r_vals = list(ratios.values())
        assert abs(r_vals[0] - r_vals[1]) <= tol, (
            f"Utilisation imbalance after half-fill: "
            + ", ".join(f"{u}={r:.1%}" for u, r in ratios.items())
            + f" (tol={tol:.1%})"
        )

        # ── Phase 2: fill remaining capacity ──────────────────────────────
        remaining_slots = sum(n["free_slots"] for n in status["nodes"])
        fill_ids = []
        i = 0
        while remaining_slots >= 16:
            tid = f"fill_{i}"
            r = await c.post("/allocate_group", json={"task_id": tid, "n_slots": 16})
            if r.status_code != 200:
                break
            fill_ids.append(tid)
            remaining_slots -= 16
            i += 1

        # 503 when totally exhausted
        r = await c.post("/allocate_group", json={"task_id": "over", "n_slots": 1})
        assert r.status_code == 503, f"Expected 503 when full, got {r.status_code}"

        # ── Phase 3: release all and verify full recovery ──────────────────
        for tid in half_ids + fill_ids:
            await c.post("/release_group", json={"task_id": tid})

        status = (await c.get("/status")).json()
        for n in status["nodes"]:
            assert n["free_slots"] == n["total_slots"], (
                f"Node {n['url']} not fully recovered: "
                f"free={n['free_slots']} total={n['total_slots']}"
            )

    node_summary = ", ".join(
        f"{n['url']}({n['total_slots']}slots, util={ratios[n['url']]:.1%})"
        for n in nodes
    )
    print(f"PASS test_http_balance [{node_summary}]")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    # Unit tests
    await test_allocate_same_node()
    await test_max_group_size()
    await test_no_free_slots()
    await test_duplicate_task_id()
    await test_release_and_reallocate()
    await test_concurrent_allocation_lock()
    await test_multi_node_overflow()
    await test_proportional_balance()

    # HTTP tests — start service, run, then kill
    print("Starting scheduler service...")
    proc = start_scheduler()
    try:
        await test_http_lifecycle()
        await test_http_balance()
    finally:
        proc.terminate()
        proc.wait()

    print("\nAll scheduler tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
