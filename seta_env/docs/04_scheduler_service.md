# Plan 04 — Scheduler Service

## Source
`seta_env/environments/scheduler_service.py`  (new file)

## Depends On
- Stage 01 (Node Manager) passing

## What It Does
FastAPI service running **locally** on the training machine. Tracks slot availability
across all remote node managers. Allocates groups of N slots atomically for GRPO
rollouts — all N trajectories of one task are allocated together, ensuring they can
all start before the gradient update waits for them.

## Concurrency Constraints
- Max **group size 16** — enforced as a hard limit in `allocate_group`.
- `asyncio.Lock` prevents concurrent allocation races.
- Balanced allocation: greedy fill from the node with the most free slots first.

## Config File (`nodes.yaml`)

```yaml
nodes:
  - url: "http://95.133.253.67:8001"
    slots: 256
```

`dataset_root` and active dataset are managed by the node manager (`POST /setup`),
not the scheduler. The scheduler only tracks slot availability.

## Class Signature

```python
# seta_env/environments/scheduler_service.py

class NodeState:
    url: str
    total_slots: int
    slots: dict[int, str | None]   # slot_id -> task_id (None = free)

    @property
    def free_count(self) -> int: ...
    def allocate(self, n: int, task_id: str) -> list[int]: ...
    def release(self, task_id: str) -> int: ...

class Scheduler:
    def __init__(self, nodes: list[NodeConfig]): ...

    async def allocate_group(self, task_id: str, n_slots: int) -> list[SlotAssignment]:
        """
        Atomically allocate n_slots for task_id.
        Raises HTTP 400 if task_id already allocated.
        Raises HTTP 503 if not enough free slots.
        Raises HTTP 400 if n_slots > 16.

        Strategy: sort nodes by free_count descending, fill greedily.
        All slots for one group prefer to land on the SAME node when possible
        (minimizes cross-node overhead), overflow to next node only if needed.
        """

    async def release_group(self, task_id: str) -> int:
        """Release all slots for task_id. Returns count released."""

    def status(self) -> dict: ...
```

## Allocation Strategy

Given one node with 256 slots, `n_slots=4`:

```
Node A: free=256
→ Allocate 4 slots from Node A

Node A: free=4  (after many allocations), need n=4
→ Allocate remaining 4 from Node A

Node A: free=3, need n=4
→ 503 — not enough free slots
```

Multi-node overflow still works if future `nodes.yaml` entries are added:
```
Node A: free=2  Node B: free=4, need n=4
→ Take 2 from A + 2 from B (overflow to next node)
```

## API Endpoints

### `GET /health`
```json
{"status": "ok"}
```

### `POST /allocate_group`
```json
// Request
{"task_id": "task0_step42", "n_slots": 4}

// Response
{
  "task_id": "task0_step42",
  "assignments": [
    {"node_url": "http://95.133.253.67:8001", "slot_id": 0},
    {"node_url": "http://95.133.253.67:8001", "slot_id": 1},
    {"node_url": "http://95.133.253.67:8001", "slot_id": 2},
    {"node_url": "http://95.133.253.67:8001", "slot_id": 3}
  ]
}
```

### `POST /release_group`
```json
// Request
{"task_id": "task0_step42"}

// Response
{"task_id": "task0_step42", "released_slots": 4}
```

### `GET /status`
```json
{
  "nodes": [
    {"url": "http://95.133.253.67:8001", "total_slots": 256, "free_slots": 252,
     "slots": {"0": "task0_step42", "1": null, ...}}
  ],
  "active_groups": {
    "task0_step42": [{"node_url": "...", "slot_id": 0}, ...]
  }
}
```

## Starting the Service

```bash
# In a separate terminal on the training machine:
cd <REPO_ROOT>/seta_env/environments
uvicorn scheduler_service:app --host 127.0.0.1 --port 8000
```

Or programmatically via `subprocess` at training start:
```python
import subprocess
proc = subprocess.Popen(["uvicorn", "scheduler_service:app",
                         "--host", "127.0.0.1", "--port", "8000"])
```

---

## Test Script
`seta_env/test/test_scheduler_service.py`

Run: `python seta_env/test/test_scheduler_service.py`

Two parts:
1. **Unit tests** — call `Scheduler` class directly, no HTTP, no service running
2. **HTTP tests** — script starts the scheduler via `subprocess`, tests via httpx

## Dependencies
- No remote node needed
- Script auto-starts `uvicorn scheduler_service:app` on port 8000 for HTTP tests

## Script Structure

```python
"""Test: Scheduler Service (unit + HTTP)
Run: python test_scheduler_service.py
"""
import asyncio, subprocess, sys, time, uuid
import httpx
from fastapi import HTTPException
from seta_env.environments.scheduler_service import Scheduler, NodeConfig


# ── Unit tests (no HTTP) ─────────────────────────────────────────────────────

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
        return_exceptions=True
    )
    successes = [r for r in results if isinstance(r, list)]
    failures  = [r for r in results if isinstance(r, Exception)]
    assert len(successes) == 16, f"Expected 16 successes, got {len(successes)}"
    assert len(failures)  == 0,  f"Expected 0 failures, got {len(failures)}"
    for assignments in successes:
        assert len(assignments) == 16   # no partial allocation
    # now exhausted — one more should fail
    extra = await asyncio.gather(s.allocate_group("overflow", 1), return_exceptions=True)
    assert isinstance(extra[0], Exception)
    print("PASS test_concurrent_allocation_lock")


# ── HTTP tests (starts uvicorn) ───────────────────────────────────────────────

def start_scheduler():
    proc = subprocess.Popen(
        ["uvicorn", "seta_env.environments.scheduler_service:app",
         "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(1.5)   # wait for startup
    return proc


async def test_http_lifecycle():
    async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=10) as c:
        r = await c.get("/health")
        assert r.status_code == 200

        r = await c.post("/allocate_group", json={"task_id": "x", "n_slots": 2})
        assert r.status_code == 200
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
        assert total_used == 0
    print("PASS test_http_lifecycle")


async def main():
    # Unit tests
    await test_allocate_same_node()
    await test_max_group_size()
    await test_no_free_slots()
    await test_duplicate_task_id()
    await test_release_and_reallocate()
    await test_concurrent_allocation_lock()

    # HTTP tests — start service, run, then kill
    proc = start_scheduler()
    try:
        await test_http_lifecycle()
    finally:
        proc.terminate()
        proc.wait()

    print("\nAll scheduler tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
```

## Test Cases

| Test | Checks |
|---|---|
| `test_allocate_same_node` | 4 slots allocated from the single 256-slot node |
| `test_max_group_size` | `n_slots=17` raises an error mentioning "16" |
| `test_no_free_slots` | second allocation raises when node is full |
| `test_duplicate_task_id` | second allocation with same `task_id` raises |
| `test_release_and_reallocate` | release frees slots; same `task_id` can be reused |
| `test_concurrent_allocation_lock` | 16 concurrent groups of 16 all succeed on 256-slot node; 257th slot request fails |
| `test_http_lifecycle` | health → allocate → status → release → status via HTTP |

## Setup Notes

- Unit tests need no service running — all in-process, fast.
- HTTP test spawns `uvicorn` in a subprocess and kills it after.
- The service uses a `nodes.yaml` file in `seta_env/environments/` — unit tests bypass this entirely.
