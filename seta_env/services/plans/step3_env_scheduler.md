# Step 3: Local Env Scheduler

**Priority**: High — routes requests to env_service nodes with affinity and load balancing.

**Status**: Not started

**Depends on**: Step 2 (env_service API contract)

## Overview

FastAPI service running on the local/GPU machine. Acts as a transparent proxy: receives StepRequests from GRPORollout, picks the best env_service node, forwards the request, and returns the response.

Key features:
- **Task affinity**: Same task_id → same node within 2-minute window (reuses built Docker images)
- **Load balancing**: When no affinity, pick node with highest free_ratio (free_slots / total_slots)
- **Local node support**: nodes.yaml can include `http://127.0.0.1:8002` for same-machine execution

## File: `seta_env/services/env_scheduler.py`

### Node State Tracking

```python
@dataclass
class NodeState:
    url: str
    total_slots: int
    active_slots: int = 0  # currently running step() calls
    # task_id → timestamp of last request sent to this node
    task_affinity: dict[str, float] = field(default_factory=dict)

    @property
    def free_slots(self) -> int:
        return max(0, self.total_slots - self.active_slots)

    @property
    def free_ratio(self) -> float:
        return self.free_slots / self.total_slots if self.total_slots > 0 else 0.0
```

### Routing Logic

```python
AFFINITY_WINDOW = 120.0  # seconds — co-locate same task_id for 2 minutes

class Scheduler:
    def __init__(self, nodes: list[NodeState]):
        self._nodes = nodes
        self._lock = asyncio.Lock()

    async def pick_node(self, task_id: str) -> NodeState:
        """Pick the best node for this task_id."""
        async with self._lock:
            now = time.monotonic()

            # 1. Check affinity: task_id recently sent to a node with capacity
            for node in self._nodes:
                ts = node.task_affinity.get(task_id)
                if ts and (now - ts) < AFFINITY_WINDOW and node.free_slots > 0:
                    node.active_slots += 1
                    node.task_affinity[task_id] = now
                    return node

            # 2. No affinity or affinity node full → best free_ratio
            candidates = [n for n in self._nodes if n.free_slots > 0]
            if not candidates:
                raise HTTPException(503, "All nodes are at capacity")

            best = max(candidates, key=lambda n: n.free_ratio)
            best.active_slots += 1
            best.task_affinity[task_id] = now
            return best

    async def release(self, node: NodeState):
        """Decrement active_slots after request completes."""
        async with self._lock:
            node.active_slots = max(0, node.active_slots - 1)
```

### Main Endpoint — Transparent Proxy

```python
@app.post("/step")
async def step(req: StepRequest, x_api_key: str = Header(...)):
    verify_api_key(x_api_key)

    # Generate task_id from task if not provided
    task_id = req.task.get("task_id") or req.task.get("task_name", req.uid)

    # Pick best node
    node = await scheduler.pick_node(task_id)

    try:
        # Forward to env_service with long timeout (step can take 10+ min)
        async with httpx.AsyncClient(timeout=httpx.Timeout(900.0)) as client:
            resp = await client.post(
                f"{node.url}/step",
                json=req.dict(),
                headers={"X-API-Key": node_api_key},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        return {"run_info": None, "reward": None, "error": str(e)}
    finally:
        await scheduler.release(node)
```

### Additional Endpoints

```python
@app.get("/health")
async def health():
    return {"status": "ok", "nodes": len(scheduler._nodes)}

@app.get("/status")
async def status():
    """Per-node breakdown: slots, active, affinity."""
    return {
        "nodes": [
            {
                "url": n.url,
                "total_slots": n.total_slots,
                "active_slots": n.active_slots,
                "free_ratio": round(n.free_ratio, 3),
                "task_affinity": {
                    k: round(time.monotonic() - v, 1)
                    for k, v in n.task_affinity.items()
                    if time.monotonic() - v < AFFINITY_WINDOW
                },
            }
            for n in scheduler._nodes
        ]
    }

@app.post("/setup_dataset")
async def setup_dataset(req: SetupRequest):
    """Fan out dataset setup to all env_service nodes in parallel."""
    async with httpx.AsyncClient(timeout=600.0) as client:
        tasks = [
            client.post(f"{n.url}/setup", json=req.dict(),
                        headers={"X-API-Key": node_api_key})
            for n in scheduler._nodes
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return {"results": [str(r) for r in results]}

@app.post("/cleanup")
async def cleanup():
    """Fan out cleanup to all nodes."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [
            client.post(f"{n.url}/cleanup", headers={"X-API-Key": node_api_key})
            for n in scheduler._nodes
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    return {"status": "ok"}
```

### Configuration

Reads from `seta_env/services/nodes.yaml`:
```yaml
nodes:
  - url: "http://65.108.32.175:8002"
    slots: 16

  - url: "http://135.181.71.8:8002"
    slots: 16

  # Optional: local node (same machine as scheduler)
  # - url: "http://127.0.0.1:8002"
  #   slots: 8
```

### Startup

```python
@app.on_event("startup")
async def startup():
    nodes_yaml = os.environ.get("NODES_YAML",
        str(Path(__file__).parent / "nodes.yaml"))
    data = yaml.safe_load(open(nodes_yaml))
    nodes = [NodeState(url=n["url"], total_slots=n["slots"])
             for n in data["nodes"]]
    global scheduler
    scheduler = Scheduler(nodes)
```

Default bind: `--host 127.0.0.1 --port 8003` (local only, not exposed)

### Affinity Cleanup

Background task runs every 60 seconds:
```python
async def _cleanup_affinity():
    while True:
        await asyncio.sleep(60)
        now = time.monotonic()
        async with scheduler._lock:
            for node in scheduler._nodes:
                expired = [k for k, v in node.task_affinity.items()
                           if now - v > AFFINITY_WINDOW * 2]
                for k in expired:
                    del node.task_affinity[k]
```

## Routing Examples

### GRPO with n_trajs=8, same task_id="task_42"

```
Request 1 (task_42): No affinity → pick node with best free_ratio → Server 1
                     Record affinity: task_42 → Server 1

Request 2-8 (task_42): Affinity hit → Server 1 (within 2 min window)
                        All 8 trajectories on same server → share built image
```

### Mixed task_ids

```
Request A1 (task_10): No affinity → Server 1 (8/16 free = 0.50 vs 16/16 = 1.0) → Server 2
Request A2 (task_10): Affinity → Server 2
Request B1 (task_20): No affinity → Server 1 (best free_ratio)
Request B2 (task_20): Affinity → Server 1
```

## Testing

### Unit test: Affinity routing
- Send 8 requests with same task_id → all routed to same node
- Wait > 2 min → next request can go to any node
- 2 different task_ids → distributed across nodes

### Unit test: Load balancing
- Node 1: 16 slots, 14 active. Node 2: 16 slots, 0 active.
- New request → goes to Node 2 (better free_ratio)

### Unit test: Capacity exhaustion
- All nodes full → HTTP 503

### Integration test:
- Start 2 env_services (localhost:8002, localhost:8012)
- Start scheduler with both in nodes.yaml
- Send 4 requests same task_id → all to one service
- Check /status shows correct active_slots and affinity
