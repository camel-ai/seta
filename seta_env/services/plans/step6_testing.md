# Step 6: End-to-End Testing Plan

**Priority**: Runs alongside each implementation step.

**Status**: Not started

## Overview

Testing is layered: unit tests for each component, integration tests for connected components, and end-to-end tests for the full pipeline. Each step has its own test section; this document describes the full validation sequence.

## Test Sequence

### Phase A: FRP Tunnel (Step 1)

#### A1. Script Smoke Test (no sglang needed)
```bash
# 1. Deploy frps to CPU Server 1
scp frps_start.sh root@65.108.32.175:/opt/frp/
ssh root@65.108.32.175 "cd /opt/frp && bash frps_start.sh"
# Expected: "frps started (PID xxx) on port 7000"

# 2. Start mock HTTP server locally
python3 -m http.server 8080 &

# 3. Start frpc locally
./frpc_start.sh --server 65.108.32.175 --ranks "127.0.0.1:8080"
# Expected: "Rank 0: 127.0.0.1:8080 → http://65.108.32.175:39001"

# 4. Test from CPU Server 1
ssh root@65.108.32.175 "curl -s http://127.0.0.1:39001/"
# Expected: directory listing from mock server

# 5. Test from CPU Server 2
ssh root@135.181.71.8 "curl -s http://65.108.32.175:39001/"
# Expected: same directory listing

# 6. Run tunnel_status.sh
./tunnel_status.sh --relay 65.108.32.175 --num-ranks 1
# Expected: "Rank 0 :39001  [OK]"

# 7. Cleanup
kill %1  # mock server
pkill -f "frpc.*-c"
```

#### A2. Multi-Rank Test
```bash
# Start 4 mock servers on different ports
for p in 8081 8082 8083 8084; do python3 -m http.server $p & done

./frpc_start.sh --server 65.108.32.175 \
  --ranks "127.0.0.1:8081,127.0.0.1:8082,127.0.0.1:8083,127.0.0.1:8084"

# Verify each rank from CPU Server 2
for p in 39001 39002 39003 39004; do
  ssh root@135.181.71.8 "curl -sf http://65.108.32.175:$p/ > /dev/null && echo 'Port $p OK' || echo 'Port $p FAIL'"
done
```

#### A3. Sglang Test (when available)
```bash
# After sglang starts, get the IP:PORT per rank, then:
./frpc_start.sh --server 65.108.32.175 --ranks "<sglang_ip>:<sglang_port>"

# From CPU Server 1:
ssh root@65.108.32.175 "python3 test_tunnel.py \
  --base-url http://127.0.0.1:39001/v1 \
  --concurrency 16"
```

---

### Phase B: Env Service (Step 2)

#### B1. Unit Test — BuildGate
```python
# test_build_gate.py
import asyncio, pytest

async def test_single_flight():
    """10 concurrent requests, same task → only 1 build call."""
    gate = BuildGate()
    build_count = 0

    async def build_fn():
        nonlocal build_count
        build_count += 1
        await asyncio.sleep(0.1)  # simulate build time

    await asyncio.gather(*[
        gate.ensure_built("task_a", build_fn) for _ in range(10)
    ])
    assert build_count == 1

async def test_parallel_tasks():
    """2 different tasks → 2 parallel builds."""
    gate = BuildGate()
    build_count = 0

    async def build_fn():
        nonlocal build_count
        build_count += 1
        await asyncio.sleep(0.1)

    await asyncio.gather(
        gate.ensure_built("task_a", build_fn),
        gate.ensure_built("task_b", build_fn),
    )
    assert build_count == 2

async def test_build_failure():
    """Build fails → all waiters get error."""
    gate = BuildGate()

    async def failing_build():
        raise RuntimeError("build failed")

    results = await asyncio.gather(*[
        gate.ensure_built("task_c", failing_build) for _ in range(5)
    ], return_exceptions=True)
    assert all(isinstance(r, RuntimeError) for r in results)
```

#### B2. Unit Test — Semaphore
```python
async def test_slot_limit():
    """MAX_SLOTS=2, send 4 requests → max 2 concurrent."""
    sem = asyncio.Semaphore(2)
    concurrent = 0
    max_concurrent = 0

    async def task():
        nonlocal concurrent, max_concurrent
        async with sem:
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
            await asyncio.sleep(0.1)
            concurrent -= 1

    await asyncio.gather(*[task() for _ in range(4)])
    assert max_concurrent == 2
```

#### B3. Integration Test — Local Env Service
```bash
# 1. Start env_service locally
MAX_SLOTS=2 ENV_SERVICE_API_KEY=test \
  uvicorn seta_env.services.env_service:app --port 8002 &

# 2. Health check
curl http://127.0.0.1:8002/health
# Expected: {"status":"ok", "max_slots":2, "available_slots":2, ...}

# 3. Setup dataset
curl -X POST http://127.0.0.1:8002/setup \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test" \
  -d '{"dataset_name": "seta-env-harbor"}'

# 4. Send step request (requires sglang or mock model)
curl -X POST http://127.0.0.1:8002/step \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test" \
  -d '{
    "task": {"task_name": "0", "instruction": "..."},
    "uid": "test_session_1",
    "traj_i": 0,
    "agent_config": {...},
    "model_config": {"model_platform": "sglang", "url": "http://...", ...},
    "runtime_config": {"trial_root": "/tmp/trials", "environment_type": "docker"},
    "env_config": {"reward_fn": "pass_ratio"},
    "dataset_name": "seta-env-harbor",
    "task_name": "0"
  }'
# Expected: {"run_info": {...}, "reward": 0.5}
```

---

### Phase C: Env Scheduler (Step 3)

#### C1. Unit Test — Routing
```python
async def test_affinity():
    """Same task_id → same node within window."""
    scheduler = Scheduler([
        NodeState(url="http://node1:8002", total_slots=16),
        NodeState(url="http://node2:8002", total_slots=16),
    ])
    node1 = await scheduler.pick_node("task_42")
    for _ in range(7):
        node = await scheduler.pick_node("task_42")
        assert node.url == node1.url  # affinity holds
        await scheduler.release(node)
    await scheduler.release(node1)

async def test_load_balance():
    """No affinity → pick node with best free_ratio."""
    nodes = [
        NodeState(url="http://node1:8002", total_slots=16, active_slots=14),
        NodeState(url="http://node2:8002", total_slots=16, active_slots=0),
    ]
    scheduler = Scheduler(nodes)
    node = await scheduler.pick_node("new_task")
    assert node.url == "http://node2:8002"
    await scheduler.release(node)
```

#### C2. Integration Test — Scheduler + Env Services
```bash
# 1. Start 2 env_services on different ports (local)
MAX_SLOTS=4 uvicorn seta_env.services.env_service:app --port 8002 &
MAX_SLOTS=4 uvicorn seta_env.services.env_service:app --port 8012 &

# 2. Write test nodes.yaml
cat > /tmp/test_nodes.yaml <<EOF
nodes:
  - url: "http://127.0.0.1:8002"
    slots: 4
  - url: "http://127.0.0.1:8012"
    slots: 4
EOF

# 3. Start scheduler
NODES_YAML=/tmp/test_nodes.yaml \
  uvicorn seta_env.services.env_scheduler:app --port 8003 &

# 4. Check status
curl http://127.0.0.1:8003/status

# 5. Send 4 requests with same task_id → check all go to same node
for i in 1 2 3 4; do
  curl -X POST http://127.0.0.1:8003/step \
    -H "Content-Type: application/json" \
    -H "X-API-Key: test" \
    -d '{"task": {"task_name": "same_task"}, "uid": "sess_'$i'", ...}' &
done
wait

# 6. Check status → one node should have 4 active (or recently active)
curl http://127.0.0.1:8003/status
```

---

### Phase D: GRPORollout Integration (Step 5)

#### D1. Unit Test — Mock Scheduler
```python
async def test_grpo_env_service():
    """GRPORollout with env_type=env_service sends correct HTTP requests."""
    # Start a mock FastAPI that returns fixed StepResponse
    # Configure GRPORollout with env_type="env_service"
    # Call rollout.run(task, n_trajs=4)
    # Verify 4 POST /step requests were made
    # Verify results match mock responses
```

#### D2. Integration Test — Full Pipeline (local)
```python
async def test_full_pipeline_local():
    """
    End-to-end: env_scheduler → env_service → TerminalEnvironment → Docker.
    Requires: local Docker, dataset, sglang (or mock model).
    """
    cfg = TerminalEnvConfig(
        runtime=RuntimeConfig(
            env_type="env_service",
            env_scheduler_url="http://127.0.0.1:8003",
        ),
        model=ModelConfig(
            model_platform="sglang",
            url="http://127.0.0.1:39001/v1",  # or FRP tunnel
        ),
    )
    rollout = GRPORollout(cfg)
    task = load_task("seta-env-harbor", "0")
    results = await rollout.run(task, n_trajs=2)
    assert len(results) == 2
    for run_info, reward in results:
        assert "task_name" in run_info
        assert reward is not None or "error" in run_info
```

---

### Phase E: Remote End-to-End (after deployment)

#### E1. Single Task on Remote Nodes
```bash
# 1. Deploy (Step 4)
bash start.sh --dataset seta-env-harbor

# 2. Single step request via scheduler
python3 -c "
import httpx, json, asyncio

async def main():
    async with httpx.AsyncClient(timeout=900) as c:
        r = await c.post('http://127.0.0.1:8003/step', json={
            'task': {'task_name': '0', 'instruction': '...'},
            'uid': 'e2e_test_1', 'traj_i': 0,
            'agent_config': {...},
            'model_config': {'url': 'http://65.108.32.175:39001/v1', ...},
            'runtime_config': {'trial_root': '/tmp/trials', 'environment_type': 'docker'},
            'env_config': {'reward_fn': 'pass_ratio'},
            'dataset_name': 'seta-env-harbor', 'task_name': '0',
        }, headers={'X-API-Key': 'env-service-dev-key'})
        print(json.dumps(r.json(), indent=2))

asyncio.run(main())
"
```

#### E2. GRPO Multi-Trajectory
```python
# test_grpo_remote.py
# Same as D2 but pointing at real remote servers
# Verify: co-location (all trajs on same node), correct results
```

#### E3. Concurrent Load
```bash
# Send 32 step requests (simulating 4 tasks × 8 trajs each)
# Monitor: scheduler /status, env_service /health on both nodes
# Verify: task_id affinity works, slots don't exceed 16 per node
```

---

## Test File Locations

| Test | File | Requires |
|------|------|----------|
| BuildGate unit | `seta_env/test/test_build_gate.py` | Nothing |
| Scheduler routing | `seta_env/test/test_env_scheduler.py` | Nothing |
| FRP tunnel smoke | `seta_env/services/frp_tunnel/test_tunnel.py` | FRP + sglang |
| Env service integration | `seta_env/test/test_env_service_integration.py` | Docker + dataset |
| GRPORollout integration | `seta_env/test/test_grpo_env_service.py` | Full stack |

## Success Criteria

1. FRP tunnel: test_tunnel.py passes with 64 concurrent requests, 0 errors
2. BuildGate: 10 concurrent requests for same task → exactly 1 build
3. Env service: POST /step returns valid run_info with reward
4. Scheduler: 8 requests same task_id → all to same node
5. Full pipeline: GRPORollout with env_type="env_service" produces same quality results as env_type="remote_docker"
