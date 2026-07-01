# Step 2: Remote Env Service

**Priority**: High — core service that runs TerminalEnvironment.step() on remote nodes.

**Status**: Not started

**Depends on**: Step 1 (FRP tunnel) for sglang access from remote nodes

## Overview

FastAPI service deployed on each CPU server (and optionally locally on the GPU machine). Receives a full task payload, runs the complete TerminalEnvironment lifecycle (build → agent → evaluate → reward → cleanup), and returns (run_info, reward).

Unlike `node_manager.py` (which is a thin Docker management layer), env_service runs the full seta_env stack — agents, models, verifiers, reward functions.

## File: `seta_env/services/env_service.py`

### Request / Response Models

```python
class StepRequest(BaseModel):
    """Same arguments as TerminalEnvironment.step() + configs to construct it."""
    # TerminalEnvironment.step() args
    task: dict              # {"task_name": str, "instruction": str, ...}
    uid: str                # unique session ID (e.g., "taskA_t0_abc123")
    traj_i: int = 0

    # TerminalEnvironment constructor args (all serializable dicts)
    agent_config: dict      # {agent, prompt, max_total_tokens, max_iteration, tool_names, ...}
    model_config: dict      # {model_platform, model_type, url, api_key, model_config_dict}
    runtime_config: dict    # {trial_root, environment_type="docker"}  -- task_dir rewritten on server
    env_config: dict        # {reward_fn, task_timeouts}

    # Dataset info for path resolution
    dataset_name: str       # e.g., "seta-env-harbor"
    task_name: str          # e.g., "0" or "my_task" — key within the dataset

class StepResponse(BaseModel):
    run_info: dict | None = None
    reward: float | None = None
    error: str | None = None
```

### Build Gate — Per-task_id Single-Flight

The critical concurrency pattern. Multiple GRPO trajectories for the same task arrive as separate requests. Only the first triggers the Docker image build; others wait.

```python
@dataclass
class BuildState:
    status: Literal["building", "built", "failed"]
    event: asyncio.Event       # signaled when build completes
    error: str | None = None

class BuildGate:
    def __init__(self):
        self._gate_lock = asyncio.Lock()      # protects _registry mutations only
        self._registry: dict[str, BuildState] = {}
        self._timestamps: dict[str, float] = {}

    async def ensure_built(self, task_name: str, build_fn: Callable) -> None:
        """
        Ensure Docker image for task_name is built.
        First caller triggers build_fn(); subsequent callers wait.
        Different task_names run in parallel (independent events).
        """
        # Phase 1: fast registry check (gate_lock held ~microseconds, no I/O)
        async with self._gate_lock:
            if task_name not in self._registry:
                self._registry[task_name] = BuildState(
                    status="building", event=asyncio.Event()
                )
                is_builder = True
            else:
                is_builder = False
            state = self._registry[task_name]

        # Phase 2: build or wait (NO lock held, fully concurrent)
        if is_builder:
            try:
                await build_fn()
                state.status = "built"
            except Exception as e:
                state.status = "failed"
                state.error = str(e)
            finally:
                self._timestamps[task_name] = time.monotonic()
                state.event.set()  # wake ALL waiters
        else:
            if state.status == "building":
                await state.event.wait()  # suspends coroutine, costs nothing

        # Phase 3: check result
        if state.status == "failed":
            raise RuntimeError(f"Build failed for {task_name}: {state.error}")

    def clear(self, older_than: float = 3600.0) -> int:
        """Remove entries older than TTL. Called by GC loop."""
        now = time.monotonic()
        to_remove = [k for k, t in self._timestamps.items()
                     if now - t > older_than]
        for k in to_remove:
            self._registry.pop(k, None)
            self._timestamps.pop(k, None)
        return len(to_remove)
```

**Concurrency timeline showing parallel task_ids**:
```
task_name="abc" req1 ─[gate: register]─[build ~~~~~~~~~~~~]─[set event]─[run env]
task_name="abc" req2 ────[gate: check]─[event.wait ~~~~~~~]────────────[run env]
task_name="abc" req3 ────[gate: check]─[event.wait ~~~~~~~]────────────[run env]

task_name="xyz" req1 ─[gate: register]─[build ~~~~~~~~]─[set event]──[run env]
task_name="xyz" req2 ──────[gate: check]─[event.wait ~~]────────────[run env]
```

### Slot Semaphore

Limits concurrent TerminalEnvironment.step() calls to the node's slot count:

```python
MAX_SLOTS = int(os.environ.get("MAX_SLOTS", "16"))
_slot_semaphore = asyncio.Semaphore(MAX_SLOTS)
```

### Main Endpoint

```python
@app.post("/step")
async def step(req: StepRequest, x_api_key: str = Header(...)):
    verify_api_key(x_api_key)

    # 1. Resolve task_dir to local dataset path
    task_dir = str(DATASET_ROOT / req.dataset_name / req.task_name)
    runtime_config = {**req.runtime_config, "task_dir": task_dir}

    # 2. Build gate — ensure image is built for this task
    async def build_fn():
        te = TerminalEnvironment(
            agent_config=req.agent_config,
            model_config=req.model_config,
            runtime_config={**runtime_config, "environment_type": "docker"},
            env_config=req.env_config,
        )
        runtime = DockerHarborRuntime(task_dir=task_dir, ...)
        await runtime.build()
        await runtime.stop()

    await _build_gate.ensure_built(req.task_name, build_fn)

    # 3. Acquire slot
    async with _slot_semaphore:
        # 4. Run TerminalEnvironment.step()
        te = TerminalEnvironment(
            agent_config=req.agent_config,
            model_config=req.model_config,
            runtime_config=runtime_config,
            env_config=req.env_config,
        )
        try:
            run_info, reward = await te.step(req.task, uid=req.uid, traj_i=req.traj_i)
            return StepResponse(run_info=run_info, reward=reward)
        except Exception as e:
            return StepResponse(error=str(e))
```

### Additional Endpoints

```python
@app.get("/health")
async def health():
    """Status + active count + semaphore availability."""
    return {
        "status": "ok",
        "max_slots": MAX_SLOTS,
        "available_slots": _slot_semaphore._value,  # approximate
        "active_builds": sum(1 for s in _build_gate._registry.values()
                            if s.status == "building"),
        "built_images": sum(1 for s in _build_gate._registry.values()
                            if s.status == "built"),
    }

@app.post("/setup")
async def setup_dataset(req: SetupRequest, ...):
    """Download/activate dataset — reuse node_manager's git clone pattern."""
    ...

@app.post("/cleanup")
async def cleanup(...):
    """Force-stop all running containers. Reset build gate and semaphore."""
    ...
```

### GC Loop

Background task that runs every 5 minutes:
- Clear BuildGate entries older than 1 hour
- Kill Docker containers that have been running > 2 hours (leaked sessions)
- Log stats

### Local Mode

When running on the GPU machine alongside sglang:
- `model_config.url` points to `http://127.0.0.1:<sglang_port>/v1`
- `environment_type = "docker"` — uses local Docker
- No FRP tunnel needed
- nodes.yaml includes `url: "http://127.0.0.1:8002"` for this node

## Configuration

Environment variables:
```bash
MAX_SLOTS=16                           # concurrent step() calls
ENV_SERVICE_API_KEY=env-service-dev-key
DATASET_ROOT=/data/harbor/dataset      # where datasets are stored
HARBOR_ROOT=/tmp/harbor                # docker compose working dir
```

## Error Handling

| Error | Behavior |
|-------|----------|
| Build fails | BuildGate marks "failed", all waiting requests get error response |
| step() timeout | TerminalEnvironment has internal timeouts per stage, returns error in run_info |
| step() exception | Caught, returned as StepResponse(error=str(e)) |
| Semaphore full | Request blocks until a slot frees up (httpx client has 900s timeout) |

## Testing

### Unit test: BuildGate
- 10 concurrent requests for same task_name → only 1 build_fn call
- 2 different task_names → 2 parallel build_fn calls
- Build failure → all waiters get RuntimeError

### Unit test: Semaphore
- MAX_SLOTS=2, send 4 requests → only 2 run concurrently

### Integration test (local):
- Start env_service on localhost:8002
- POST /step with a simple task from seta-env-harbor dataset
- Verify response has run_info with task_name, reward, timings

## Dependencies

Full seta_env stack:
- `camel-ai` (agents, models, toolkits)
- `harbor` (environments, tasks, verifiers)
- `seta_env` (agent, reward, configs)
- `fastapi`, `uvicorn`, `httpx`, `pydantic`
- `openai` (for model calls to sglang)
- Docker (for container management)
