# Remote Docker Rollout — Implementation & Test Plans

## Overview

This folder documents the staged implementation of remote Docker container execution
for GRPO rollout across multiple cloud nodes, integrated with the AReaL training workflow.

Each stage has a concrete implementation plan and a test plan.
Stages must be tested in order — each builds on the layer below.

## Package Root

```
<REPO_ROOT>/seta_env/
```

## Remote Node for Experimentation

```
95.133.253.67   (SSH access available, Docker installed)
```

## Concurrency Targets

- Max **16 concurrent Docker builds** per node (semaphore in node manager)
- Max **group size 16** per GRPO rollout (scheduler enforces)
- Max **256 concurrent HTTP connections** from training machine to node managers

## Implementation & Test Stages

| # | Plan File | Impl File(s) | What |
|---|-----------|--------------|------|
| 1 | [01_node_manager.md](01_node_manager.md) | `seta_env/environments/node_manager.py` | FastAPI service on each cloud node — build, compose up/down, exec, file transfer |
| 2 | [02_remote_docker_environment.md](02_remote_docker_environment.md) | `harbor/.../remote_docker_environment.py` | `BaseEnvironment` subclass using httpx to talk to node manager |
| 3 | [03_runtime_build.md](03_runtime_build.md) | `seta_env/runtimes/docker_harbor_runtime.py` (modify) | Add `build()` method to runtime, separate from `reset()` |
| 4 | [04_scheduler_service.md](04_scheduler_service.md) | `seta_env/environments/scheduler_service.py` | FastAPI group-based slot allocator running locally |
| 5 | [05_remote_terminal_env.md](05_remote_terminal_env.md) | — (no new files) | Wire `RemoteDockerEnvironment` into `TerminalEnvironment.step()` via `runtime_config["environment"]` |
| 6 | [06_concurrent_grpo_rollout.md](06_concurrent_grpo_rollout.md) | `seta_env/environments/grpo_rollout.py` | Orchestrate N concurrent trajectories with scheduler + pre-build |
| 7 | [07_areal_workflow.md](07_areal_workflow.md) | `src/tbench_areal_workflow/train_remote.py` | New `CamelRLVRWorkflow` using remote rollout, replaces `CamelTerminalAgent` |

## Key Conventions

- All tests use `pytest` + `pytest-asyncio`.
- Async tests need `@pytest.mark.asyncio`.
- API key for node manager is set via `NODE_MANAGER_API_KEY` env var.
- Node manager must be running on `95.133.253.67:8001` for stages 1–7.
- Scheduler service must be running locally on `localhost:8000` for stages 4–7.
- Tests that hit real remote nodes are tagged `@pytest.mark.remote` and skipped if
  `NODE_MANAGER_URL` env var is not set.

## Chosen Task for Testing (stages 1–7)

```
<REPO_ROOT>/dataset/seta-env-harbor/0
```

- **Category:** software-engineering, difficulty: medium
- **Environment:** single `Dockerfile` under `environment/`
- **Tests:** `tests/test.sh` + `tests/test_outputs.py`
- **Reward:** written to `$VERIFIER_DIR/reward.txt` or `reward.json`

## Available Dataset

```
<REPO_ROOT>/dataset/seta-env-harbor/
```

Each numbered subfolder is a self-contained Harbor-format task.

## Node Manager Deployment (one-time setup via SSH)

```bash
# On 95.133.253.67:
pip install fastapi uvicorn httpx
# copy node_manager.py
NODE_MANAGER_API_KEY=<secret> uvicorn node_manager:app --host 0.0.0.0 --port 8001
```

Or as a systemd service — see Stage 1 plan for details.

## Architecture Summary

```
Local training machine
  ├── Scheduler Service (localhost:8000)   — in-memory slot state, no docker
  │
  │   HTTP + X-API-Key header
  ├──► Node Manager (95.133.253.67:8001)  — wraps local docker compose
  └──► Node Manager (node-2:8001)

Data flow per trajectory:
  1. scheduler.allocate_group(task_id, n_trajs) → [(node_url, slot_id), ...]
  2. node_manager.build(task)               → docker compose build on remote
  3. node_manager.compose_up(session_id)    → docker compose up -d
  4. runtime.exec(...)                      → docker compose exec (via node manager)
  5. node_manager.download(verifier_dir)    → reward.txt back to local
  6. node_manager.compose_down(session_id) → docker compose down
  7. scheduler.release_group(task_id)
```
