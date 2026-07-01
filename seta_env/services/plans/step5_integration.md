# Step 5: Integration — Proxy + Env Service Workflow

**Status**: Implemented

## Architecture

Same pattern as `gsm8k_grpo_proxy.py`:

```
GPU Machine                               CPU Server (env_service)
┌─────────────────────────────┐           ┌──────────────────────────┐
│ rl_train_env_service.py     │           │ env_service :8002        │
│   │                         │           │   (owns TerminalEnvConfig)│
│   ├─ ProxyServer :PORT      │◄── FRP ───│   │                      │
│   │   (ArealOpenAI client)  │  or direct│   ├─ TerminalEnvironment  │
│   │                         │           │   │   agent calls model   │
│   ├─ ProcessPoolExecutor    │           │   │   via OPENAI_BASE_URL │
│   │   └─ per traj process:  │           │   │   (ProxySession URL)  │
│   │      ProxySession       │           │   │                       │
│   │      → OPENAI_BASE_URL  │           │   └─ returns run_info     │
│   │      → OPENAI_API_KEY   │           │                           │
│   │      → HTTP to scheduler│── HTTP ──►│                           │
│   │      → set_reward()     │           └───────────────────────────┘
│   │                         │
│   └─ get_completions()      │           env_scheduler :8003
│       from ProxyServer      │           ┌───────────────────────┐
│       for training          │── HTTP ──►│ routes by task_id     │
│                             │           │ url_rewrite model_url │
└─────────────────────────────┘           │ forwards to best node │
                                          └───────────────────────┘
```

## Data Flow per Episode

1. **GPU**: ProcessPoolExecutor submits N tasks (one per trajectory)
2. **Each process**: `async with ProxySession(proxy_addr)` → gets session_id
   - ProxySession sets `OPENAI_BASE_URL = proxy_addr/v1/{session_id}`
   - ProxySession sets `OPENAI_API_KEY = session_id`
3. **Each process**: Sends thin StepRequest to env_scheduler:
   - `model_url = OPENAI_BASE_URL` (from ProxySession)
   - `model_api_key = OPENAI_API_KEY` (= session_id)
   - `task`, `uid`, `dataset_name`, `task_name`
4. **Scheduler**: Applies `url_rewrite` (FRP/RunPod/direct), forwards to best node
5. **env_service**: Uses own config + request's model_url → creates TerminalEnvironment → runs agent
6. **Agent**: Calls model at model_url → goes through FRP tunnel → hits ProxyServer on GPU
7. **ProxyServer**: Routes by session_id, captures interactions via ArealOpenAI
8. **env_service**: Returns (run_info, reward) via HTTP
9. **Each process**: `session.set_reward(reward)` → ProxySession ends session
10. **GPU**: `proxy_server.get_completions(session_ids)` → AReaL training data

## Files

| File | Purpose |
|------|---------|
| `scripts/areal/workflow_env_service.py` | `EnvServiceRLVRWorkflow` + `_sync_run_task` |
| `scripts/areal/rl_train_env_service.py` | Training script using env_service workflow |
| `seta_env/services/env_service.py` | Owns TerminalEnvConfig, receives thin StepRequest |
| `seta_env/services/env_scheduler.py` | Routes + url_rewrite |
| `seta_env/services/nodes.yaml` | `url_rewrite` map for network topology |

## What Changed from Original Design

- **env_service owns its config** — StepRequest is thin (task + model_url + api_key)
- **Config fan-out** via `POST /config` on scheduler → fans to all nodes
- **url_rewrite in scheduler** — handles FRP/RunPod/direct transparently
- **No changes to grpo_rollout.py or configs.py** — env_service is a separate layer
- **ProcessPoolExecutor** for ProxySession env var isolation (same as gsm8k_grpo_proxy)

## Three Network Scenarios

```yaml
# nodes.yaml

# FRP tunnel:
url_rewrite:
  "http://172.17.0.2:8400": "http://65.108.32.175:39001"

# RunPod:
url_rewrite:
  "http://172.17.0.2:8400": "https://xyz-8400.proxy.runpod.net"

# Direct (public IP / same network):
url_rewrite: {}
```
