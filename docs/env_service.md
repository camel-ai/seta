# Env Service — Remote TerminalEnvironment Execution

Run TerminalEnvironment on remote CPU servers while training on GPU machines.

## Architecture

```
GPU Machine A                          CPU Server 1 (+ optional FRP relay)
┌──────────────────────────┐           ┌──────────────────────────────────┐
│ AReaL launcher           │           │ frps :7000 (relay, optional)     │
│  ├─ sglang (DP4)         │           │                                  │
│  │   Rank 0 :PORT_0      │           │ env_service :8002                │
│  │   Rank 1 :PORT_1      │           │  ├─ TerminalEnvConfig (owns)     │
│  │   Rank 2 :PORT_2      │           │  ├─ BuildGate (per-task build)   │
│  │   Rank 3 :PORT_3      │           │  ├─ Semaphore (16 slots)         │
│  │                        │           │  └─ Docker containers            │
│  ├─ ProxyServer per rank  │           │                                  │
│  │   :PROXY_0 → Rank 0   │           │ :39001 ─── frps relay ──► local  │
│  │   :PROXY_1 → Rank 1   │  frpc     │ :39002     (optional)            │
│  │   :PROXY_2 → Rank 2   │ ────────► │ :39003                           │
│  │   :PROXY_3 → Rank 3   │ outbound  │ :39004                           │
│  │                        │           └──────────────────────────────────┘
│  ├─ eval/train script     │
│  │   setup_proxy_tunnels()│           CPU Server 2
│  │   (auto frpc + rewrite)│           ┌──────────────────────────────────┐
│  │                        │           │ env_service :8002                │
│  └─ env_scheduler :8003   │── HTTP ──►│  ├─ TerminalEnvConfig (owns)     │
│      ├─ task_id affinity  │           │  ├─ BuildGate                    │
│      ├─ load balancing    │           │  ├─ Semaphore (16 slots)         │
│      └─ url_rewrite       │           │  └─ Docker containers            │
└──────────────────────────┘           └──────────────────────────────────┘

GPU Machine B (optional, joins later)
┌──────────────────────────┐
│ sglang (DP2)             │  frpc
│  Rank 0, Rank 1          │ ────────► same relay :39101, :39102
│  ProxyServer per rank    │
│  env_scheduler :8003     │── HTTP ──► same CPU servers
└──────────────────────────┘

Data flow per trajectory:
  1. ProxySession → start session → get session_id
  2. StepRequest to scheduler (model_url = proxy via relay or direct)
  3. Scheduler: url_rewrite + route by task affinity → env_service node
  4. env_service: build image → run agent → agent calls model via proxy URL
  5. Model calls → [FRP tunnel if needed] → ProxyServer → sglang
  6. env_service returns (run_info, reward) → ProxySession sets reward
  7. ProxyServer.get_completions() → AReaL training data

Three network modes (set in nodes.yaml url_rewrite):
  FRP:       "http://172.17.0.2:8400" → "http://cpu-server-1:39001"
  RunPod:    "http://172.17.0.2:8400" → "https://xyz-8400.proxy.runpod.net"
  Direct:    url_rewrite: {}  (no rewrite needed)
```

## Folder Structure

```
seta_env/services/
├── env_service.py           # Remote FastAPI service (deployed per node)
├── env_scheduler.py         # Local request router
├── proxy_setup.py           # Auto FRP + url_rewrite setup
├── deploy_env_service.sh    # Per-node deploy (git clone + pip install)
├── start.sh                 # Single entrypoint: deploy all + scheduler + dataset
├── nodes.yaml               # Node config (gitignored, create per deployment)
├── frp_tunnel/
│   ├── manage_tunnel.py     # FRP orchestrator (deploy-relay, start, stop, status)
│   ├── tunnel_config.yaml   # FRP config (relay host, GPU machine port ranges)
│   ├── frps_start.sh        # Start relay on CPU server
│   ├── frpc_start.sh        # Start client on GPU machine
│   └── test_tunnel.py       # HTTP smoke + load test
├── tests/                   # 25 unit tests
└── plans/                   # Implementation design docs

scripts/areal/
├── workflow_env_service.py                  # AReaL workflow using env_service
├── eval_env_service.py                      # Evaluation script
├── rl_train_env_service.py                  # Training script
└── configs/
    └── config_eval_env_service_seta_v2.yaml # Example eval config
```

## Setup from Scratch

### 1. Create `nodes.yaml`

```bash
cd seta_env/services
cat > nodes.yaml <<'EOF'
url_rewrite: {}
api_key: env-service-dev-key

nodes:
  - url: "http://<CPU_SERVER_1>:8002"
    slots: 16
    deploy:
      ssh_key: ~/.ssh/id_ed25519
      ssh_user: root
      api_key: env-service-dev-key

  - url: "http://<CPU_SERVER_2>:8002"
    slots: 16
    deploy:
      ssh_key: ~/.ssh/id_ed25519
      ssh_user: root
      api_key: env-service-dev-key
EOF
```

### 2. Deploy + start (one command)

```bash
GH_TOKEN=ghp_xxx HF_TOKEN=hf_xxx bash seta_env/services/start.sh --dataset seta-env-v2
```

Deploys env_service to all nodes in parallel, starts the scheduler, downloads dataset.

### 3. FRP relay (one-time, only if GPU is in Docker with no inbound ports)

```bash
cd seta_env/services/frp_tunnel
# Edit tunnel_config.yaml: set relay host and GPU machine port ranges
python manage_tunnel.py deploy-relay
```

FRP client starts automatically per run via `setup_proxy_tunnels()`.

### 4. Run evaluation

```bash
python -m areal.launcher.local \
    scripts/areal/eval_env_service.py \
    --config scripts/areal/configs/config_eval_env_service_seta_v2.yaml \
    allocation_mode=sglang:d4p1t1+eval
```

### 5. Run training

```bash
python -m areal.launcher.local \
    scripts/areal/rl_train_env_service.py \
    --config scripts/areal/configs/config_train_env_service.yaml
```

## Key Design Decisions

- **env_service owns its config**: Caller sends only task + model URL. Config fan-out via `POST /config` on scheduler.
- **BuildGate**: Per-task single-flight. First request builds Docker image; others wait. Different tasks build in parallel.
- **Task affinity**: Same task_id → same node within 2-minute window (reuses built images).
- **url_rewrite**: Scheduler rewrites model URLs before forwarding. One config for FRP/RunPod/direct.
- **Trial logs**: Organized by `trial_name` on remote (`/opt/env_service/data/trials/<trial_name>/`) and locally.
- **Multi-GPU**: Each GPU machine gets its own FRP port range in `tunnel_config.yaml`. Adding a machine = add config entry.

## Operations

```bash
# Code update (git pull + restart, ~15s)
GH_TOKEN=ghp_xxx bash seta_env/services/start.sh --skip-deps

# Full redeploy (pip reinstall, ~5min)
GH_TOKEN=ghp_xxx HF_TOKEN=hf_xxx bash seta_env/services/start.sh --dataset seta-env-v2

# Docker cleanup on all nodes
curl -X POST http://127.0.0.1:8003/cleanup

# Update config on all nodes
curl -X POST http://127.0.0.1:8003/config \
  -H "Content-Type: application/json" \
  -d '{"agent": {"max_iteration": 50}}'

# Check status
curl http://127.0.0.1:8003/status
```
