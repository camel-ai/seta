# Env Services Usage

## Quick Start (one command)

```bash
cd seta_env/services

# First time: deploy + start scheduler + setup dataset (~5 min)
HF_TOKEN=hf_... bash start.sh --dataset seta-env-v2

# Code update: redeploy code + restart + dataset (~20s)
HF_TOKEN=hf_... bash start.sh --skip-deps --dataset seta-env-v2

# Scheduler only (nodes already running):
bash start.sh --skip-deploy

# Background mode:
HF_TOKEN=hf_... bash start.sh --daemon --dataset seta-env-v2
bash start.sh --stop
```

`start.sh` reads `nodes.yaml`, fans out deploy + dataset setup to all nodes in parallel, then starts the scheduler locally.

## Run Evaluation

```bash
# 1. Start services (if not already running)
cd seta_env/services
HF_TOKEN=hf_... bash start.sh --skip-deploy --dataset seta-env-v2

# 2. (If FRP needed) Start tunnel
cd seta_env/services/frp_tunnel
python manage_tunnel.py deploy-relay          # one-time
python manage_tunnel.py start gpu-a --ranks "<sglang_ip>:<port>"

# 3. Configure url_rewrite in nodes.yaml (if FRP)
#    url_rewrite:
#      "http://<proxy_internal>:<port>": "http://65.108.32.175:39001"

# 4. Run eval (launcher starts sglang + sets RANK automatically)
cd /data/qijia/terminal_agent
python -m areal.launcher.local \
  scripts/areal/eval_env_service.py \
  --config scripts/areal/configs/config_eval_env_service_seta_v2.yaml \
  allocation_mode=sglang:d1p1t1+eval
```

## Update Config on All Nodes

```bash
# Fan out via scheduler (no redeployment needed)
curl -X POST http://127.0.0.1:8003/config \
  -H "Content-Type: application/json" \
  -d '{"agent": {"max_iteration": 50, "thinking": true}, ...}'
```

## FRP Tunnel

```bash
cd seta_env/services/frp_tunnel
python manage_tunnel.py deploy-relay                        # one-time relay setup
python manage_tunnel.py start gpu-a --ranks "ip:port,..."   # per training run
python manage_tunnel.py status                              # check all tunnels
python manage_tunnel.py stop gpu-a                          # cleanup
```

## Monitoring

```bash
curl http://127.0.0.1:8003/status         # scheduler: slots, affinity
curl http://65.108.32.175:8002/health      # node health + active steps
curl http://65.108.32.175:8002/config      # current TerminalEnvConfig
```

## Adding a New Node

1. Add entry to `nodes.yaml` with `deploy:` block
2. Run `bash start.sh --dataset seta-env-v2` (deploys new node + existing nodes are idempotent)
