# Step 4: Deployment Scripts & Configuration

**Priority**: Medium — needed to deploy env_service to remote nodes.

**Status**: Not started

**Depends on**: Step 2 (env_service.py), Step 3 (env_scheduler.py)

## Overview

Deploy the full seta_env stack to remote CPU servers and provide one-click start/stop for the entire system. Follows the same patterns as `slot_pool_service/start.sh` and `deploy_node.sh` but deploys the full repo instead of a single file.

## Files to Create

### 4.1 `nodes.yaml` — Node Configuration

```yaml
nodes:
  - url: "http://65.108.32.175:8002"
    slots: 16
    deploy:
      ssh_key: /data/qijia/.ssh/id_ed25519
      ssh_user: root
      ssh_port: 22
      api_key: env-service-dev-key
      data_root: /data/harbor/dataset
      app_dir: /opt/env_service

  - url: "http://135.181.71.8:8002"
    slots: 16
    deploy:
      ssh_key: /data/qijia/.ssh/id_ed25519
      ssh_user: root
      ssh_port: 22
      api_key: env-service-dev-key
      data_root: /data/harbor/dataset
      app_dir: /opt/env_service
```

### 4.2 `deploy_env_service.sh` — Per-Node Deploy Script

**Usage**:
```bash
./deploy_env_service.sh <host> <ssh_key> [options]
```

**Options**: Same as deploy_node.sh plus additional ones.

**Key difference from deploy_node.sh**: This deploys the full terminal_agent repo, not just node_manager.py. The env_service imports TerminalEnvironment, which needs camel, harbor, seta_env, and all their dependencies.

**Steps**:

```
[1/8] Install system packages
  apt-get install python3-pip python3-venv python3-dev git git-lfs curl docker.io
  # Also need build tools for some Python packages:
  apt-get install build-essential

[2/8] Configure Docker daemon (subnet pools)
  # Same as deploy_node.sh — 10.200.0.0/16 /28 pools
  # Restart Docker

[3/8] Rsync terminal_agent repo to remote
  rsync -az --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'outputs/' \
    --exclude 'dataset/' \
    --exclude '.venv/' \
    --exclude 'external/areal/' \
    /data/qijia/terminal_agent/ \
    ${SSH_USER}@${HOST}:${APP_DIR}/terminal_agent/

  # Note: exclude external/areal (not needed on env_service nodes)
  # Note: exclude dataset/ (downloaded separately via /setup endpoint)

[4/8] Create virtualenv and install dependencies
  python3 -m venv /opt/env_service_venv
  source /opt/env_service_venv/bin/activate

  # Install the project and its deps
  cd ${APP_DIR}/terminal_agent
  pip install -e external/camel
  pip install -e external/harbor   # if exists
  pip install -e .

  # Service deps (may already be in requirements.txt)
  pip install fastapi "uvicorn[standard]" httpx aiofiles PyYAML

[5/8] Write systemd service
  cat > /etc/systemd/system/env-service.service <<EOF
  [Unit]
  Description=Env Service (TerminalEnvironment executor)
  After=network.target docker.service
  Requires=docker.service

  [Service]
  ExecStart=/opt/env_service_venv/bin/uvicorn \
    seta_env.services.env_service:app \
    --host 0.0.0.0 --port 8002 --workers 1
  WorkingDirectory=${APP_DIR}/terminal_agent
  Environment=PYTHONPATH=${APP_DIR}/terminal_agent
  Environment=ENV_SERVICE_API_KEY=${API_KEY}
  Environment=DATASET_ROOT=${DATA_ROOT}
  Environment=HARBOR_ROOT=/tmp/harbor
  Environment=MAX_SLOTS=${SLOTS}
  Environment=HF_TOKEN=${HF_TOKEN:-}
  Restart=always
  RestartSec=5
  StandardOutput=journal
  StandardError=journal

  [Install]
  WantedBy=multi-user.target
  EOF
  systemctl daemon-reload

[6/8] Open firewall port
  ufw allow 8002/tcp || firewall-cmd ... || iptables ...

[7/8] Start service
  systemctl enable env-service
  systemctl restart env-service

[8/8] Health check
  curl -sf http://${HOST}:8002/health
```

### 4.3 `start.sh` — One-Click Startup

**Usage**:
```bash
# Full deploy + start
bash start.sh

# Skip deploy (nodes already running)
bash start.sh --skip-deploy

# With dataset activation
bash start.sh --dataset seta-env-harbor

# Daemon mode
bash start.sh --daemon
bash start.sh --stop

# Custom scheduler port
bash start.sh --port 8003 --host 127.0.0.1

# Local-only mode (no remote nodes, env_service on same machine)
bash start.sh --local --port 8003
```

**Steps**:
```
1. Parse args (same flags as slot_pool_service/start.sh + --local)
2. If not --skip-deploy:
   - Parse nodes.yaml for nodes with deploy: block
   - Run deploy_env_service.sh for each node in parallel
   - Wait for all deployments to complete
3. If --local:
   - Start env_service locally in background (port 8002)
4. Start env_scheduler locally (default port 8003)
5. If --dataset provided:
   - Wait for scheduler to be ready
   - POST /setup_dataset to scheduler (fans out to all nodes)
6. Block until Ctrl-C (or background if --daemon)
```

### 4.4 `setup_dataset.sh` — Fan Out Dataset

```bash
#!/usr/bin/env bash
# Usage: ./setup_dataset.sh <dataset_name> [--scheduler http://127.0.0.1:8003]
#
# Sends POST /setup_dataset to the env_scheduler, which fans out to all nodes.

DATASET_NAME="$1"
SCHEDULER_URL="${2:-http://127.0.0.1:8003}"

curl -sf -X POST "${SCHEDULER_URL}/setup_dataset" \
  -H "Content-Type: application/json" \
  -d "{\"dataset_name\": \"${DATASET_NAME}\"}"
```

## Deployment Diagram

```
Local (GPU) Machine                     Remote CPU Servers
┌───────────────────────┐               ┌──────────────────────┐
│ env_scheduler :8003   │               │ Server 1             │
│   ├─ routes requests  │               │ (65.108.32.175)      │
│   └─ reads nodes.yaml │──── HTTP ────►│ env_service :8002    │
│                       │               │   ├─ BuildGate       │
│ (optional)            │               │   ├─ Semaphore(16)   │
│ env_service :8002     │               │   └─ TerminalEnv     │
│   (local mode)        │               └──────────────────────┘
│                       │               ┌──────────────────────┐
│ GRPORollout           │               │ Server 2             │
│   └─ POST /step ──────│──── HTTP ────►│ (135.181.71.8)       │
│      to scheduler     │               │ env_service :8002    │
└───────────────────────┘               └──────────────────────┘
```

## Re-deployment (code updates)

When seta_env code changes:
```bash
# Re-rsync and restart (no full redeploy needed)
bash deploy_env_service.sh 65.108.32.175 /data/qijia/.ssh/id_ed25519 --skip-deps
```

Add `--skip-deps` flag to deploy_env_service.sh that skips apt-get and pip install, only rsyncs code and restarts the service.

## Testing

1. Deploy to one server: `bash deploy_env_service.sh 65.108.32.175 /data/qijia/.ssh/id_ed25519`
2. Check health: `curl http://65.108.32.175:8002/health`
3. Start scheduler: `bash start.sh --skip-deploy`
4. Check scheduler: `curl http://127.0.0.1:8003/status`
5. Setup dataset: `bash setup_dataset.sh seta-env-harbor`
6. Verify dataset on nodes: `curl http://65.108.32.175:8002/health` (should show active dataset)
