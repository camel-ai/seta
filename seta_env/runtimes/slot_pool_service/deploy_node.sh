#!/usr/bin/env bash
# deploy_node.sh — one-click setup of the node manager on a remote server
#
# Usage:
#   ./deploy_node.sh <host> <ssh_key> [options]
#
# Required:
#   host        IP or hostname of the remote node, e.g. 95.133.253.67
#   ssh_key     Path to SSH private key, e.g. ~/.ssh/id_ed25519
#
# Options:
#   --user      SSH user             (default: root)
#   --port      SSH port             (default: 8001)
#   --api-key   NODE_MANAGER_API_KEY (default: harbor-node-dev-key)
#   --data-root Remote dir for datasets (default: /data/harbor/dataset)
#   --app-dir   Remote dir for app files (default: /opt/node_manager)
#
# Example (with defaults, prompts for API key):
#   ./deploy_node.sh 95.133.253.67 ~/.ssh/id_ed25519
#
# Example (fully scripted):
#   ./deploy_node.sh 95.133.253.67 ~/.ssh/id_ed25519 \
#       --user root --port 8001 --api-key mysecret \
#       --data-root /data/harbor/dataset --app-dir /opt/node_manager

set -euo pipefail

# ── Parse args ────────────────────────────────────────────────────────────────

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <host> <ssh_key> [--user USER] [--port PORT] [--api-key KEY] [--data-root DIR] [--app-dir DIR]"
    exit 1
fi

HOST="$1"; shift
SSH_KEY="$1"; shift

SSH_USER="root"
SERVICE_PORT="8001"
API_KEY="${NODE_MANAGER_API_KEY:-harbor-node-dev-key}"
DATA_ROOT="/data/harbor/dataset"
APP_DIR="/opt/node_manager"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --user)      SSH_USER="$2";    shift 2 ;;
        --port)      SERVICE_PORT="$2"; shift 2 ;;
        --api-key)   API_KEY="$2";     shift 2 ;;
        --data-root) DATA_ROOT="$2";   shift 2 ;;
        --app-dir)   APP_DIR="$2";     shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no -o BatchMode=yes"

echo "=== Deploying node manager to ${SSH_USER}@${HOST} ==="
echo "    App dir:   $APP_DIR"
echo "    Data root: $DATA_ROOT"
echo "    Port:      $SERVICE_PORT"
echo ""

# ── Helper: run command on remote ─────────────────────────────────────────────

remote() {
    ssh $SSH_OPTS "${SSH_USER}@${HOST}" "$@"
}

# ── Step 1: Install system deps ───────────────────────────────────────────────

echo "[1/7] Installing system packages..."
remote bash -s <<'REMOTE'
set -euo pipefail
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv git git-lfs curl 2>&1 | tail -5
git lfs install
echo "apt done"
REMOTE

# ── Step 2: Configure Docker daemon (subnet pools) ────────────────────────────

echo "[2/7] Configuring Docker daemon subnet pools..."
remote bash -s <<'REMOTE'
set -euo pipefail
DOCKER_DAEMON_CONFIG=/etc/docker/daemon.json
mkdir -p /etc/docker
# Merge subnet config; preserve any existing keys
if [ -f "$DOCKER_DAEMON_CONFIG" ] && command -v python3 &>/dev/null; then
    python3 - <<'PY'
import json, sys
cfg = json.load(open("/etc/docker/daemon.json")) if open("/etc/docker/daemon.json").read().strip() else {}
cfg["default-address-pools"] = [{"base": "10.200.0.0/16", "size": 28}]
json.dump(cfg, open("/etc/docker/daemon.json", "w"), indent=2)
PY
else
    cat > "$DOCKER_DAEMON_CONFIG" <<EOF
{
  "default-address-pools": [
    {
      "base": "10.200.0.0/16",
      "size": 28
    }
  ]
}
EOF
fi
systemctl restart docker || true
# Verify docker actually came up (restart can return non-zero on some nodes due to nftables warnings)
for i in $(seq 1 10); do
    systemctl is-active docker &>/dev/null && break
    sleep 2
done
systemctl is-active docker || { echo "ERROR: Docker failed to start after restart"; exit 1; }
echo "docker daemon configured"
REMOTE

# ── Step 3: Create virtualenv + install Python deps ───────────────────────────

echo "[3/7] Creating virtualenv and installing Python packages..."
remote bash -s <<REMOTE
set -euo pipefail
python3 -m venv /opt/node_manager_venv
/opt/node_manager_venv/bin/pip install -q --upgrade pip
/opt/node_manager_venv/bin/pip install -q \
    fastapi \
    "uvicorn[standard]" \
    httpx \
    aiofiles \
    python-multipart \
    PyYAML
echo "pip done"
REMOTE

# ── Step 4: Copy app files ────────────────────────────────────────────────────

echo "[4/7] Copying node manager files..."
remote mkdir -p "$APP_DIR" "$DATA_ROOT"

scp $SSH_OPTS \
    "$SCRIPT_DIR/node_manager.py" \
    "$REPO_ROOT/seta_env/dataset/datasets.yaml" \
    "$SCRIPT_DIR/docker-compose-build.yaml" \
    "$SCRIPT_DIR/docker-compose-prebuilt.yaml" \
    "${SSH_USER}@${HOST}:${APP_DIR}/"

echo "    Copied: node_manager.py datasets.yaml (from seta_env/dataset/) docker-compose-build.yaml docker-compose-prebuilt.yaml"

# ── Step 5: Write systemd service ─────────────────────────────────────────────

echo "[5/7] Writing systemd service..."
remote bash -s <<REMOTE
set -euo pipefail
cat > /etc/systemd/system/node-manager.service <<EOF
[Unit]
Description=Harbor Node Manager
After=network.target docker.service
Requires=docker.service

[Service]
ExecStart=/opt/node_manager_venv/bin/uvicorn node_manager:app --host 0.0.0.0 --port ${SERVICE_PORT} --workers 1
WorkingDirectory=${APP_DIR}
Environment=NODE_MANAGER_API_KEY=${API_KEY}
Environment=DATASET_ROOT=${DATA_ROOT}
Environment=HARBOR_ROOT=/tmp/harbor
Environment=HF_TOKEN=${HF_TOKEN:-}
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
echo "systemd service written"
REMOTE

# ── Step 6: Open firewall port ────────────────────────────────────────────────

echo "[6/8] Opening firewall port ${SERVICE_PORT}..."
remote bash -s <<REMOTE
set -euo pipefail
if command -v ufw &>/dev/null; then
    ufw allow ${SERVICE_PORT}/tcp
    echo "ufw: port ${SERVICE_PORT} opened"
elif command -v firewall-cmd &>/dev/null; then
    firewall-cmd --permanent --add-port=${SERVICE_PORT}/tcp
    firewall-cmd --reload
    echo "firewalld: port ${SERVICE_PORT} opened"
else
    # Fall back to iptables
    iptables -C INPUT -p tcp --dport ${SERVICE_PORT} -j ACCEPT 2>/dev/null || \
        iptables -I INPUT -p tcp --dport ${SERVICE_PORT} -j ACCEPT
    echo "iptables: port ${SERVICE_PORT} opened"
fi
REMOTE

# ── Step 7: Enable and start service ─────────────────────────────────────────

echo "[7/8] Starting node manager service..."
remote bash -s <<'REMOTE'
set -euo pipefail
systemctl enable node-manager
systemctl restart node-manager
sleep 2
systemctl is-active node-manager && echo "Service is running" || {
    echo "ERROR: Service failed to start. Logs:"
    journalctl -u node-manager -n 30 --no-pager
    exit 1
}
REMOTE

# ── Step 8: Health check from local machine ───────────────────────────────────

echo "[8/8] Health check..."
sleep 1
STATUS=$(curl -sf "http://${HOST}:${SERVICE_PORT}/health" 2>&1 || echo "FAILED")
echo "    Response: $STATUS"

if echo "$STATUS" | grep -q '"status":"ok"'; then
    echo ""
    echo "=== Node manager deployed successfully ==="
    echo "    URL:     http://${HOST}:${SERVICE_PORT}"
    echo "    Health:  http://${HOST}:${SERVICE_PORT}/health"
    echo ""
    echo "    To run tests:"
    echo "    export NODE_MANAGER_URL=http://${HOST}:${SERVICE_PORT}"
    echo "    export NODE_MANAGER_API_KEY=${API_KEY}"
    echo "    python seta_env/test/test_node_manager.py"
else
    echo "WARNING: Health check failed — service may still be starting."
    echo "    Check logs: ssh -i $SSH_KEY ${SSH_USER}@${HOST} journalctl -u node-manager -f"
fi
