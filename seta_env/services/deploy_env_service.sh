#!/usr/bin/env bash
# deploy_env_service.sh — Deploy env_service to a remote server.
#
# Git clones the repo at the current commit, pip installs camel + harbor + seta_env,
# starts the systemd service. Requires GH_TOKEN env var.
#
# Usage:
#   GH_TOKEN=ghp_xxx HF_TOKEN=hf_xxx ./deploy_env_service.sh <host> <ssh_key> [options]
#
# Options:
#   --user        SSH user              (default: root)
#   --port        Service port          (default: 8002)
#   --slots       MAX_SLOTS             (default: 16)
#   --api-key     ENV_SERVICE_API_KEY   (default: env-service-dev-key)
#   --skip-deps   Skip apt + pip install (only git pull + restart)
#   --dataset     Dataset to activate after deploy

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: GH_TOKEN=ghp_xxx HF_TOKEN=hf_xxx $0 <host> <ssh_key> [options]"
    exit 1
fi

HOST="$1"; shift
SSH_KEY="$1"; shift

SSH_USER="root"
SERVICE_PORT="8002"
MAX_SLOTS="16"
API_KEY="${ENV_SERVICE_API_KEY:-env-service-dev-key}"
SKIP_DEPS=false
DATASET_NAME=""
DATA_ROOT=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMMIT="$(cd "$REPO_ROOT" && git rev-parse HEAD)"

if [[ -z "${GH_TOKEN:-}" ]]; then
    echo "ERROR: GH_TOKEN env var required."; exit 1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --user)      SSH_USER="$2";     shift 2 ;;
        --port)      SERVICE_PORT="$2"; shift 2 ;;
        --slots)     MAX_SLOTS="$2";    shift 2 ;;
        --api-key)   API_KEY="$2";      shift 2 ;;
        --skip-deps) SKIP_DEPS=true;    shift ;;
        --dataset)   DATASET_NAME="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2";    shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no -o BatchMode=yes"
APP_DIR="/opt/env_service"
REPO_URL="https://${GH_TOKEN}@github.com/camel-ai/terminal_agent.git"

echo "=== Deploying env_service to ${SSH_USER}@${HOST} ==="
echo "    Commit: ${COMMIT:0:8}  Port: $SERVICE_PORT  Slots: $MAX_SLOTS"

remote() { ssh $SSH_OPTS "${SSH_USER}@${HOST}" "$@"; }

# ── Step 1: System packages + Python 3.12 + Docker ───────────────────────────

if [[ "$SKIP_DEPS" == "false" ]]; then
    echo "[1/4] System packages..."
    remote bash -s <<'REMOTE'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq software-properties-common git git-lfs curl build-essential 2>&1 | tail -3
git lfs install 2>/dev/null
if ! python3.12 --version &>/dev/null; then
    add-apt-repository -y ppa:deadsnakes/ppa 2>&1 | tail -2
    apt-get update -qq
    apt-get install -y -qq python3.12 python3.12-venv python3.12-dev 2>&1 | tail -3
fi

# Docker daemon config (same as setup.sh)
DOCKER_DAEMON_CONFIG='/etc/docker/daemon.json'
if [ -f "$DOCKER_DAEMON_CONFIG" ]; then
    cp "$DOCKER_DAEMON_CONFIG" "${DOCKER_DAEMON_CONFIG}.backup" 2>/dev/null || true
fi
cat > "$DOCKER_DAEMON_CONFIG" <<DEOF
{
  "default-address-pools": [
    {
      "base": "10.200.0.0/16",
      "size": 28
    }
  ]
}
DEOF
systemctl restart docker 2>/dev/null || true
echo "done"
REMOTE
fi

# ── Step 2: Git clone / fetch + checkout commit ──────────────────────────────

echo "[2/4] Git checkout ${COMMIT:0:8}..."
remote bash -s <<REMOTE
set -euo pipefail
if [ -d "${APP_DIR}/terminal_agent/.git" ]; then
    cd "${APP_DIR}/terminal_agent"
    git fetch origin
    git checkout ${COMMIT}
else
    rm -rf "${APP_DIR}/terminal_agent"
    mkdir -p "${APP_DIR}"
    git clone "${REPO_URL}" "${APP_DIR}/terminal_agent"
    cd "${APP_DIR}/terminal_agent"
    git checkout ${COMMIT}
fi
git submodule update --init external/camel external/harbor
echo "at \$(git rev-parse --short HEAD)"
REMOTE

# ── Step 3: Pip install ──────────────────────────────────────────────────────

if [[ "$SKIP_DEPS" == "false" ]]; then
    echo "[3/4] Pip install..."
    remote bash -s <<REMOTE
set -euo pipefail
VENV="${APP_DIR}/venv"
if [ -d "\$VENV" ]; then
    PY_VER=\$("\$VENV/bin/python3" --version 2>/dev/null | grep -oP '3\.\K[0-9]+' || echo "0")
    [ "\$PY_VER" -lt 12 ] && rm -rf "\$VENV"
fi
[ ! -d "\$VENV" ] && python3.12 -m venv "\$VENV"
source "\$VENV/bin/activate"
pip install -q --upgrade pip
cd "${APP_DIR}/terminal_agent"
pip install -q -e external/camel 2>&1 | tail -2
pip install -q -e external/harbor 2>&1 | tail -2
pip install -q -e . 2>&1 | tail -2
pip install -q fastapi "uvicorn[standard]" httpx aiofiles PyYAML docker
echo "pip done"
REMOTE
fi

# ── Step 4: Systemd + start ──────────────────────────────────────────────────

EFFECTIVE_DATA_ROOT="${DATA_ROOT:-${APP_DIR}/data/dataset}"
echo "[4/4] Start service..."
remote bash -s <<REMOTE
mkdir -p "${EFFECTIVE_DATA_ROOT}" "${APP_DIR}/data/trials"
cat > /etc/systemd/system/env-service.service <<EOF
[Unit]
Description=Env Service
After=network.target docker.service
Requires=docker.service
[Service]
ExecStart=${APP_DIR}/venv/bin/uvicorn seta_env.services.env_service:app --host 0.0.0.0 --port ${SERVICE_PORT} --workers 1
WorkingDirectory=${APP_DIR}/terminal_agent
Environment=PYTHONPATH=${APP_DIR}/terminal_agent
Environment=ENV_SERVICE_API_KEY=${API_KEY}
Environment=DATASET_ROOT=${EFFECTIVE_DATA_ROOT}
Environment=HARBOR_ROOT=${APP_DIR}/data
Environment=MAX_SLOTS=${MAX_SLOTS}
Environment=HF_TOKEN=${HF_TOKEN:-}
LimitNOFILE=1048576
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
command -v ufw &>/dev/null && ufw status | grep -q "active" && ufw allow ${SERVICE_PORT}/tcp 2>/dev/null || true
systemctl enable env-service
systemctl restart env-service
REMOTE

# Health check
for i in $(seq 1 15); do
    STATUS=$(curl -sf "http://${HOST}:${SERVICE_PORT}/health" 2>&1 || echo "")
    [[ -n "$STATUS" ]] && break; sleep 2
done
if echo "${STATUS:-}" | grep -q '"status":"ok"'; then
    echo "    OK: http://${HOST}:${SERVICE_PORT}"
else
    echo "    WARN: health check failed"
fi

# Optional dataset
if [[ -n "$DATASET_NAME" ]]; then
    echo "Setting up dataset: $DATASET_NAME ..."
    curl -sf --max-time 600 -X POST "http://${HOST}:${SERVICE_PORT}/setup" \
        -H "Content-Type: application/json" -H "X-API-Key: ${API_KEY}" \
        -d "{\"dataset_name\": \"${DATASET_NAME}\", \"hf_token\": \"${HF_TOKEN:-}\"}" 2>&1
    echo
fi
