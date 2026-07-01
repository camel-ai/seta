#!/usr/bin/env bash
# start.sh — One-click startup for the env services stack.
#
# What it does:
#   1. For each node in nodes.yaml with a "deploy:" block, run deploy_env_service.sh
#   2. Start the env_scheduler locally
#   3. If --dataset is given, activate it across all nodes
#
# Usage:
#   bash start.sh                                  # deploy nodes + start scheduler
#   bash start.sh --skip-deploy                    # start scheduler only
#   bash start.sh --dataset seta-env-harbor        # deploy + start + activate dataset
#   bash start.sh --skip-deploy --dataset seta-env-harbor
#   bash start.sh --port 8003 --host 127.0.0.1     # custom scheduler bind
#   bash start.sh --daemon                          # background mode
#   bash start.sh --stop                            # stop daemon

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON="${PYTHON:-$(command -v python3)}"
UVICORN="${UVICORN:-$(command -v uvicorn)}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

SCHEDULER_HOST="127.0.0.1"
SCHEDULER_PORT="8003"
SKIP_DEPLOY=false
DATASET_NAME=""
DAEMON=false
STOP=false
SKIP_DEPS=false

LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$LOG_DIR/scheduler.pid"
LOG_FILE="$LOG_DIR/scheduler.log"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-deploy) SKIP_DEPLOY=true; shift ;;
        --skip-deps)   SKIP_DEPS=true;   shift ;;
        --port)        SCHEDULER_PORT="$2"; shift 2 ;;
        --host)        SCHEDULER_HOST="$2"; shift 2 ;;
        --dataset)     DATASET_NAME="$2"; shift 2 ;;
        --daemon)      DAEMON=true;       shift ;;
        --stop)        STOP=true;         shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Fail fast on missing tokens ──────────────────────────────────────────────

if [[ "$STOP" != "true" && "$SKIP_DEPLOY" != "true" ]]; then
    if [[ -z "${GH_TOKEN:-}" ]]; then
        echo "ERROR: GH_TOKEN env var required for deploying to remote nodes."
        echo "       export GH_TOKEN=ghp_xxx"
        exit 1
    fi
fi
if [[ "$STOP" != "true" && -n "$DATASET_NAME" && -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN env var required for dataset setup."
    echo "       export HF_TOKEN=hf_xxx"
    exit 1
fi

# ── Stop daemon ──────────────────────────────────────────────────────────────

if [[ "$STOP" == "true" ]]; then
    if [[ ! -f "$PID_FILE" ]]; then
        echo "No PID file found — scheduler may not be running."
        exit 0
    fi
    PID="$(cat "$PID_FILE")"
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping scheduler (PID $PID)..."
        kill "$PID" 2>/dev/null
        for i in $(seq 1 10); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
        if kill -0 "$PID" 2>/dev/null; then kill -9 "$PID" 2>/dev/null; fi
        echo "Scheduler stopped."
    else
        echo "No process for PID $PID — already stopped."
    fi
    rm -f "$PID_FILE"
    exit 0
fi

# ── Daemon re-exec ───────────────────────────────────────────────────────────

if [[ "$DAEMON" == "true" ]]; then
    mkdir -p "$LOG_DIR"
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "Scheduler already running (PID $(cat "$PID_FILE")). Use --stop first."
        exit 1
    fi
    ARGS=()
    [[ "$SKIP_DEPLOY" == "true" ]] && ARGS+=(--skip-deploy)
    [[ "$SKIP_DEPS" == "true" ]] && ARGS+=(--skip-deps)
    [[ -n "$DATASET_NAME" ]] && ARGS+=(--dataset "$DATASET_NAME")
    ARGS+=(--host "$SCHEDULER_HOST" --port "$SCHEDULER_PORT")
    echo "Starting scheduler in daemon mode. Log: $LOG_FILE"
    setsid env PATH="$PATH" PYTHONPATH="$PYTHONPATH" GH_TOKEN="${GH_TOKEN:-}" HF_TOKEN="${HF_TOKEN:-}" bash "$0" "${ARGS[@]}" >> "$LOG_FILE" 2>&1 &
    echo "$!" > "$PID_FILE"
    echo "Scheduler started (PID $!). PID file: $PID_FILE"
    exit 0
fi

mkdir -p "$LOG_DIR"
NODES_YAML="$SCRIPT_DIR/nodes.yaml"

# ── Step 1: Deploy env_service to nodes ──────────────────────────────────────

if [[ "$SKIP_DEPLOY" == "false" ]]; then
    echo "=== Deploying env_service to nodes ==="

    DEPLOY_NODES=$($PYTHON - "$NODES_YAML" <<'PY'
import sys, yaml
data = yaml.safe_load(open(sys.argv[1]))
for n in data.get("nodes", []):
    d = n.get("deploy")
    if not d:
        continue
    url = n["url"]
    host = d.get("host") or url.split("//")[1].split(":")[0]
    port = url.split(":")[-1] if ":" in url.split("//")[1] else "8002"
    print("|".join([
        host,
        d.get("ssh_key", ""),
        d.get("ssh_user", "root"),
        port,
        d.get("api_key", ""),
        d.get("data_root", "/data/harbor/dataset"),
        str(n.get("slots", 16)),
    ]))
PY
    )

    if [[ -z "$DEPLOY_NODES" ]]; then
        echo "  No nodes with 'deploy:' block — skipping."
    else
        declare -a DEPLOY_PIDS=()
        while IFS="|" read -r HOST SSH_KEY SSH_USER PORT API_KEY DATA_ROOT SLOTS; do
            echo "--- Deploying $HOST (background) ---"
            DEPLOY_ARGS=("$HOST" "$SSH_KEY"
                "--user"      "$SSH_USER"
                "--port"      "$PORT"
                "--slots"     "$SLOTS"
                "--data-root" "$DATA_ROOT"
            )
            [[ -n "$API_KEY" ]] && DEPLOY_ARGS+=("--api-key" "$API_KEY")
            [[ "$SKIP_DEPS" == "true" ]] && DEPLOY_ARGS+=("--skip-deps")

            bash "$SCRIPT_DIR/deploy_env_service.sh" "${DEPLOY_ARGS[@]}" < /dev/null &
            DEPLOY_PIDS+=($!)
        done <<< "$DEPLOY_NODES"

        DEPLOY_FAILED=0
        for pid in "${DEPLOY_PIDS[@]}"; do
            wait "$pid" || DEPLOY_FAILED=1
        done
        if [[ "$DEPLOY_FAILED" -ne 0 ]]; then
            echo "ERROR: One or more deployments failed."
            exit 1
        fi
    fi
    echo ""
fi

# ── Step 2: Start scheduler ─────────────────────────────────────────────────

EXISTING_PID=$(ss -tlnp 2>/dev/null | grep ":${SCHEDULER_PORT} " | grep -oP 'pid=\K[0-9]+' | head -1 || true)
if [[ -n "$EXISTING_PID" ]]; then
    echo "Port ${SCHEDULER_PORT} in use by PID $EXISTING_PID — killing..."
    kill "$EXISTING_PID" 2>/dev/null
    sleep 2
fi

echo "=== Starting env_scheduler on ${SCHEDULER_HOST}:${SCHEDULER_PORT} ==="
echo "    nodes.yaml: $NODES_YAML"

export NODES_YAML

if [[ -z "$DATASET_NAME" ]]; then
    exec "$UVICORN" seta_env.services.env_scheduler:app \
        --host "$SCHEDULER_HOST" \
        --port "$SCHEDULER_PORT"
fi

# Dataset requested — start scheduler in background, wait for ready, then setup.
"$UVICORN" seta_env.services.env_scheduler:app \
    --host "$SCHEDULER_HOST" \
    --port "$SCHEDULER_PORT" &
SCHEDULER_PID=$!
trap 'kill $SCHEDULER_PID 2>/dev/null; wait $SCHEDULER_PID 2>/dev/null; exit 0' INT TERM

# ── Step 3: Wait for scheduler ───────────────────────────────────────────────

SCHEDULER_URL="http://${SCHEDULER_HOST}:${SCHEDULER_PORT}"
echo "=== Waiting for scheduler... ==="
for i in $(seq 1 30); do
    if curl -sf "${SCHEDULER_URL}/health" &>/dev/null; then
        echo "    Scheduler is up."
        break
    fi
    if ! kill -0 $SCHEDULER_PID 2>/dev/null; then
        echo "ERROR: Scheduler died."
        exit 1
    fi
    sleep 1
done

# ── Step 4: Setup dataset ────────────────────────────────────────────────────

echo ""
echo "=== Setting up dataset: $DATASET_NAME ==="
HF_TOKEN="${HF_TOKEN:-}"
RESP=$(curl -sf -X POST "${SCHEDULER_URL}/setup_dataset" \
    -H "Content-Type: application/json" \
    -d "{\"dataset_name\": \"${DATASET_NAME}\", \"hf_token\": \"${HF_TOKEN}\"}" \
    2>&1 || echo "FAILED")
echo "    Response: $RESP"

echo ""
echo "=== All done. Scheduler running (PID $SCHEDULER_PID). Press Ctrl-C to stop. ==="
wait $SCHEDULER_PID
