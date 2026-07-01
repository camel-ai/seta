#!/usr/bin/env bash
# start.sh — One-click startup for the slot-pool stack.
#
# What it does:
#   1. For each node in nodes.yaml that has a "deploy:" block,
#      run deploy_node.sh to install/restart the node manager on that host.
#   2. Start the scheduler service locally (blocks until Ctrl-C).
#   3. If --dataset is given, activate that dataset across all nodes once
#      the scheduler is ready.
#
# Usage:
#   bash start.sh                                  # deploy nodes + start scheduler
#   bash start.sh --skip-deploy                    # start scheduler only
#   bash start.sh --dataset seta-env-harbor        # deploy + start + activate dataset
#   bash start.sh --skip-deploy --dataset seta-env-harbor  # start + activate dataset
#   bash start.sh --port 9000 --host 0.0.0.0       # custom scheduler bind address
#   bash start.sh --daemon [...]                   # run as background daemon (logs → logs/)
#   bash start.sh --stop                           # stop a running daemon
#
# nodes.yaml controls which nodes get deployed — add a "deploy:" block to
# any node entry to enable auto-deploy for that node (see nodes.yaml).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PYTHON="${PYTHON:-$(command -v python3)}"
UVICORN="${UVICORN:-$(command -v uvicorn)}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

SCHEDULER_HOST="127.0.0.1"
SCHEDULER_PORT="8000"
SKIP_DEPLOY=false
DATASET_NAME=""
DAEMON=false
STOP=false

LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$LOG_DIR/scheduler.pid"
LOG_FILE="$LOG_DIR/scheduler.log"

# ── Parse args ────────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-deploy)  SKIP_DEPLOY=true ; shift ;;
        --port)         SCHEDULER_PORT="$2" ; shift 2 ;;
        --host)         SCHEDULER_HOST="$2" ; shift 2 ;;
        --dataset)      DATASET_NAME="$2" ; shift 2 ;;
        --daemon)       DAEMON=true ; shift ;;
        --stop)         STOP=true ; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Stop daemon ───────────────────────────────────────────────────────────────

if [[ "$STOP" == "true" ]]; then
    if [[ ! -f "$PID_FILE" ]]; then
        echo "No PID file found at $PID_FILE — scheduler may not be running."
        exit 0
    fi
    PID="$(cat "$PID_FILE")"
    # Kill the entire process group (PID == PGID because we used setsid).
    # This ensures uvicorn and all children are killed, not just the bash wrapper.
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping scheduler process group (PGID $PID)..."
        kill -- -"$PID"
        # Wait up to 10s for the group to exit
        for i in $(seq 1 10); do
            kill -0 "$PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$PID" 2>/dev/null; then
            echo "Scheduler did not stop gracefully — sending SIGKILL to process group."
            kill -9 -- -"$PID"
        fi
        echo "Scheduler stopped."
    else
        echo "No process found for PID $PID — already stopped."
    fi
    rm -f "$PID_FILE"
    exit 0
fi

# ── Daemon re-exec ────────────────────────────────────────────────────────────

if [[ "$DAEMON" == "true" ]]; then
    mkdir -p "$LOG_DIR"
    if [[ -f "$PID_FILE" ]]; then
        EXISTING_PID="$(cat "$PID_FILE")"
        if kill -0 "$EXISTING_PID" 2>/dev/null; then
            echo "Scheduler already running (PID $EXISTING_PID). Use --stop to stop it."
            exit 1
        fi
        rm -f "$PID_FILE"
    fi
    # Also guard against a stale PID file where the process died but port is still bound
    if ss -tlnp 2>/dev/null | grep -q ":${SCHEDULER_PORT} "; then
        echo "ERROR: Port ${SCHEDULER_PORT} is already in use but no matching PID file found."
        echo "       Run: kill \$(ss -tlnp | grep :${SCHEDULER_PORT} | awk '{print \$6}' | grep -oP 'pid=\K[0-9]+')"
        exit 1
    fi
    # Re-invoke this script without --daemon in a new process group via setsid,
    # so killing the group PID on --stop takes down uvicorn and all children.
    ARGS=()
    [[ "$SKIP_DEPLOY" == "true" ]] && ARGS+=(--skip-deploy)
    [[ -n "$DATASET_NAME" ]] && ARGS+=(--dataset "$DATASET_NAME")
    ARGS+=(--host "$SCHEDULER_HOST" --port "$SCHEDULER_PORT")
    echo "Starting scheduler in daemon mode. Log: $LOG_FILE"
    setsid env PATH="$PATH" PYTHONPATH="$PYTHONPATH" bash "$0" "${ARGS[@]}" >> "$LOG_FILE" 2>&1 &
    DAEMON_PID=$!
    echo "$DAEMON_PID" > "$PID_FILE"
    echo "Scheduler started (PID $DAEMON_PID). PID file: $PID_FILE"
    exit 0
fi

mkdir -p "$LOG_DIR"

NODES_YAML="$SCRIPT_DIR/nodes.yaml"

# ── Step 1: Deploy node managers ──────────────────────────────────────────────

if [[ "$SKIP_DEPLOY" == "false" ]]; then
    echo "=== Deploying node managers ==="

    # Parse nodes with a deploy block using Python (avoids a yq dependency).
    DEPLOY_NODES=$($PYTHON - "$NODES_YAML" <<'PY'
import sys, yaml
data = yaml.safe_load(open(sys.argv[1]))
for n in data.get("nodes", []):
    d = n.get("deploy")
    if not d:
        continue
    # Extract host from url if not explicitly set in deploy block.
    url = n["url"]  # e.g. http://1.2.3.4:8001
    host = d.get("host") or url.split("//")[1].split(":")[0]
    print("|".join([
        host,
        d.get("ssh_key", ""),
        d.get("ssh_user", "root"),
        str(d.get("port", 8001)),
        d.get("api_key", ""),
        d.get("data_root", "/data/harbor/dataset"),
        d.get("app_dir", "/opt/node_manager"),
    ]))
PY
    )

    if [[ -z "$DEPLOY_NODES" ]]; then
        echo "  No nodes with 'deploy:' block found in nodes.yaml — skipping."
    else
        declare -a DEPLOY_PIDS=()
        while IFS="|" read -r HOST SSH_KEY SSH_USER PORT API_KEY DATA_ROOT APP_DIR; do
            echo "--- Deploying $HOST (background) ---"
            DEPLOY_ARGS=("$HOST" "$SSH_KEY"
                "--user"      "$SSH_USER"
                "--port"      "$PORT"
                "--data-root" "$DATA_ROOT"
                "--app-dir"   "$APP_DIR"
            )
            [[ -n "$API_KEY" ]] && DEPLOY_ARGS+=("--api-key" "$API_KEY")

            bash "$SCRIPT_DIR/deploy_node.sh" "${DEPLOY_ARGS[@]}" < /dev/null &
            DEPLOY_PIDS+=($!)
        done <<< "$DEPLOY_NODES"

        DEPLOY_FAILED=0
        for pid in "${DEPLOY_PIDS[@]}"; do
            wait "$pid" || DEPLOY_FAILED=1
        done
        if [[ "$DEPLOY_FAILED" -ne 0 ]]; then
            echo "ERROR: One or more node deployments failed — aborting."
            exit 1
        fi
    fi
    echo ""
fi

# ── Step 2: Start scheduler ───────────────────────────────────────────────────

# Kill any existing process on the target port before starting.
EXISTING_PID=$(ss -tlnp 2>/dev/null | grep ":${SCHEDULER_PORT} " | grep -oP 'pid=\K[0-9]+' | head -1)
if [[ -n "$EXISTING_PID" ]]; then
    echo "Port ${SCHEDULER_PORT} is in use by PID $EXISTING_PID — killing it..."
    kill "$EXISTING_PID" 2>/dev/null
    for i in $(seq 1 5); do
        kill -0 "$EXISTING_PID" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "Process $EXISTING_PID did not stop — sending SIGKILL."
        kill -9 "$EXISTING_PID" 2>/dev/null
        sleep 1
    fi
    echo "Port ${SCHEDULER_PORT} freed."
fi

echo "=== Starting scheduler on ${SCHEDULER_HOST}:${SCHEDULER_PORT} ==="
echo "    nodes.yaml: $NODES_YAML"
echo "    Press Ctrl-C to stop."
echo ""

if [[ -z "$DATASET_NAME" ]]; then
    # No dataset — simple blocking exec, no background process needed.
    exec "$UVICORN" seta_env.runtimes.slot_pool_service.scheduler_service:app \
        --host "$SCHEDULER_HOST" \
        --port "$SCHEDULER_PORT"
fi

# Dataset requested — start scheduler in background, wait for ready, then setup.
"$UVICORN" seta_env.runtimes.slot_pool_service.scheduler_service:app \
    --host "$SCHEDULER_HOST" \
    --port "$SCHEDULER_PORT" &
SCHEDULER_PID=$!

# Forward Ctrl-C to the scheduler process.
trap 'kill $SCHEDULER_PID 2>/dev/null; wait $SCHEDULER_PID 2>/dev/null; exit 0' INT TERM

# ── Step 3: Wait for scheduler to be ready ────────────────────────────────────

SCHEDULER_URL="http://${SCHEDULER_HOST}:${SCHEDULER_PORT}"
echo "=== Waiting for scheduler to be ready... ==="
for i in $(seq 1 30); do
    if curl -sf "${SCHEDULER_URL}/health" &>/dev/null; then
        echo "    Scheduler is up."
        break
    fi
    if ! kill -0 $SCHEDULER_PID 2>/dev/null; then
        echo "ERROR: Scheduler process died unexpectedly."
        exit 1
    fi
    sleep 1
done

if ! curl -sf "${SCHEDULER_URL}/health" &>/dev/null; then
    echo "ERROR: Scheduler did not become ready in time."
    kill $SCHEDULER_PID 2>/dev/null
    exit 1
fi

# ── Step 4: Activate dataset across all nodes ─────────────────────────────────

echo ""
bash "$SCRIPT_DIR/setup_dataset.sh" "$DATASET_NAME" --scheduler "$SCHEDULER_URL"

echo ""
echo "=== All done. Scheduler running (PID $SCHEDULER_PID). Press Ctrl-C to stop. ==="
wait $SCHEDULER_PID
