#!/usr/bin/env bash
# Start the seta_env env_service in a long-running tmux session.
#
# Standalone env_service launcher but:
#   - bypasses the missing logging_config.json reference
#   - explicitly uses the CAMEL stack at /root/data/terminal_agent/venv_cpu
#   - runs in a named tmux session (default: train1)
#
# Usage:
#   ./env_service.sh start       # creates tmux session "$TMUX_SESSION"
#   ./env_service.sh wait        # blocks until /health responds
#   ./env_service.sh attach
#   ./env_service.sh logs
#   ./env_service.sh status
#   ./env_service.sh stop

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/terminal_agent}"
VENV_PY="${VENV_PY:-$REPO_ROOT/venv_cpu/bin/python}"

# Pull DAYTONA_API_KEY / DAYTONA_API_URL / HF_TOKEN / WANDB_API_KEY etc.
# from ~/.bashrc into the launcher's env so they propagate into the tmux
# session. Use `bash -c` in interactive mode to bypass the standard
# `[ -z "$PS1" ] && return` guard, then export the wanted vars back into
# our shell. We disable `set -e` around it so any harmless rc errors
# don't kill the launcher.
if [ -f "$HOME/.bashrc" ]; then
    set +e
    eval "$(bash -ic '
        for v in DAYTONA_API_KEY DAYTONA_API_URL HF_HOME HF_TOKEN WANDB_API_KEY; do
            val="${!v-}"
            if [ -n "$val" ]; then
                printf "export %s=%q\n" "$v" "$val"
            fi
        done
    ' 2>/dev/null)"
    set -e
fi
: "${DAYTONA_API_KEY:?DAYTONA_API_KEY is not set; check $HOME/.bashrc}"
: "${DAYTONA_API_URL:?DAYTONA_API_URL is not set; check $HOME/.bashrc}"

TMUX_SESSION="${TMUX_SESSION:-train1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"
MAX_SLOTS="${MAX_SLOTS:-128}"             # high-concurrency eval; verify Daytona quota tolerance
# HARBOR_DAYTONA_MAX_CREATES caps concurrent daytona.create() calls inside
# harbor's daytona.py. Tier 3 Daytona allows 500 creations/min so the 128
# traj/step burst doesn't hit the rate limit — leave gating OFF (0) by default.
# Set to a positive int (e.g. 8) if running on lower tiers and seeing 429s.
HARBOR_DAYTONA_MAX_CREATES="${HARBOR_DAYTONA_MAX_CREATES:-0}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/dataset}"
SETUP_DATASET="${SETUP_DATASET:-}"        # don't auto-download; we have seta-env-final already
CONFIG="${CONFIG:-$REPO_ROOT/scripts/miles/core/configs/seta_env_config.yaml}"
LOG_LEVEL="${LOG_LEVEL:-info}"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/env_service}"
LOG_FILE="$LOG_DIR/env_service.log"
mkdir -p "$LOG_DIR"

is_running() { tmux has-session -t "$TMUX_SESSION" 2>/dev/null; }
is_healthy() { curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; }

start() {
    if [ ! -x "$VENV_PY" ]; then
        echo "ERROR: venv python missing: $VENV_PY" >&2
        exit 1
    fi
    if [ ! -f "$CONFIG" ]; then
        echo "ERROR: config missing: $CONFIG" >&2
        exit 1
    fi
    if is_running; then
        echo "tmux session '$TMUX_SESSION' already exists. '$0 status' or '$0 stop' first." >&2
        exit 1
    fi

    echo "[start] config=$CONFIG"
    echo "[start] dataset_root=$DATASET_ROOT  setup_dataset='${SETUP_DATASET:-(none)}'"
    echo "[start] max_slots=$MAX_SLOTS  port=$PORT  python=$VENV_PY"
    echo "[start] HARBOR_DAYTONA_MAX_CREATES=$HARBOR_DAYTONA_MAX_CREATES"
    echo "[start] log=$LOG_FILE  tmux=$TMUX_SESSION"

    tmux new-session -d -s "$TMUX_SESSION" -x 220 -y 50 \
        "cd '$REPO_ROOT' && \
         echo '=== env_service starting at \$(date) ===' && \
         ENV_SERVICE_CONFIG='$CONFIG' \
         DATASET_ROOT='$DATASET_ROOT' \
         SETUP_DATASET='$SETUP_DATASET' \
         MAX_SLOTS='$MAX_SLOTS' \
         HARBOR_DAYTONA_MAX_CREATES='$HARBOR_DAYTONA_MAX_CREATES' \
         DAYTONA_SNAPSHOT_EVICT_AGE_HOURS='${DAYTONA_SNAPSHOT_EVICT_AGE_HOURS:-0}' \
         DAYTONA_DECLARATIVE='${DAYTONA_DECLARATIVE:-0}' \
         DAYTONA_CANARY_PLACEMENT='${DAYTONA_CANARY_PLACEMENT:-1}' \
         CANARY_PLACEMENT_TIMEOUT='${CANARY_PLACEMENT_TIMEOUT:-420}' \
         BUILD_TIMEOUT_SECONDS='${BUILD_TIMEOUT_SECONDS:-600}' \
         STEP_TIMEOUT_SECONDS='${STEP_TIMEOUT_SECONDS:-1800}' \
         STEP_USE_SUBPROCESS='${STEP_USE_SUBPROCESS:-0}' \
         BUILD_CONCURRENCY='${BUILD_CONCURRENCY:-8}' \
         INFLIGHT_LEAD='${INFLIGHT_LEAD:-48}' \
         CREATE_CAPACITY_MAX_ATTEMPTS='${CREATE_CAPACITY_MAX_ATTEMPTS:-75}' \
         HARBOR_ROOT='${HARBOR_ROOT:-/data/terminal_agent/harbor}' \
         MODEL_TIMEOUT='${MODEL_TIMEOUT:-180}' \
         DAYTONA_API_KEY='$DAYTONA_API_KEY' \
         DAYTONA_API_URL='$DAYTONA_API_URL' \
         HF_HOME='${HF_HOME:-/data/HF_HOME}' \
         HF_TOKEN='${HF_TOKEN:-}' \
         '$VENV_PY' -m uvicorn seta_env.services.env_service:app \
            --host '$HOST' --port '$PORT' \
            --log-level '$LOG_LEVEL' 2>&1; \
         echo '=== env_service exited at \$(date), code=\$? ===' && exec bash"
    tmux pipe-pane -t "$TMUX_SESSION" "cat >> '$LOG_FILE'"

    echo "[start] tmux session created. Use:"
    echo "  $0 wait   # block until /health responds"
    echo "  $0 logs   # tail the log"
    echo "  $0 attach # interactive (Ctrl-b d to detach)"
}

wait_ready() {
    local max_wait=180
    echo "[wait] polling /health on http://127.0.0.1:$PORT (up to ${max_wait}s) …"
    for i in $(seq 1 "$max_wait"); do
        if is_healthy; then
            echo "[wait] ready (took ${i}s). Endpoint: http://127.0.0.1:$PORT"
            return 0
        fi
        if ! is_running; then
            echo "[wait] tmux session died. tail of log:" >&2
            tail -40 "$LOG_FILE" >&2
            return 1
        fi
        sleep 1
    done
    echo "[wait] timeout after ${max_wait}s — last log:" >&2
    tail -20 "$LOG_FILE" >&2
    return 1
}

attach() { is_running || { echo "no session"; exit 1; }; tmux attach -t "$TMUX_SESSION"; }
logs()   { tail -f "$LOG_FILE"; }
status() {
    is_running && echo "tmux '$TMUX_SESSION': UP" || echo "tmux '$TMUX_SESSION': down"
    is_healthy && echo "endpoint  http://127.0.0.1:$PORT: HEALTHY" || echo "endpoint  http://127.0.0.1:$PORT: not responding"
    [ -f "$LOG_FILE" ] && echo "log tail:" && tail -5 "$LOG_FILE"
}
stop() {
    is_running && tmux kill-session -t "$TMUX_SESSION" || true
    pkill -f "uvicorn .*seta_env.services.env_service" 2>/dev/null || true
    sleep 1
    echo "[stop] done"
}

case "${1:-help}" in
    start)  start ;;
    wait)   wait_ready ;;
    attach) attach ;;
    logs)   logs ;;
    status) status ;;
    stop)   stop ;;
    *) sed -n '1,18p' "$0"; exit 0 ;;
esac
