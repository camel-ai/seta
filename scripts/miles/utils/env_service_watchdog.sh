#!/usr/bin/env bash
# env_service watchdog (P1) — auto-restart env_service on the zombie-thread leak
# or hang, so the rollout stays alive across a week-long run.
#
# Restarts env_service when ANY of:
#   1. /health unreachable (curl timeout) for K consecutive checks  -> hard hang
#   2. uvicorn process threads > THREAD_MAX                         -> zombie leak
#   3. available_slots==0 AND active_steps==0 for STUCK_CYCLES      -> leaked slots
#      (slots all held by dead steps; nothing will ever free them)
#
# Restart = env_service.sh stop+start with the run's config; env_service.sh
# sources DAYTONA creds from ~/.bashrc itself (creds never written here).
# Exponential-ish backoff prevents thrash.
#
# Usage: RUN_ROOT=<run dir> bash env_service_watchdog.sh
set -uo pipefail

REPO=/data/terminal_agent
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT to the run dir}"
HEALTH="http://127.0.0.1:8002/health"
CONFIG="${ENV_SERVICE_CONFIG:-$REPO/scripts/miles/seta_env_config_milesrouter_r3.yaml}"

INTERVAL="${INTERVAL:-30}"
# Threads scale with ACTIVE steps (~10 sub-threads/step), so a busy service at 160
# active legitimately has ~1600 threads. The LEAK signal is high threads while FEW
# steps are active (zombies). So the thread trigger requires BOTH threads>THREAD_MAX
# AND active<ACTIVE_LOW. The primary signal is the leaked-slot state below.
THREAD_MAX="${THREAD_MAX:-2200}"       # zombie ceiling; only with low active
ACTIVE_LOW="${ACTIVE_LOW:-25}"         # "few active" — distinguishes zombies from busy
K="${K:-3}"                            # consecutive /health failures -> hard hang
STUCK_CYCLES="${STUCK_CYCLES:-12}"     # 12*30s = 6min leaked-slot -> restart.
# MUST exceed the admission-backlog drain (MAX_SLOTS*ADMISSION_INTERVAL ~= 160*0.8 = 128s),
# during which available:0+active:0 is NORMAL (backlog draining, not a leak).
MIN_RESTART_GAP="${MIN_RESTART_GAP:-240}"   # don't restart more often than every 4min

fails=0; stuck=0; last_restart=0; restarts=0

log() { echo "[watchdog $(date -u +%H:%M:%S)] $*"; }

do_restart() {
  local reason="$1" now; now=$(date +%s)
  if [ $((now - last_restart)) -lt "$MIN_RESTART_GAP" ]; then
    log "WANT restart ($reason) but within backoff ${MIN_RESTART_GAP}s — skipping"; return
  fi
  restarts=$((restarts+1)); last_restart=$now
  log "RESTART #$restarts reason=[$reason]"
  MAX_SLOTS="${MAX_SLOTS:-160}" TMUX_SESSION=train1 \
    LOG_DIR="$RUN_ROOT/env_service" HARBOR_ROOT="$RUN_ROOT" \
    HARBOR_DAYTONA_MAX_CREATES="${HARBOR_DAYTONA_MAX_CREATES:-16}" \
    STEP_TIMEOUT_SECONDS="${STEP_TIMEOUT_SECONDS:-3000}" STEP_USE_SUBPROCESS="${STEP_USE_SUBPROCESS:-1}" DAYTONA_DECLARATIVE=1 \
    DAYTONA_SNAPSHOT_EVICT_AGE_HOURS=3 CONFIG="$CONFIG" \
    bash "$REPO/scripts/miles/env_service.sh" stop 2>&1 | tail -1
  sleep 2
  MAX_SLOTS="${MAX_SLOTS:-160}" TMUX_SESSION=train1 \
    LOG_DIR="$RUN_ROOT/env_service" HARBOR_ROOT="$RUN_ROOT" \
    HARBOR_DAYTONA_MAX_CREATES="${HARBOR_DAYTONA_MAX_CREATES:-16}" \
    STEP_TIMEOUT_SECONDS="${STEP_TIMEOUT_SECONDS:-3000}" STEP_USE_SUBPROCESS="${STEP_USE_SUBPROCESS:-1}" DAYTONA_DECLARATIVE=1 \
    BUILD_CONCURRENCY="${BUILD_CONCURRENCY:-8}" INFLIGHT_LEAD="${INFLIGHT_LEAD:-48}" \
    DAYTONA_SNAPSHOT_EVICT_AGE_HOURS=3 CONFIG="$CONFIG" \
    bash "$REPO/scripts/miles/env_service.sh" start 2>&1 | tail -1
  sleep 45   # let it reconcile + start serving
  fails=0; stuck=0
  log "RESTART #$restarts done"
}

log "watchdog start: RUN_ROOT=$RUN_ROOT THREAD_MAX=$THREAD_MAX interval=${INTERVAL}s"
while true; do
  body=$(curl -fsS --max-time 10 "$HEALTH" 2>/dev/null || true)
  if [ -z "$body" ]; then
    fails=$((fails+1)); stuck=0
    log "health unreachable ($fails/$K)"
    [ "$fails" -ge "$K" ] && do_restart "health-unreachable x$fails"
  else
    fails=0
    threads=$(echo "$body" | grep -oE '"uvicorn_threads":[0-9-]+' | grep -oE '[0-9-]+' | head -1)
    avail=$(echo "$body" | grep -oE '"available_slots":[0-9]+' | grep -oE '[0-9]+' | head -1)
    active=$(echo "$body" | grep -oE '"active_steps":[0-9]+' | grep -oE '[0-9]+' | head -1)
    if [ "${threads:-0}" -gt "$THREAD_MAX" ] && [ "${active:-999}" -lt "$ACTIVE_LOW" ]; then
      log "THREAD LEAK threads=$threads active=$active (zombies)"; do_restart "thread-leak=$threads active=$active"; stuck=0
    elif [ "${avail:-1}" -eq 0 ] && [ "${active:-1}" -ne 0 ]; then
      stuck=0   # genuinely busy (saturated but working)
    elif [ "${avail:-1}" -eq 0 ] && [ "${active:-1}" -eq 0 ]; then
      stuck=$((stuck+1)); log "leaked-slot state avail=0 active=0 ($stuck/$STUCK_CYCLES) threads=$threads"
      [ "$stuck" -ge "$STUCK_CYCLES" ] && { do_restart "leaked-slots x$stuck threads=$threads"; stuck=0; }
    else
      stuck=0
    fi
  fi
  sleep "$INTERVAL"
done
