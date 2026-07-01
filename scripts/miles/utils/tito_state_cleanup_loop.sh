#!/bin/bash
# Periodically scan & delete ALL tito_state.json anywhere under training_runs
# to reclaim space (depth-agnostic: any <ROOT>/**/tito_state.json).
#
# SAFETY: only deletes files NOT modified in the last AGE_MIN minutes, so the
# state of a currently-running trial is never removed mid-write. (The current
# session-server runs don't write tito_state.json at all; this guard only
# matters if a custom-backend run is active.)
#
# Tunables (env):
#   ROOT      base dir            (default /root/data/training_runs)
#   AGE_MIN   skip-if-newer-than  (default 15 min)
#   INTERVAL  loop sleep seconds  (default 900 = 15 min)
#   LOG       log file            (default /root/tito_state_cleanup.log)
set -uo pipefail
ROOT="${ROOT:-/root/data/training_runs}"
AGE_MIN="${AGE_MIN:-15}"
INTERVAL="${INTERVAL:-900}"
LOG="${LOG:-/root/tito_state_cleanup.log}"
# Depth-agnostic: ALL tito_state.json anywhere under ROOT (was '*/trials/*/*/...').
PATTERN='tito_state.json'

log(){ echo "[$(date -u +'%F %T') UTC] $*" | tee -a "$LOG"; }

log "START loop: root=$ROOT name=$PATTERN age>${AGE_MIN}min interval=${INTERVAL}s"
while true; do
  listing="$(find "$ROOT" -name "$PATTERN" -type f -mmin +"$AGE_MIN" -printf '%s\t%p\n' 2>/dev/null)"
  if [ -n "$listing" ]; then
    cnt="$(printf '%s\n' "$listing" | wc -l)"
    mb="$(printf '%s\n' "$listing" | awk -F'\t' '{s+=$1} END{printf "%.1f", s/1048576}')"
    printf '%s\n' "$listing" | cut -f2- | while IFS= read -r f; do rm -f -- "$f"; done
    log "deleted $cnt files, freed ~${mb} MB"
  else
    log "nothing to delete (no matches older than ${AGE_MIN}min)"
  fi
  sleep "$INTERVAL"
done
