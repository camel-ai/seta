#!/usr/bin/env bash
# tunnel_status.sh — Check if FRP tunnel is alive.
#
# Config-driven (checks all machines):
#   ./tunnel_status.sh --config tunnel_config.yaml
#
# Standalone (single machine):
#   ./tunnel_status.sh --relay <IP> [--base-remote-port 39001] [--num-ranks 1]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG=""
RELAY=""
BASE_REMOTE_PORT=39001
NUM_RANKS=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)            CONFIG="$2";           shift 2 ;;
        --relay)             RELAY="$2";            shift 2 ;;
        --base-remote-port)  BASE_REMOTE_PORT="$2"; shift 2 ;;
        --num-ranks)         NUM_RANKS="$2";        shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# If --config given, delegate to manage_tunnel.py status
if [[ -n "$CONFIG" ]]; then
    exec python3 "$SCRIPT_DIR/manage_tunnel.py" --config "$CONFIG" status
fi

if [[ -z "$RELAY" ]]; then
    echo "Usage: $0 --config <tunnel_config.yaml>"
    echo "   or: $0 --relay <RELAY_IP> [--base-remote-port 39001] [--num-ranks 1]"
    exit 1
fi

echo "=== FRP Tunnel Status ==="
echo ""

# Check local processes
echo "Processes:"
if pgrep -f "frps.*-c" > /dev/null 2>&1; then
    echo "  frps: RUNNING (PID $(pgrep -f 'frps.*-c' | head -1))"
else
    echo "  frps: NOT RUNNING (may be on relay server)"
fi

if pgrep -f "frpc.*-c" > /dev/null 2>&1; then
    echo "  frpc: RUNNING (PID $(pgrep -f 'frpc.*-c' | head -1))"
else
    echo "  frpc: NOT RUNNING"
fi

echo ""
echo "Relay ports (${RELAY}):"

OK=0
FAIL=0
for i in $(seq 0 $((NUM_RANKS - 1))); do
    PORT=$((BASE_REMOTE_PORT + i))
    if timeout 2 bash -c "echo > /dev/tcp/${RELAY}/${PORT}" 2>/dev/null; then
        echo "  Rank ${i} :${PORT}  [OK]"
        OK=$((OK + 1))
    else
        echo "  Rank ${i} :${PORT}  [FAIL]"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "Summary: ${OK}/${NUM_RANKS} ranks reachable"
[[ "$FAIL" -eq 0 ]] && exit 0 || exit 1
