#!/usr/bin/env bash
# deploy_and_test.sh — End-to-end: deploy relay → start frpc with mock → verify from CPU servers.
#
# Usage:
#   ./deploy_and_test.sh [--config tunnel_config.yaml] [--machine gpu-a]
#                        [--num-mock-ranks 1] [--test-from "135.181.71.8"] [--cleanup]
#
# What it does:
#   1. Validate config (no port overlaps)
#   2. Deploy frps to relay via manage_tunnel.py
#   3. Start a local mock HTTP server (one per rank)
#   4. Start frpc via manage_tunnel.py (pointing at mock servers)
#   5. Test connectivity from relay (localhost + public IP)
#   6. Test connectivity from all cpu_servers in config via SSH
#   7. Run tunnel_status.sh
#   8. Report results
#   9. Optionally cleanup
#
# Example:
#   ./deploy_and_test.sh --config tunnel_config.yaml --machine gpu-a --cleanup

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG="$SCRIPT_DIR/tunnel_config.yaml"
MACHINE="gpu-a"
NUM_MOCK_RANKS=1
TEST_FROM=""
CLEANUP=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)          CONFIG="$2";          shift 2 ;;
        --machine)         MACHINE="$2";         shift 2 ;;
        --num-mock-ranks)  NUM_MOCK_RANKS="$2";  shift 2 ;;
        --test-from)       TEST_FROM="$2";       shift 2 ;;
        --cleanup)         CLEANUP=true;         shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

PYTHON="${PYTHON:-python3}"
MANAGE="$PYTHON $SCRIPT_DIR/manage_tunnel.py --config $CONFIG"
PASSED=0
FAILED=0
MOCK_PIDS=()

pass() { echo "  PASS: $1"; PASSED=$((PASSED + 1)); }
fail() { echo "  FAIL: $1"; FAILED=$((FAILED + 1)); }

cleanup() {
    echo ""
    echo "=== Cleanup ==="
    # Kill mock servers
    for pid in "${MOCK_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            echo "  Killed mock server (PID $pid)"
        fi
    done
    # Stop frpc
    $MANAGE stop "$MACHINE" 2>/dev/null || true
    echo "  Stopped frpc for $MACHINE"
}

trap 'cleanup' EXIT

# ── Read config values for testing ────────────────────────────────────────────

# Extract relay host and machine port info from config via Python
eval "$($PYTHON - "$CONFIG" "$MACHINE" <<'PY'
import sys, yaml, shlex
cfg = yaml.safe_load(open(sys.argv[1]))
def q(v):
    return shlex.quote(str(v))
print(f"RELAY_HOST={q(cfg['relay']['host'])}")
print(f"RELAY_SSH_KEY={q(cfg['relay'].get('ssh_key', ''))}")
print(f"RELAY_SSH_USER={q(cfg['relay'].get('ssh_user', 'root'))}")
for m in cfg.get('gpu_machines', []):
    if m['name'] == sys.argv[2]:
        print(f"BASE_PORT={q(m['base_remote_port'])}")
        break
PY
)"

SSH_OPTS="-o StrictHostKeyChecking=no -o BatchMode=yes"

# ── Step 1: Validate config ──────────────────────────────────────────────────

echo "=== Step 1: Validate config ==="
if $MANAGE validate; then
    pass "Config validation"
else
    fail "Config validation"
    exit 1
fi

# ── Step 2: Deploy frps to relay ─────────────────────────────────────────────

echo ""
echo "=== Step 2: Deploy frps to relay ==="
if $MANAGE deploy-relay; then
    pass "frps deployed to $RELAY_HOST"
else
    fail "frps deployment"
    exit 1
fi

# ── Step 3: Start mock HTTP servers ──────────────────────────────────────────

echo ""
echo "=== Step 3: Start mock HTTP servers (${NUM_MOCK_RANKS} ranks) ==="

MOCK_BASE_PORT=18080
RANKS_ARG=""

for i in $(seq 0 $((NUM_MOCK_RANKS - 1))); do
    PORT=$((MOCK_BASE_PORT + i))

    # Kill anything on this port
    if lsof -ti:${PORT} > /dev/null 2>&1; then
        kill $(lsof -ti:${PORT}) 2>/dev/null || true
        sleep 0.5
    fi

    $PYTHON -c "
import http.server, json

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'status':'ok','rank':${i},'path':self.path}).encode())
    def do_POST(self): self.do_GET()
    def log_message(self, *a): pass

http.server.HTTPServer(('0.0.0.0', ${PORT}), H).serve_forever()
" &
    MOCK_PIDS+=($!)
    sleep 0.3

    # Build ranks argument
    [[ -n "$RANKS_ARG" ]] && RANKS_ARG+=","
    RANKS_ARG+="127.0.0.1:${PORT}"
done

# Verify mock servers
ALL_MOCK_OK=true
for i in $(seq 0 $((NUM_MOCK_RANKS - 1))); do
    PORT=$((MOCK_BASE_PORT + i))
    if curl -sf "http://127.0.0.1:${PORT}/test" > /dev/null 2>&1; then
        pass "Mock server rank $i on port $PORT"
    else
        fail "Mock server rank $i on port $PORT"
        ALL_MOCK_OK=false
    fi
done
[[ "$ALL_MOCK_OK" == "false" ]] && exit 1

# ── Step 4: Start frpc ───────────────────────────────────────────────────────

echo ""
echo "=== Step 4: Start frpc for $MACHINE ==="
if $MANAGE start "$MACHINE" --ranks "$RANKS_ARG"; then
    pass "frpc started for $MACHINE"
else
    fail "frpc start"
    exit 1
fi

# Wait for tunnel to establish
sleep 3

# ── Step 5: Test from relay ──────────────────────────────────────────────────

echo ""
echo "=== Step 5: Test connectivity from relay ==="

for i in $(seq 0 $((NUM_MOCK_RANKS - 1))); do
    REMOTE_PORT=$((BASE_PORT + i))

    # From relay localhost
    RESP=$(ssh $SSH_OPTS -i "$RELAY_SSH_KEY" "${RELAY_SSH_USER}@${RELAY_HOST}" \
        "curl -sf http://127.0.0.1:${REMOTE_PORT}/tunnel-test 2>&1" || echo "FAILED")
    if echo "$RESP" | grep -q '"status"'; then
        pass "Relay localhost → rank $i (:${REMOTE_PORT})"
    else
        fail "Relay localhost → rank $i (:${REMOTE_PORT}): $RESP"
    fi

    # From relay public IP
    RESP2=$(ssh $SSH_OPTS -i "$RELAY_SSH_KEY" "${RELAY_SSH_USER}@${RELAY_HOST}" \
        "curl -sf http://${RELAY_HOST}:${REMOTE_PORT}/tunnel-test 2>&1" || echo "FAILED")
    if echo "$RESP2" | grep -q '"status"'; then
        pass "Relay public IP → rank $i (${RELAY_HOST}:${REMOTE_PORT})"
    else
        fail "Relay public IP → rank $i: $RESP2"
    fi
done

# ── Step 6: Test from CPU servers ─────────────────────────────────────────────

if [[ -n "$TEST_FROM" ]]; then
    echo ""
    echo "=== Step 6: Test connectivity from other servers ==="

    IFS=',' read -ra SERVER_LIST <<< "$TEST_FROM"
    for HOST in "${SERVER_LIST[@]}"; do
        HOST=$(echo "$HOST" | xargs)  # trim whitespace
        [[ "$HOST" == "$RELAY_HOST" ]] && continue  # already tested from relay

        for i in $(seq 0 $((NUM_MOCK_RANKS - 1))); do
            REMOTE_PORT=$((BASE_PORT + i))
            RESP=$(ssh $SSH_OPTS -i "$RELAY_SSH_KEY" "root@${HOST}" \
                "curl -sf http://${RELAY_HOST}:${REMOTE_PORT}/tunnel-test 2>&1" || echo "FAILED")
            if echo "$RESP" | grep -q '"status"'; then
                pass "${HOST} → rank $i (${RELAY_HOST}:${REMOTE_PORT})"
            else
                fail "${HOST} → rank $i: $RESP"
            fi
        done
    done
fi

# ── Step 7: Run tunnel_status.sh ─────────────────────────────────────────────

echo ""
echo "=== Step 7: tunnel_status.sh ==="
if bash "$SCRIPT_DIR/tunnel_status.sh" --relay "$RELAY_HOST" \
    --base-remote-port "$BASE_PORT" --num-ranks "$NUM_MOCK_RANKS"; then
    pass "tunnel_status.sh reports healthy"
else
    fail "tunnel_status.sh reports unhealthy"
fi

# ── Results ───────────────────────────────────────────────────────────────────

echo ""
echo "========================================="
echo "Results: ${PASSED} passed, ${FAILED} failed"
echo "========================================="

if [[ "$CLEANUP" != "true" ]]; then
    echo ""
    echo "Tunnel still running. Use --cleanup or Ctrl-C to stop."
    echo "To run sglang test later:"
    echo "  python $SCRIPT_DIR/test_tunnel.py --base-url http://${RELAY_HOST}:${BASE_PORT}/v1"
    # Keep alive until Ctrl-C
    trap 'cleanup; exit 0' INT TERM
    wait
fi

[[ "$FAILED" -eq 0 ]] && exit 0 || exit 1
