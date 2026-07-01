#!/usr/bin/env bash
# frpc_start.sh — Start FRP client on GPU machine. Creates one tunnel per sglang rank.
#
# Usage:
#   ./frpc_start.sh --server <RELAY_IP> --ranks "<IP:PORT>,<IP:PORT>,..." \
#     [--name <machine-name>] [--base-remote-port 39001] [--server-port 7000] [--dir ./frp]
#
# --name is used as a prefix for FRP proxy names to avoid collisions when
# multiple GPU machines register with the same frps relay. Defaults to "sglang".
#
# Example (DP4, single machine):
#   ./frpc_start.sh --server 65.108.32.175 --name gpu-a \
#     --ranks "172.18.0.2:31051,172.18.0.2:31052,172.18.0.2:31053,172.18.0.2:31054"
#
# Example (multi-machine, GPU Machine B with ranks 2-3):
#   ./frpc_start.sh --server 65.108.32.175 --name gpu-b \
#     --ranks "172.18.0.2:31051,172.18.0.3:31051" --base-remote-port 39003

set -euo pipefail

SERVER_ADDR=""
RANKS=""
MACHINE_NAME="sglang"
BASE_REMOTE_PORT=39001
SERVER_PORT=7000
FRP_DIR="$(pwd)/frp"
FRP_VERSION="0.61.1"
FRP_ARCHIVE="frp_${FRP_VERSION}_linux_amd64"
FRP_URL="https://github.com/fatedier/frp/releases/download/v${FRP_VERSION}/${FRP_ARCHIVE}.tar.gz"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --server)            SERVER_ADDR="$2";      shift 2 ;;
        --ranks)             RANKS="$2";            shift 2 ;;
        --name)              MACHINE_NAME="$2";     shift 2 ;;
        --base-remote-port)  BASE_REMOTE_PORT="$2"; shift 2 ;;
        --server-port)       SERVER_PORT="$2";      shift 2 ;;
        --dir)               FRP_DIR="$2";          shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$SERVER_ADDR" || -z "$RANKS" ]]; then
    echo "Usage: $0 --server <RELAY_IP> --ranks '<IP:PORT>,<IP:PORT>,...'"
    exit 1
fi

mkdir -p "$FRP_DIR"
cd "$FRP_DIR"

# Download frp if not present
if [[ ! -x ./frpc ]]; then
    echo "Downloading frp v${FRP_VERSION}..."
    curl -sSL "$FRP_URL" -o frp.tar.gz
    tar xzf frp.tar.gz --strip-components=1
    rm -f frp.tar.gz
    chmod +x ./frps ./frpc
    echo "Downloaded to $FRP_DIR"
fi

# Parse ranks into array
IFS=',' read -ra RANK_ARRAY <<< "$RANKS"

# Generate frpc.toml
cat > frpc.toml <<EOF
serverAddr = "${SERVER_ADDR}"
serverPort = ${SERVER_PORT}
loginFailExit = false

[transport]
poolCount = 50
heartbeatInterval = 10
heartbeatTimeout = 30
EOF

for i in "${!RANK_ARRAY[@]}"; do
    ENTRY="${RANK_ARRAY[$i]}"
    LOCAL_IP="${ENTRY%%:*}"
    LOCAL_PORT="${ENTRY##*:}"
    REMOTE_PORT=$((BASE_REMOTE_PORT + i))

    cat >> frpc.toml <<EOF

[[proxies]]
name = "${MACHINE_NAME}-rank${i}"
type = "tcp"
localIP = "${LOCAL_IP}"
localPort = ${LOCAL_PORT}
remotePort = ${REMOTE_PORT}
EOF
done

# Kill existing frpc
if pgrep -f "frpc.*-c" > /dev/null 2>&1; then
    echo "Killing existing frpc..."
    pkill -f "frpc.*-c" || true
    sleep 1
fi

# Start
nohup ./frpc -c frpc.toml > frpc.log 2>&1 &
FRPC_PID=$!
sleep 2

if kill -0 "$FRPC_PID" 2>/dev/null; then
    echo "Tunnel active (PID ${FRPC_PID})"
    for i in "${!RANK_ARRAY[@]}"; do
        ENTRY="${RANK_ARRAY[$i]}"
        REMOTE_PORT=$((BASE_REMOTE_PORT + i))
        echo "  Rank ${i}: ${ENTRY} -> http://${SERVER_ADDR}:${REMOTE_PORT}"
    done
    echo ""
    echo "Agents use: base_url = http://${SERVER_ADDR}:<RANK_PORT>/v1/{SESSION_ID}"
    echo "Log: $FRP_DIR/frpc.log"
else
    echo "ERROR: frpc failed to start. Check $FRP_DIR/frpc.log"
    tail -10 frpc.log 2>/dev/null
    exit 1
fi
