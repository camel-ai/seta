#!/usr/bin/env bash
# frps_start.sh — Start FRP server (relay) on a CPU server.
#
# Usage: ./frps_start.sh [--port 7000] [--dir ./frp]
#
# Downloads frp binary if not present. Runs in background with nohup.
# Safe to re-run (kills existing frps first).

set -euo pipefail

BIND_PORT=7000
FRP_DIR="$(pwd)/frp"
FRP_VERSION="0.61.1"
FRP_ARCHIVE="frp_${FRP_VERSION}_linux_amd64"
FRP_URL="https://github.com/fatedier/frp/releases/download/v${FRP_VERSION}/${FRP_ARCHIVE}.tar.gz"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) BIND_PORT="$2"; shift 2 ;;
        --dir)  FRP_DIR="$2";   shift 2 ;;
        *)      echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$FRP_DIR"
cd "$FRP_DIR"

# Download frp if not present
if [[ ! -x ./frps ]]; then
    echo "Downloading frp v${FRP_VERSION}..."
    curl -sSL "$FRP_URL" -o frp.tar.gz
    tar xzf frp.tar.gz --strip-components=1
    rm -f frp.tar.gz
    chmod +x ./frps ./frpc
    echo "Downloaded to $FRP_DIR"
fi

# Kill existing frps
if pgrep -f "frps.*-c" > /dev/null 2>&1; then
    echo "Killing existing frps..."
    pkill -f "frps.*-c" || true
    sleep 1
fi

# Open firewall ports (frps control + tunnel port range)
if command -v ufw &>/dev/null && ufw status | grep -q "active"; then
    ufw allow "${BIND_PORT}/tcp" 2>/dev/null || true
    # Open a generous tunnel port range (39001-39999)
    ufw allow 39001:39999/tcp 2>/dev/null || true
    echo "Firewall: opened port ${BIND_PORT} + 39001-39999"
elif command -v firewall-cmd &>/dev/null; then
    firewall-cmd --permanent --add-port="${BIND_PORT}/tcp" 2>/dev/null || true
    firewall-cmd --permanent --add-port=39001-39999/tcp 2>/dev/null || true
    firewall-cmd --reload 2>/dev/null || true
fi

# Generate config
cat > frps.toml <<EOF
bindPort = ${BIND_PORT}
EOF

# Start
nohup ./frps -c frps.toml > frps.log 2>&1 &
FRPS_PID=$!
sleep 1

if kill -0 "$FRPS_PID" 2>/dev/null; then
    echo "frps started (PID ${FRPS_PID}) on port ${BIND_PORT}"
    echo "Log: $FRP_DIR/frps.log"
else
    echo "ERROR: frps failed to start. Check $FRP_DIR/frps.log"
    tail -5 frps.log 2>/dev/null
    exit 1
fi
