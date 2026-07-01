# FRP Tunnel: GPU Container ↔ CPU Agent Servers

## Problem

RL training runs inside a Docker container on a GPU server with no exposed ports. Agentic workloads (256+ concurrent agents) run on 1-N remote CPU servers and need to make OpenAI-compatible HTTP requests to sglang inside the container. The container allows outbound connections but no inbound.

## Solution

Use FRP (Fast Reverse Proxy) to create a TCP tunnel from the GPU container to a relay on CPU Server 1. All agents across all CPU servers hit the relay endpoint.

## Architecture

With data parallelism (e.g., DP4), each rank runs its own sglang proxy server with a distinct IP:port. Each rank needs its own tunnel.

### Single GPU machine (all ranks co-located)

One frpc process with multiple proxy entries:

```
GPU Docker Container                     CPU Server 1 (relay)
┌──────────────────────────┐             ┌──────────────────────┐
│ Rank 0: <IP_0>:<PORT_0>  │             │ frps :7000 (control) │
│ Rank 1: <IP_1>:<PORT_1>  │  1x frpc    │                      │
│ Rank 2: <IP_2>:<PORT_2>  │ ──────────► │ :39001 → Rank 0      │
│ Rank 3: <IP_3>:<PORT_3>  │  outbound   │ :39002 → Rank 1      │
└──────────────────────────┘             │ :39003 → Rank 2      │
                                         │ :39004 → Rank 3      │
                                         └──────────────────────┘
```

### Multi GPU machine (ranks across machines)

Each GPU machine runs its own frpc, all connecting to the same frps:

```
GPU Machine A                            CPU Server 1 (relay)
┌──────────────────────┐                 ┌──────────────────────┐
│ Rank 0: <IP_0>:<PORT>│  frpc A ──────► │ frps :7000           │
│ Rank 1: <IP_1>:<PORT>│                 │                      │
└──────────────────────┘                 │ :39001 → Rank 0      │
                                         │ :39002 → Rank 1      │
GPU Machine B                            │ :39003 → Rank 2      │
┌──────────────────────┐                 │ :39004 → Rank 3      │
│ Rank 2: <IP_2>:<PORT>│  frpc B ──────► │                      │
│ Rank 3: <IP_3>:<PORT>│                 └──────────┬───────────┘
└──────────────────────┘                            │
                                         All CPU servers' agents
                                         hit CPU_SERVER_1_IP:3900X
```

### Port mapping convention

Remote port = `BASE_REMOTE_PORT + rank_index`. Default base: 39001.

```
Rank 0 → :39001
Rank 1 → :39002
Rank 2 → :39003
...
Rank N → :39001+N
```

## Lifecycle

- **frps (relay)**: Long-lived. Set up once on CPU Server 1, leave it running permanently.
- **frpc (client)**: Ephemeral. Launched dynamically at training start inside each GPU container, reads the actual sglang IP + port per rank at runtime.

Workflow:
1. CPU servers: `frps_start.sh` runs once (or on boot), stays up indefinitely
2. Training starts → each rank's sglang proxy binds to `<IP_R>:<PORT_R>`
3. `frpc_start.sh` is called with a rank config (list of local_ip:local_port pairs), generates one proxy entry per rank, starts frpc
4. Per agent session, the RL framework assigns a session-specific URL per rank. Agents use the tunneled equivalent: `http://<CPU_SERVER_1_IP>:<39001+rank>/v1/<SESSION_ID>`
5. Training ends → frpc dies with the container (or is killed). frps stays up for the next run.

## URL Format

The RL framework assigns per-session URLs:

```
http://<SGLANG_IP>:{PORT}/v1/{SESSION_ID}
```

- PORT: fixed per training run (e.g., 31051), selectable
- SESSION_ID: unique per session (e.g., 49d0568a995346568d929eb3024041bf)
- API key: same as SESSION_ID

FRP is a raw TCP tunnel — URLs, paths, headers, and keys pass through byte-for-byte unchanged.

## Components to Implement

### 1. `frps_start.sh` — Run on CPU Server 1

Starts the FRP server (relay). Downloads the binary if not present.

```
Usage: ./frps_start.sh [--port 7000]
```

Config (`frps.toml`):
```toml
bindPort = 7000
```

Requirements:
- No sudo needed
- Download frp v0.61.1 linux amd64 binary if not present
- Run in background with nohup, log to frps.log
- Print confirmation message with PID

### 2. `frpc_start.sh` — Run inside GPU Docker container at training start

Starts the FRP client with one proxy entry per rank. Session IDs are per-agent-session and handled at the application layer, not the tunnel.

```
Usage: ./frpc_start.sh --server CPU_SERVER_IP --ranks "<IP_0>:<PORT_0>,<IP_1>:<PORT_1>,..." [--base-remote-port 39001] [--server-port 7000]
```

The script must:
1. Parse `--ranks` comma-separated list of `ip:port` pairs
2. Generate frpc.toml with one `[[proxies]]` entry per rank, remote port = base + rank index
3. Kill any existing frpc process
4. Start frpc
5. Print the agent-facing relay endpoint for each rank

Example (DP4, single GPU machine):
```bash
./frpc_start.sh --server 10.0.1.50 \
  --ranks "172.18.0.2:31051,172.18.0.2:31052,172.18.0.2:31053,172.18.0.2:31054"

# Output:
# Tunnel active (PID 12345)
# Rank 0: 172.18.0.2:31051 → http://10.0.1.50:39001
# Rank 1: 172.18.0.2:31052 → http://10.0.1.50:39002
# Rank 2: 172.18.0.2:31053 → http://10.0.1.50:39003
# Rank 3: 172.18.0.2:31054 → http://10.0.1.50:39004
# Agents use: base_url = http://10.0.1.50:<RANK_PORT>/v1/{SESSION_ID}
```

Example (DP4, multi GPU machine — run on each machine with its local ranks):
```bash
# GPU Machine A (ranks 0-1)
./frpc_start.sh --server 10.0.1.50 \
  --ranks "172.18.0.2:31051,172.18.0.3:31051" \
  --base-remote-port 39001

# GPU Machine B (ranks 2-3)
./frpc_start.sh --server 10.0.1.50 \
  --ranks "172.18.0.2:31051,172.18.0.3:31051" \
  --base-remote-port 39003
```

Generated config (`frpc.toml`) for DP4:
```toml
serverAddr = "<CPU_SERVER_IP>"
serverPort = 7000
loginFailExit = false
transport.poolCount = 50
transport.heartbeatInterval = 10
transport.heartbeatTimeout = 30

[[proxies]]
name = "sglang-rank0"
type = "tcp"
localIP = "<IP_0>"
localPort = <PORT_0>
remotePort = 39001

[[proxies]]
name = "sglang-rank1"
type = "tcp"
localIP = "<IP_1>"
localPort = <PORT_1>
remotePort = 39002

[[proxies]]
name = "sglang-rank2"
type = "tcp"
localIP = "<IP_2>"
localPort = <PORT_2>
remotePort = 39003

[[proxies]]
name = "sglang-rank3"
type = "tcp"
localIP = "<IP_3>"
localPort = <PORT_3>
remotePort = 39004
```

Requirements:
- No sudo needed
- Download frp binary if not present
- Kill any existing frpc process before starting
- Run in background with nohup, log to frpc.log
- Print the resulting agent base_url

### 3. `test_tunnel.py` — Verify the tunnel works

Quick smoke test from any CPU server.

```
Usage: python test_tunnel.py --base-url http://CPU_SERVER_1_IP:39001/v1/SESSION_ID --api-key SESSION_ID [--concurrency 64]
```

Steps:
1. Health check: GET /models
2. Single completion request
3. Concurrent load test: N async requests using asyncio + openai AsyncOpenAI
4. Report success/failure count and latency stats (min, max, mean, p99)

Requirements:
- Use `openai` Python package (AsyncOpenAI)
- Timeout 120s per request (sglang may queue)
- max_retries=3 on the client
- Print clear pass/fail output

### 4. `tunnel_status.sh` — Check if tunnel is alive

```
Usage: ./tunnel_status.sh [--base-remote-port 39001] [--num-ranks 4]
```

- Check if frps process is running (on relay server)
- Check if frpc process is running (in GPU container)
- Attempt a TCP connection to each rank's relay port (39001 through 39001+num_ranks-1)
- Print per-rank status summary

## Agent Configuration

Each agent is assigned a rank. All agents across all CPU servers use the same relay IP, differing only by rank port:

```python
from openai import AsyncOpenAI

# rank_port = 39001 + assigned_rank_index
client = AsyncOpenAI(
    base_url=f"http://{CPU_SERVER_1_IP}:{rank_port}/v1/{SESSION_ID}",
    api_key=SESSION_ID,
    timeout=120.0,
    max_retries=3,
)
```

The RL framework distributes agents across ranks for load balancing (e.g., 64 agents per rank in DP4 with 256 total agents).

When ports are directly accessible (no Docker restriction), skip FRP entirely and point agents directly at sglang.

## Key Design Decisions

- **FRP over SSH tunnels**: SSH works but doesn't have connection pooling; FRP pre-warms 50 connections for burst traffic
- **FRP over Cloudflare tunnels**: Cloudflare has undocumented rate limits and adds public internet latency; FRP is self-hosted with zero limits
- **Single relay**: All CPU servers route through CPU Server 1; no per-server tunnel setup needed
- **TCP mode**: FRP forwards raw TCP, completely protocol-agnostic; all HTTP paths/headers/auth pass through unchanged
- **`transport.poolCount = 50`**: Pre-established connections avoid handshake latency when 256 agents burst simultaneously
- **`loginFailExit = false`**: frpc retries forever on disconnect; agents see brief errors then auto-recover

## Failure Modes

| Failure | Impact | Recovery |
|---------|--------|----------|
| frpc disconnects | All agents get connection refused | frpc auto-reconnects; agents retry via max_retries=3 |
| frps crashes | All agents get connection refused | Restart frps_start.sh; frpc reconnects automatically |
| Single agent stream dies | One request fails | Other agents unaffected; agent retries |
| sglang overloaded | Slow responses (queued) | Not a tunnel issue; tune sglang --max-num-seqs |

## File Structure

```
frp-tunnel/
├── frps_start.sh        # CPU server relay
├── frpc_start.sh        # GPU container client
├── test_tunnel.py        # Smoke + load test
└── tunnel_status.sh      # Health check
```