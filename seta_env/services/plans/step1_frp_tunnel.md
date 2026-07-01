# Step 1: FRP Tunnel Implementation

**Priority**: Highest — prerequisite for remote env_service to reach sglang on GPU machine.

**Status**: Not started

## Problem

RL training runs sglang inside Docker containers on GPU machines with outbound-only networking. Agents on remote CPU servers need to reach sglang. When new GPU machines join (different IPs, different rank counts), the tunnel setup must accommodate without editing scripts.

## Architecture

```
GPU Machine A (single machine today)     CPU Server 1 (65.108.32.175) — relay
┌──────────────────────────┐              ┌──────────────────────────┐
│ Rank 0: <IP_0>:<PORT_0>  │   frpc A     │ frps :7000 (control)     │
│ Rank 1: <IP_1>:<PORT_1>  │ ───────────► │                          │
│ ...                       │  outbound    │ :39001 → A Rank 0        │
│ Rank N: <IP_N>:<PORT_N>  │              │ :39002 → A Rank 1        │
└──────────────────────────┘              │ ...                      │
                                          │ :39001+N → A Rank N      │
GPU Machine B (joins later)               │                          │
┌──────────────────────────┐              │ :39101 → B Rank 0        │
│ Rank 0: <IP_0>:<PORT_0>  │   frpc B     │ :39102 → B Rank 1        │
│ Rank 1: <IP_1>:<PORT_1>  │ ───────────► │ ...                      │
└──────────────────────────┘  outbound    └──────────────────────────┘
                                                   ▲
                                          All CPU servers' agents
                                          hit relay_ip:port per rank
```

**Key**: Each GPU machine gets a non-overlapping port range on the relay. The port assignment lives in `tunnel_config.yaml`, not in CLI args.

## Design: Config-Driven Multi-Machine

### `tunnel_config.yaml` — Single source of truth

```yaml
relay:
  host: "65.108.32.175"         # CPU Server 1 runs frps
  port: 7000                    # frps control port
  ssh_key: /data/qijia/.ssh/id_ed25519
  ssh_user: root

gpu_machines:
  - name: gpu-a                 # human-readable name
    # ssh is optional — if omitted, frpc_start.sh must be run manually on that machine
    ssh_host: "185.216.21.30"
    ssh_user: root
    ssh_key: /data/qijia/.ssh/id_ed25519
    base_remote_port: 39001     # relay ports 39001..39001+num_ranks-1
    num_ranks: 4                # max ranks (port range reservation)
    # ranks are NOT in config — they're runtime-discovered from sglang
    # and passed to frpc_start.sh via --ranks

  # Adding a new GPU machine = add an entry with a new port range
  # - name: gpu-b
  #   ssh_host: "10.0.2.50"
  #   base_remote_port: 39101   # non-overlapping with gpu-a
  #   num_ranks: 8
```

**Port allocation convention**: Each machine reserves `num_ranks` ports starting at `base_remote_port`. To add a new machine, pick a base that doesn't overlap. Suggested: 39001, 39101, 39201, etc. (gaps of 100 for headroom).

### Layer 1: Low-level scripts (machine-agnostic)

These are building blocks. They take explicit args and know nothing about config files or machine names. They can be used standalone.

| Script | Runs on | Purpose |
|--------|---------|---------|
| `frps_start.sh` | Relay (CPU server) | Start frps with a bind port |
| `frpc_start.sh` | GPU machine | Start frpc with server + ranks + port range |
| `tunnel_status.sh` | Anywhere | TCP-probe relay ports |

### Layer 2: `manage_tunnel.py` (config-driven orchestrator)

Reads `tunnel_config.yaml` and calls the low-level scripts. Handles multi-machine coordination.

```
manage_tunnel.py deploy-relay          Deploy + start frps on relay via SSH
manage_tunnel.py start <machine>       Start frpc on a GPU machine (requires --ranks at runtime)
  --ranks "ip:port,ip:port"            Ranks discovered from sglang at training start
manage_tunnel.py stop <machine>        Kill frpc on a GPU machine
manage_tunnel.py status                Check all machines: frps + frpc + TCP probe all ports
manage_tunnel.py validate              Validate config: no port overlaps, SSH reachable
manage_tunnel.py info <machine>        Print the relay endpoints for a machine's ranks
```

**Why Python for orchestration**:
- YAML parsing is cleaner than shell
- Can validate port range overlaps at config load time
- Can be imported by training scripts to auto-start tunnels programmatically

### Layer 3: `test_tunnel.py` (verification)

HTTP-level test from CPU server. Uses httpx (no openai dependency needed for basic tests). Supports health-only mode (no sglang needed) and full load test (with sglang).

## Files to Create

```
frp_tunnel/
├── DESIGN.md               # (exists) architecture reference
├── tunnel_config.yaml       # multi-machine config (single source of truth)
├── frps_start.sh            # low-level: start frps on relay
├── frpc_start.sh            # low-level: start frpc on GPU machine
├── tunnel_status.sh         # low-level: TCP probe relay ports
├── manage_tunnel.py         # orchestrator: reads config, deploys, starts, stops
├── test_tunnel.py           # HTTP-level smoke + load test
└── deploy_and_test.sh       # end-to-end: deploy relay → start frpc → verify from CPU servers
```

## Implementation Details

### 1.1 `tunnel_config.yaml`

See above. Current config for single-machine case:

```yaml
relay:
  host: "65.108.32.175"
  port: 7000
  ssh_key: /data/qijia/.ssh/id_ed25519
  ssh_user: root

gpu_machines:
  - name: gpu-a
    base_remote_port: 39001
    num_ranks: 4
```

No `ssh_host` for gpu-a since it's this local machine (frpc runs locally).

### 1.2 `frps_start.sh`

No changes from current implementation. Takes `--port` and `--dir`. Downloads frp binary, generates frps.toml, starts with nohup.

### 1.3 `frpc_start.sh`

No changes needed — already parameterized with `--server`, `--ranks`, `--base-remote-port`, `--server-port`, `--dir`. Each GPU machine calls it with its own params.

One addition: `--name` flag for the proxy name prefix (default "sglang"). This prevents name collisions when multiple frpc instances register proxies on the same frps:

```toml
# With --name gpu-a:
[[proxies]]
name = "gpu-a-rank0"       # instead of "sglang-rank0"
```

### 1.4 `tunnel_status.sh`

Update: accept `--config` to read tunnel_config.yaml and check all machines' port ranges. Falls back to explicit `--relay`/`--num-ranks` args for standalone use.

```bash
# Config-driven (checks all machines):
./tunnel_status.sh --config tunnel_config.yaml

# Standalone (single machine):
./tunnel_status.sh --relay 65.108.32.175 --base-remote-port 39001 --num-ranks 4
```

### 1.5 `manage_tunnel.py`

```python
#!/usr/bin/env python3
"""Config-driven FRP tunnel orchestrator for multi-GPU-machine setups."""

import argparse, yaml, subprocess, sys, os
from pathlib import Path
from dataclasses import dataclass

SCRIPT_DIR = Path(__file__).parent

@dataclass
class RelayConfig:
    host: str
    port: int = 7000
    ssh_key: str = ""
    ssh_user: str = "root"

@dataclass
class GpuMachineConfig:
    name: str
    base_remote_port: int
    num_ranks: int
    ssh_host: str = ""          # empty = local machine
    ssh_user: str = "root"
    ssh_key: str = ""

class TunnelConfig:
    def __init__(self, path: str):
        data = yaml.safe_load(open(path))
        r = data["relay"]
        self.relay = RelayConfig(
            host=r["host"], port=r.get("port", 7000),
            ssh_key=r.get("ssh_key", ""), ssh_user=r.get("ssh_user", "root"),
        )
        self.machines = []
        for m in data.get("gpu_machines", []):
            self.machines.append(GpuMachineConfig(
                name=m["name"],
                base_remote_port=m["base_remote_port"],
                num_ranks=m["num_ranks"],
                ssh_host=m.get("ssh_host", ""),
                ssh_user=m.get("ssh_user", "root"),
                ssh_key=m.get("ssh_key", ""),
            ))

    def get_machine(self, name: str) -> GpuMachineConfig:
        for m in self.machines:
            if m.name == name:
                return m
        raise ValueError(f"Machine '{name}' not found in config. "
                         f"Available: {[m.name for m in self.machines]}")

    def validate(self):
        """Check for port range overlaps."""
        ranges = []
        for m in self.machines:
            start = m.base_remote_port
            end = start + m.num_ranks - 1
            for name, s, e in ranges:
                if start <= e and s <= end:
                    raise ValueError(
                        f"Port overlap: {m.name} [{start}-{end}] "
                        f"overlaps {name} [{s}-{e}]"
                    )
            ranges.append((m.name, start, end))

    def get_relay_endpoints(self, machine_name: str) -> list[str]:
        """Return relay URLs for a machine's ranks."""
        m = self.get_machine(machine_name)
        return [
            f"http://{self.relay.host}:{m.base_remote_port + i}"
            for i in range(m.num_ranks)
        ]

# Commands: deploy_relay, start, stop, status, validate, info
# Each calls the low-level shell scripts with the right args from config.
```

**`deploy-relay`**: SSH to relay host, copy frps_start.sh, run it.

**`start <machine> --ranks "ip:port,..."**:
```python
def cmd_start(cfg, args):
    machine = cfg.get_machine(args.machine)
    frpc_cmd = [
        "bash", str(SCRIPT_DIR / "frpc_start.sh"),
        "--server", cfg.relay.host,
        "--server-port", str(cfg.relay.port),
        "--ranks", args.ranks,
        "--base-remote-port", str(machine.base_remote_port),
        "--name", machine.name,
    ]
    if machine.ssh_host:
        # Run on remote GPU machine via SSH
        ssh_run(machine, frpc_cmd)
    else:
        # Run locally
        subprocess.run(frpc_cmd, check=True)
```

**`status`**: For each machine, TCP-probe its port range on the relay.

**`info <machine>`**: Print the relay endpoints that agents should use:
```
Machine: gpu-a
  Rank 0: http://65.108.32.175:39001
  Rank 1: http://65.108.32.175:39002
  ...
  Agent base_url: http://65.108.32.175:{RANK_PORT}/v1/{SESSION_ID}
```

### 1.6 `test_tunnel.py`

Updated to support both config-driven and standalone modes:

```bash
# Config-driven: test all ranks of a machine
python test_tunnel.py --config tunnel_config.yaml --machine gpu-a

# Standalone: test a single endpoint
python test_tunnel.py --base-url http://65.108.32.175:39001/v1

# Health-only (no sglang needed, just check TCP + HTTP)
python test_tunnel.py --config tunnel_config.yaml --machine gpu-a --health-only

# Load test with sglang
python test_tunnel.py --base-url http://65.108.32.175:39001/v1 --concurrency 64
```

### 1.7 `deploy_and_test.sh`

End-to-end script that:
1. Reads `tunnel_config.yaml` (via manage_tunnel.py)
2. Deploys frps to relay
3. Starts a mock HTTP server locally (for testing without sglang)
4. Starts frpc locally with mock server as rank
5. Tests connectivity from relay and all CPU servers
6. Reports pass/fail
7. Cleans up (--cleanup flag)

```bash
#!/usr/bin/env bash
# Usage: ./deploy_and_test.sh [--config tunnel_config.yaml] [--cleanup]
#        [--test-from "65.108.32.175,135.181.71.8"]

# 1. python manage_tunnel.py validate --config $CONFIG
# 2. python manage_tunnel.py deploy-relay --config $CONFIG
# 3. Start mock HTTP server locally on a temp port
# 4. python manage_tunnel.py start gpu-a --config $CONFIG --ranks "127.0.0.1:$MOCK_PORT"
# 5. Test from relay: curl http://$RELAY:$REMOTE_PORT/test
# 6. Test from each --test-from server via SSH
# 7. python manage_tunnel.py status --config $CONFIG
# 8. Report results
# 9. Optional cleanup
```

## Multi-Machine Workflow

### Adding a new GPU machine

1. Edit `tunnel_config.yaml`:
```yaml
gpu_machines:
  - name: gpu-a
    base_remote_port: 39001
    num_ranks: 4
  - name: gpu-b                 # NEW
    ssh_host: "10.0.2.50"       # NEW
    ssh_key: /data/qijia/.ssh/id_ed25519
    base_remote_port: 39101     # non-overlapping
    num_ranks: 8
```

2. Validate: `python manage_tunnel.py validate` (catches port overlaps)

3. At training start on gpu-b:
```bash
python manage_tunnel.py start gpu-b --ranks "172.18.0.2:31051,172.18.0.2:31052,..."
```

4. Agents on CPU servers use endpoints from `manage_tunnel.py info gpu-b`:
```
http://65.108.32.175:39101/v1/{SESSION_ID}   # gpu-b rank 0
http://65.108.32.175:39102/v1/{SESSION_ID}   # gpu-b rank 1
```

No script changes needed. Only the config file changes.

### Programmatic usage (from training script)

```python
from seta_env.services.frp_tunnel.manage_tunnel import TunnelConfig

cfg = TunnelConfig("tunnel_config.yaml")
cfg.validate()
endpoints = cfg.get_relay_endpoints("gpu-a")
# → ["http://65.108.32.175:39001", "http://65.108.32.175:39002", ...]

# Pass to model config:
model_config = {"url": endpoints[rank_index] + "/v1", ...}
```

## Testing Plan

### Test 1: Config validation
```bash
python manage_tunnel.py validate --config tunnel_config.yaml
# Expect: "Config valid: 1 machine, no port overlaps"

# Add overlapping port range → expect error
```

### Test 2: Deploy + mock test (no sglang)
```bash
./deploy_and_test.sh --config tunnel_config.yaml \
  --test-from "65.108.32.175,135.181.71.8" --cleanup
# Deploys frps, starts mock server + frpc, tests from all CPU servers
```

### Test 3: Multi-rank mock test
```bash
# Manually start 4 mock servers, test 4-rank tunnel
./deploy_and_test.sh --config tunnel_config.yaml --num-mock-ranks 4 --cleanup
```

### Test 4: Sglang test (when available)
```bash
# After sglang starts:
python manage_tunnel.py start gpu-a --ranks "<sglang_ranks>"
python test_tunnel.py --config tunnel_config.yaml --machine gpu-a --concurrency 64
```

## Dependencies

- `curl`, `tar` (for downloading frp binary)
- `python3`, `pyyaml` (for manage_tunnel.py)
- `httpx` (for test_tunnel.py)
- No sudo, no root, no Docker needed for the tunnel itself
