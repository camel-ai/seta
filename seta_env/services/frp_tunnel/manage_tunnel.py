#!/usr/bin/env python3
"""Config-driven FRP tunnel orchestrator for multi-GPU-machine setups.

Commands:
    validate                       Check config for port overlaps and SSH reachability
    deploy-relay                   Deploy + start frps on the relay server
    start <machine> --ranks ...    Start frpc on a GPU machine (local or remote)
    stop <machine>                 Kill frpc on a GPU machine
    status                         Check all tunnels: frps + frpc + TCP probe
    info <machine>                 Print relay endpoints for a machine's ranks

Usage:
    python manage_tunnel.py --config tunnel_config.yaml validate
    python manage_tunnel.py --config tunnel_config.yaml deploy-relay
    python manage_tunnel.py --config tunnel_config.yaml start gpu-a --ranks "127.0.0.1:31051"
    python manage_tunnel.py --config tunnel_config.yaml status
    python manage_tunnel.py --config tunnel_config.yaml info gpu-a
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).parent


# ── Config dataclasses ──────────────────────────────────────────────────────


@dataclass
class RelayConfig:
    host: str
    port: int = 7000
    ssh_key: str = ""
    ssh_user: str = "root"


@dataclass
class CpuServerConfig:
    host: str
    ssh_key: str = ""
    ssh_user: str = "root"


@dataclass
class GpuMachineConfig:
    name: str
    base_remote_port: int
    num_ranks: int
    ssh_host: str = ""  # empty = local machine (run frpc here)
    ssh_user: str = "root"
    ssh_key: str = ""

    @property
    def is_local(self) -> bool:
        return not self.ssh_host

    @property
    def port_range(self) -> tuple[int, int]:
        return (self.base_remote_port, self.base_remote_port + self.num_ranks - 1)


@dataclass
class TunnelConfig:
    relay: RelayConfig
    cpu_servers: list[CpuServerConfig] = field(default_factory=list)
    machines: list[GpuMachineConfig] = field(default_factory=list)

    @classmethod
    def load(cls, path: str) -> TunnelConfig:
        data = yaml.safe_load(open(path))
        r = data["relay"]
        relay = RelayConfig(
            host=r["host"],
            port=r.get("port", 7000),
            ssh_key=r.get("ssh_key", ""),
            ssh_user=r.get("ssh_user", "root"),
        )
        cpu_servers = []
        machines = [
            GpuMachineConfig(
                name=m["name"],
                base_remote_port=m["base_remote_port"],
                num_ranks=m["num_ranks"],
                ssh_host=m.get("ssh_host", ""),
                ssh_user=m.get("ssh_user", "root"),
                ssh_key=m.get("ssh_key", ""),
            )
            for m in data.get("gpu_machines", [])
        ]
        return cls(relay=relay, cpu_servers=cpu_servers, machines=machines)

    def get_machine(self, name: str) -> GpuMachineConfig:
        for m in self.machines:
            if m.name == name:
                return m
        available = [m.name for m in self.machines]
        raise ValueError(f"Machine '{name}' not found. Available: {available}")

    def validate(self) -> list[str]:
        """Check for port range overlaps. Returns list of error strings."""
        errors = []
        ranges = []
        for m in self.machines:
            start, end = m.port_range
            for other_name, other_start, other_end in ranges:
                if start <= other_end and other_start <= end:
                    errors.append(
                        f"Port overlap: {m.name} [{start}-{end}] "
                        f"overlaps {other_name} [{other_start}-{other_end}]"
                    )
            ranges.append((m.name, start, end))
        return errors

    def get_relay_endpoints(self, machine_name: str) -> list[str]:
        """Return relay URLs for a machine's ranks."""
        m = self.get_machine(machine_name)
        return [
            f"http://{self.relay.host}:{m.base_remote_port + i}"
            for i in range(m.num_ranks)
        ]


# ── SSH helper ──────────────────────────────────────────────────────────────


def ssh_cmd(host: str, user: str, key: str, cmd: str, check: bool = True):
    """Run a command on a remote host via SSH."""
    ssh_args = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
    ]
    if key:
        ssh_args += ["-i", key]
    ssh_args += [f"{user}@{host}", cmd]
    return subprocess.run(ssh_args, capture_output=True, text=True, check=check)


def scp_to(host: str, user: str, key: str, local_path: str, remote_path: str):
    """Copy a file to a remote host via SCP."""
    scp_args = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
    ]
    if key:
        scp_args += ["-i", key]
    scp_args += [local_path, f"{user}@{host}:{remote_path}"]
    return subprocess.run(scp_args, capture_output=True, text=True, check=True)


def tcp_probe(host: str, port: int, timeout: float = 2.0) -> bool:
    """Test TCP connectivity to host:port."""
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except (OSError, socket.timeout):
        return False


# ── Commands ────────────────────────────────────────────────────────────────


def cmd_validate(cfg: TunnelConfig, args: argparse.Namespace) -> int:
    errors = cfg.validate()
    if errors:
        print("Config validation FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(
        f"Config valid: {len(cfg.machines)} machine(s), "
        f"no port overlaps"
    )
    for m in cfg.machines:
        start, end = m.port_range
        loc = "local" if m.is_local else m.ssh_host
        print(f"  {m.name}: ports {start}-{end} ({loc})")
    return 0


def cmd_deploy_relay(cfg: TunnelConfig, args: argparse.Namespace) -> int:
    relay = cfg.relay
    print(f"Deploying frps to {relay.host}...")

    # Create dir on relay
    ssh_cmd(relay.host, relay.ssh_user, relay.ssh_key, "mkdir -p /opt/frp")

    # Copy frps_start.sh
    scp_to(
        relay.host, relay.ssh_user, relay.ssh_key,
        str(SCRIPT_DIR / "frps_start.sh"), "/opt/frp/frps_start.sh",
    )

    # Run frps_start.sh
    result = ssh_cmd(
        relay.host, relay.ssh_user, relay.ssh_key,
        f"cd /opt/frp && bash frps_start.sh --port {relay.port} --dir /opt/frp",
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Verify
    result = ssh_cmd(
        relay.host, relay.ssh_user, relay.ssh_key,
        "pgrep -f 'frps.*-c'", check=False,
    )
    if result.returncode == 0:
        print(f"frps running on {relay.host}:{relay.port}")
        return 0
    else:
        print(f"ERROR: frps not running on {relay.host}")
        return 1


def cmd_start(cfg: TunnelConfig, args: argparse.Namespace) -> int:
    machine = cfg.get_machine(args.machine)
    if not args.ranks:
        print(f"ERROR: --ranks required (e.g., --ranks '127.0.0.1:31051,127.0.0.1:31052')")
        return 1

    frpc_args = [
        "bash", str(SCRIPT_DIR / "frpc_start.sh"),
        "--server", cfg.relay.host,
        "--server-port", str(cfg.relay.port),
        "--ranks", args.ranks,
        "--base-remote-port", str(machine.base_remote_port),
        "--name", machine.name,
    ]

    if machine.is_local:
        print(f"Starting frpc locally for {machine.name}...")
        result = subprocess.run(frpc_args, capture_output=False)
        return result.returncode
    else:
        print(f"Starting frpc on {machine.ssh_host} for {machine.name}...")
        # Copy frpc_start.sh to remote
        ssh_cmd(machine.ssh_host, machine.ssh_user, machine.ssh_key,
                "mkdir -p /opt/frp")
        scp_to(machine.ssh_host, machine.ssh_user, machine.ssh_key,
               str(SCRIPT_DIR / "frpc_start.sh"), "/opt/frp/frpc_start.sh")
        # Run remotely
        remote_cmd = (
            f"cd /opt/frp && bash frpc_start.sh"
            f" --server {cfg.relay.host}"
            f" --server-port {cfg.relay.port}"
            f" --ranks '{args.ranks}'"
            f" --base-remote-port {machine.base_remote_port}"
            f" --name {machine.name}"
            f" --dir /opt/frp"
        )
        result = ssh_cmd(
            machine.ssh_host, machine.ssh_user, machine.ssh_key,
            remote_cmd, check=False,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result.returncode


def cmd_stop(cfg: TunnelConfig, args: argparse.Namespace) -> int:
    machine = cfg.get_machine(args.machine)

    if machine.is_local:
        print(f"Stopping frpc locally for {machine.name}...")
        subprocess.run(["pkill", "-f", "frpc.*-c"], check=False)
    else:
        print(f"Stopping frpc on {machine.ssh_host} for {machine.name}...")
        ssh_cmd(
            machine.ssh_host, machine.ssh_user, machine.ssh_key,
            "pkill -f 'frpc.*-c'", check=False,
        )
    print("Done.")
    return 0


def cmd_status(cfg: TunnelConfig, args: argparse.Namespace) -> int:
    relay = cfg.relay
    print(f"=== FRP Tunnel Status ===\n")

    # Check frps on relay
    result = ssh_cmd(
        relay.host, relay.ssh_user, relay.ssh_key,
        "pgrep -f 'frps.*-c'", check=False,
    )
    frps_running = result.returncode == 0
    print(f"Relay ({relay.host}:{relay.port}): "
          f"{'RUNNING' if frps_running else 'NOT RUNNING'}")

    # Check each machine
    all_ok = frps_running
    for m in cfg.machines:
        print(f"\n{m.name} (ports {m.port_range[0]}-{m.port_range[1]}):")

        # Check frpc process
        if m.is_local:
            import shutil
            frpc_running = subprocess.run(
                ["pgrep", "-f", "frpc.*-c"], capture_output=True
            ).returncode == 0
        else:
            r = ssh_cmd(m.ssh_host, m.ssh_user, m.ssh_key,
                        "pgrep -f 'frpc.*-c'", check=False)
            frpc_running = r.returncode == 0

        loc = "local" if m.is_local else m.ssh_host
        print(f"  frpc ({loc}): {'RUNNING' if frpc_running else 'NOT RUNNING'}")

        # TCP probe each port
        ok = 0
        for i in range(m.num_ranks):
            port = m.base_remote_port + i
            reachable = tcp_probe(relay.host, port)
            status = "OK" if reachable else "FAIL"
            print(f"  Rank {i} :{port}  [{status}]")
            if reachable:
                ok += 1
            else:
                all_ok = False

        print(f"  {ok}/{m.num_ranks} ranks reachable")

    return 0 if all_ok else 1


def cmd_info(cfg: TunnelConfig, args: argparse.Namespace) -> int:
    machine = cfg.get_machine(args.machine)
    endpoints = cfg.get_relay_endpoints(args.machine)

    print(f"Machine: {machine.name}")
    print(f"Port range: {machine.port_range[0]}-{machine.port_range[1]}")
    print(f"Location: {'local' if machine.is_local else machine.ssh_host}")
    print(f"\nRelay endpoints:")
    for i, ep in enumerate(endpoints):
        print(f"  Rank {i}: {ep}")
    print(f"\nAgent base_url: http://{cfg.relay.host}:<RANK_PORT>/v1/{{SESSION_ID}}")
    return 0


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="FRP tunnel orchestrator for multi-GPU-machine setups"
    )
    parser.add_argument(
        "--config", default=str(SCRIPT_DIR / "tunnel_config.yaml"),
        help="Path to tunnel_config.yaml",
    )

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("validate", help="Check config for port overlaps")
    sub.add_parser("deploy-relay", help="Deploy + start frps on relay")

    p_start = sub.add_parser("start", help="Start frpc on a GPU machine")
    p_start.add_argument("machine", help="Machine name from config")
    p_start.add_argument("--ranks", required=True,
                         help="Comma-separated ip:port pairs for sglang ranks")

    p_stop = sub.add_parser("stop", help="Stop frpc on a GPU machine")
    p_stop.add_argument("machine", help="Machine name from config")

    sub.add_parser("status", help="Check all tunnels")

    p_info = sub.add_parser("info", help="Print relay endpoints for a machine")
    p_info.add_argument("machine", help="Machine name from config")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    cfg = TunnelConfig.load(args.config)

    commands = {
        "validate": cmd_validate,
        "deploy-relay": cmd_deploy_relay,
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "info": cmd_info,
    }
    return commands[args.command](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
