"""Automates FRP tunnel + scheduler url_rewrite setup after ProxyServer starts.

Called from eval/train scripts after ProxyServer ports are known.
Handles both single-rank and multi-rank (DP) setups.

Usage from eval/train script:
    from seta_env.services.proxy_setup import setup_proxy_tunnels

    # After ProxyServer starts and all_gather:
    setup_proxy_tunnels(
        proxy_addresses=["http://172.17.0.2:48712", "http://172.17.0.2:48713"],
        scheduler_url="http://127.0.0.1:8003",
    )
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

TUNNEL_CONFIG = Path(__file__).parent / "frp_tunnel" / "tunnel_config.yaml"
MANAGE_TUNNEL = Path(__file__).parent / "frp_tunnel" / "manage_tunnel.py"


def setup_proxy_tunnels(
    proxy_addresses: list[str],
    scheduler_url: str = "http://127.0.0.1:8003",
    tunnel_machine: str = "gpu-a",
    use_frp: bool | None = None,
) -> dict[str, str]:
    """Set up FRP tunnels and scheduler url_rewrite for ProxyServer addresses.

    Args:
        proxy_addresses: List of ProxyServer public addresses, one per rank.
            e.g. ["http://172.17.0.2:48712", "http://172.17.0.2:48713"]
        scheduler_url: env_scheduler URL.
        tunnel_machine: Machine name in tunnel_config.yaml.
        use_frp: Whether to use FRP tunnel. None = auto-detect from tunnel_config.yaml.

    Returns:
        url_rewrite dict that was applied.
    """
    # Parse proxy addresses into ip:port pairs
    ranks = []
    for addr in proxy_addresses:
        # "http://172.17.0.2:48712" → "172.17.0.2:48712"
        host_port = addr.replace("http://", "").replace("https://", "").rstrip("/")
        ranks.append(host_port)

    # Auto-detect FRP: use it if tunnel_config.yaml exists and has the machine
    if use_frp is None:
        use_frp = False
        if TUNNEL_CONFIG.exists():
            try:
                import yaml
                cfg = yaml.safe_load(TUNNEL_CONFIG.read_text())
                machines = [m["name"] for m in cfg.get("gpu_machines", [])]
                use_frp = tunnel_machine in machines
            except Exception:
                pass

    url_rewrite = {}

    if use_frp:
        # Read relay info from tunnel_config
        import yaml
        cfg = yaml.safe_load(TUNNEL_CONFIG.read_text())
        relay_host = cfg["relay"]["host"]

        machine = None
        for m in cfg.get("gpu_machines", []):
            if m["name"] == tunnel_machine:
                machine = m
                break

        if machine is None:
            logger.warning("Machine %s not in tunnel_config.yaml, skipping FRP", tunnel_machine)
        else:
            base_port = machine["base_remote_port"]

            # Start frpc with actual ProxyServer ports
            ranks_arg = ",".join(ranks)
            logger.info("Starting FRP tunnel: %s → relay %s:%d+", ranks_arg, relay_host, base_port)

            result = subprocess.run(
                [
                    sys.executable, str(MANAGE_TUNNEL),
                    "--config", str(TUNNEL_CONFIG),
                    "start", tunnel_machine,
                    "--ranks", ranks_arg,
                ],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                logger.error("FRP tunnel start failed: %s", result.stderr)
            else:
                logger.info("FRP tunnel started:\n%s", result.stdout.strip())

            # Build url_rewrite: proxy internal addr → relay addr
            for i, addr in enumerate(proxy_addresses):
                relay_url = f"http://{relay_host}:{base_port + i}"
                url_rewrite[addr] = relay_url

    # Update scheduler url_rewrite
    if url_rewrite:
        logger.info("Updating scheduler url_rewrite: %s", url_rewrite)
        try:
            resp = httpx.post(
                f"{scheduler_url}/url_rewrite",
                json=url_rewrite,
                timeout=10.0,
            )
            resp.raise_for_status()
            logger.info("Scheduler url_rewrite updated")
        except Exception as e:
            logger.error("Failed to update scheduler url_rewrite: %s", e)
    else:
        logger.info("No url_rewrite needed (direct connectivity or no FRP)")

    return url_rewrite
