#!/usr/bin/env python3
"""Fan out local-only patches from the head node to every Ray worker.

`/root/miles` and `/root/Megatron-LM` are baked into the container image (no
.git on disk), so any patch we make on the head node lives in the local
overlay filesystem only. After a container rebuild every node ships with the
upstream files and the patches need to be re-applied + replicated.

This helper:
  1. Reads each patched file from the HEAD node's local copy
  2. Submits one Ray task per node (pinned via NodeAffinitySchedulingStrategy)
  3. The task writes the same bytes to the same path on each worker
  4. Confirms via SHA-256 that every node ended up with identical content

Patches covered (see PATCHES.md for descriptions):
  - /root/miles/miles_plugins/models/deepseek_v4/ops/v4_indexer.py
  - /root/Megatron-LM/megatron/core/transformer/experimental_attention_variant/dsa.py

Add new entries to PATCHED_FILES as we accumulate more local patches.

Usage:
    python3 /data/terminal_agent/scripts/miles/fan_out_node_patches.py
"""
from __future__ import annotations

import hashlib
import sys

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


PATCHED_FILES = [
    # Patch 1 — fp8_simulate_qat positional-args fix (see PATCHES.md)
    "/root/miles/miles_plugins/models/deepseek_v4/ops/v4_indexer.py",
    "/root/Megatron-LM/megatron/core/transformer/experimental_attention_variant/dsa.py",
    # Patch 2 — --max-weight-staleness backport (see PATCHES.md)
    "/root/miles/miles/utils/types.py",
    "/root/miles/miles/utils/arguments.py",
    "/root/miles/examples/fully_async/fully_async_rollout.py",
    # Patch 5 — kl-loss coef-0 guard (no NaN grad from 0*kl_loss; see PATCHES.md)
    "/root/miles/miles/backends/training_utils/loss.py",
]


@ray.remote(num_cpus=1)
def _write_files(payloads: list[tuple[str, str]]) -> dict:
    """payloads: list of (target_path, content_string). Writes each on the local node."""
    import hashlib, os, socket
    result = {"host": socket.gethostname(), "files": []}
    for path, content in payloads:
        if not os.path.isdir(os.path.dirname(path)):
            result["files"].append({"path": path, "status": "missing-parent"})
            continue
        with open(path, "w") as f:
            f.write(content)
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]
        result["files"].append({"path": path, "sha": sha})
    return result


def main() -> int:
    # Pre-read local files + their SHAs (the canonical "source of truth")
    payloads: list[tuple[str, str]] = []
    expected_shas: dict[str, str] = {}
    for p in PATCHED_FILES:
        try:
            content = open(p).read()
        except FileNotFoundError:
            print(f"[err] head-node file missing: {p}", file=sys.stderr)
            return 1
        sha = hashlib.sha256(content.encode()).hexdigest()[:12]
        expected_shas[p] = sha
        payloads.append((p, content))
        print(f"[src] {sha}  {len(content):>7d} bytes  {p}")

    ray.init(address="auto")
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    print(f"\n[ray] {len(nodes)} live nodes")

    futures = [
        _write_files.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=n["NodeID"], soft=False
            )
        ).remote(payloads)
        for n in nodes
    ]
    results = ray.get(futures, timeout=60)

    print("\n[fan-out]")
    all_ok = True
    for r in results:
        for entry in r["files"]:
            sha = entry.get("sha")
            ok = sha == expected_shas.get(entry["path"])
            mark = "✓" if ok else "✗"
            print(f"  {mark} {r['host']:32s} {sha or entry.get('status'):14s} {entry['path']}")
            if not ok:
                all_ok = False

    ray.shutdown()
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
