#!/usr/bin/env python3
"""Wait until ALL ray nodes can run a task (i.e. node 10.220.51.27 is freed/recycled
and no node is wedged), then fan out patches and launch the 1+7 async training.

Armed because 10.220.51.27 is disk-full and cannot spawn ray workers, which blocks
every multi-node job (miles PACKs bundles across all nodes; no node-exclusion flag).
This does NOT touch the ray cluster — it only probes, then submits a job once healthy.

Logs to /data/training_runs/_relaunch_logs/await_cluster_monitor.log
"""
import subprocess, sys, time
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

MIN_NODES = 8
POLL_SECS = 90
MAX_ITERS = 320  # ~8h

LAUNCH = "bash /data/terminal_agent/scripts/miles/run_deepseek_v4_seta_fully_async.sh"
FANOUT = "python3 /data/terminal_agent/scripts/miles/fan_out_node_patches.py"


@ray.remote(num_cpus=1, max_retries=0)
def _probe():
    # Running at all proves the node can spawn a worker (write its log) => disk has space.
    import socket, tempfile, os
    fd, p = tempfile.mkstemp()
    os.write(fd, b"ok"); os.close(fd); os.remove(p)
    return socket.gethostname()


def healthy_launch_mode():
    """Return (num_nodes_to_launch or None, status_msg). All alive nodes must be probe-able.
    8 healthy -> 1+7 (NUM_NODES=8); exactly 7 healthy (.27 dropped out) -> 1+6 (NUM_NODES=7)."""
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    futs = {
        _probe.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=n["NodeID"], soft=False)).remote(): n["NodeManagerAddress"]
        for n in nodes
    }
    ready, not_ready = ray.wait(list(futs), num_returns=len(futs), timeout=40)
    bad = [futs[f] for f in not_ready]
    for f in ready:
        try:
            ray.get(f)
        except Exception:
            bad.append(futs[f])
    if bad:
        return None, f"wedged: {bad}"
    if len(nodes) >= 8:
        return 8, f"{len(nodes)} healthy -> 1+7"
    if len(nodes) == 7:
        return 7, "7 healthy (.27 dropped) -> 1+6 fallback"
    return None, f"only {len(nodes)} healthy"


def main():
    ray.init(address="auto")
    for i in range(1, MAX_ITERS + 1):
        nnodes, msg = healthy_launch_mode()
        print(f"[{i}/{MAX_ITERS}] launch_nodes={nnodes} ({msg})", flush=True)
        if nnodes:
            print(f"[launch] cluster healthy -> fan out patches + launch (NUM_NODES={nnodes})", flush=True)
            ray.shutdown()
            subprocess.run(FANOUT, shell=True)
            # best-effort daytona cleanup before relaunch (creds from ~/.bashrc)
            subprocess.run(
                "cd /data/terminal_agent && "
                "eval \"$(bash -ic 'for v in DAYTONA_API_KEY DAYTONA_API_URL; do val=\\\"${!v-}\\\"; "
                "[ -n \\\"$val\\\" ] && printf \\\"export %s=%q\\\\n\\\" \\\"$v\\\" \\\"$val\\\"; done' 2>/dev/null)\" && "
                "DELETE=1 CONC=12 venv_cpu/bin/python scripts/miles/daytona_cleanup_ours.py || true",
                shell=True,
            )
            stamp = subprocess.run("date -u +%Y%m%d-%H%M%S", shell=True, capture_output=True, text=True).stdout.strip()
            wlog = f"/data/training_runs/_relaunch_logs/seta_async_auto_{stamp}.log"
            subprocess.run(
                f"tmux kill-session -t seta_async 2>/dev/null; "
                f"tmux new-session -d -s seta_async 'NUM_NODES={nnodes} {LAUNCH} > {wlog} 2>&1'",
                shell=True,
            )
            print(f"[launch] submitted; wrapper log: {wlog}", flush=True)
            return 0
        time.sleep(POLL_SECS)
    print("[timeout] cluster never became fully healthy; .27 still needs infra recovery", flush=True)
    ray.shutdown()
    return 1


if __name__ == "__main__":
    sys.exit(main())
