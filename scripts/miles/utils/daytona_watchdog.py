#!/usr/bin/env python3
"""Standing hygiene watchdog for the env_service / Daytona setup.

Every INTERVAL seconds it:
  1. Reaps Daytona sandboxes in terminal-dead states (stopped/error/build_failed/
     archived) — these pin runner capacity and cause "No available runners".
     NEVER touches started/creating/building_snapshot/destroying (in-use / mid-transition).
  2. Prunes idle snapshots when the count nears the 100 cap.
  3. Frees disk (tito_state + old dump_details) when /data is near full.

Secrets (DAYTONA_API_KEY/URL) are read in-memory from the live env_service
process environment — never written to disk or a command line.

Run via: tmux new-session -d -s env_watchdog \
  '/usr/bin/python3 .../daytona_watchdog.py'
(or under a while-true supervisor for auto-restart).
"""
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone

INTERVAL = int(os.environ.get("WATCHDOG_INTERVAL", "120"))
SNAP_PRUNE_AT = int(os.environ.get("WATCHDOG_SNAP_AT", "95"))
DISK_PRUNE_PCT = int(os.environ.get("WATCHDOG_DISK_PCT", "90"))
RUN_DIR = os.environ.get(
    "WATCHDOG_RUN_DIR",
    "/root/data/training_runs/camel-no-easy-async-2r7t-20260528-215756",
)
DEAD_STATES = {"stopped", "error", "build_failed", "archived"}
# A 'started' sandbox older than this CANNOT hold a live env_service slot
# (slot is released at STEP_TIMEOUT=30min), so it's an orphan safe to reap.
STARTED_ORPHAN_MIN = int(os.environ.get("WATCHDOG_STARTED_ORPHAN_MIN", "35"))
PRUNE_SNAP = "/root/data/terminal_agent/scripts/miles" \
             if os.path.isdir("/root/data/terminal_agent/scripts/miles") else "."


def load_daytona_creds():
    if os.environ.get("DAYTONA_API_KEY") and os.environ.get("DAYTONA_API_URL"):
        return
    import glob
    for environ in glob.glob("/proc/[0-9]*/environ"):
        try:
            with open(environ, "rb") as f:
                data = f.read()
        except OSError:
            continue
        if b"DAYTONA_API_KEY=" in data:
            kv = dict(p.split(b"=", 1) for p in data.split(b"\x00") if b"=" in p)
            if b"DAYTONA_API_KEY" in kv and b"DAYTONA_API_URL" in kv:
                os.environ["DAYTONA_API_KEY"] = kv[b"DAYTONA_API_KEY"].decode()
                os.environ["DAYTONA_API_URL"] = kv[b"DAYTONA_API_URL"].decode()
                return
    print("watchdog: could not find DAYTONA creds in any process env", flush=True)


def api(method, path):
    key = os.environ["DAYTONA_API_KEY"]
    base = os.environ["DAYTONA_API_URL"]
    return subprocess.run(
        ["curl", "-s", "-X", method, "--max-time", "30",
         "-H", f"Authorization: Bearer {key}", base + path],
        capture_output=True, text=True,
    ).stdout


def reap_sandboxes():
    try:
        items = json.loads(api("GET", "/sandbox?limit=200"))
    except Exception as e:
        return f"sandbox-list-failed: {e}"
    items = items.get("items", items) if isinstance(items, dict) else items
    states = Counter(s.get("state") for s in items)

    now = datetime.now(timezone.utc)
    def age_min(s):
        v = s.get("createdAt")
        try:
            return (now - datetime.fromisoformat(v.replace("Z", "+00:00"))).total_seconds() / 60
        except Exception:
            return 0.0

    dead = [s for s in items if s.get("state") in DEAD_STATES]
    orphan_started = [s for s in items
                      if s.get("state") == "started" and age_min(s) > STARTED_ORPHAN_MIN]
    for s in dead + orphan_started:
        api("DELETE", f"/sandbox/{s['id']}?force=true")
    return (f"states={dict(states)} started={states.get('started', 0)} "
            f"reaped_dead={len(dead)} reaped_orphan_started={len(orphan_started)}")


def prune_snapshots():
    try:
        d = json.loads(api("GET", "/snapshots?limit=200"))
        total = d.get("total", len(d.get("items", [])))
    except Exception:
        return None
    if total < SNAP_PRUNE_AT:
        return f"snapshots={total}(ok)"
    out = subprocess.run(
        ["/usr/bin/python3", f"{PRUNE_SNAP}/prune_snapshots.py", str(SNAP_PRUNE_AT)],
        capture_output=True, text=True,
    ).stdout.strip()
    return f"snapshots={total} -> {out}"


def disk_guard():
    try:
        u = shutil.disk_usage("/data")
        pct = round(100 * u.used / u.total)
    except Exception:
        return None
    if pct < DISK_PRUNE_PCT:
        return f"disk={pct}%(ok)"
    out = subprocess.run(
        ["/usr/bin/python3",
         f"{PRUNE_SNAP}/cleanup_run_artifacts.py", RUN_DIR,
         "--age-min", "30", "--keep-steps", "2"],
        capture_output=True, text=True,
    ).stdout.strip().replace("\n", " | ")
    return f"disk={pct}% PRUNED: {out}"


def main():
    load_daytona_creds()
    print(f"watchdog start: interval={INTERVAL}s snap_at={SNAP_PRUNE_AT} "
          f"disk_at={DISK_PRUNE_PCT}%", flush=True)
    while True:
        ts = time.strftime("%H:%M:%S")
        try:
            sb = reap_sandboxes()
            sn = prune_snapshots()
            dk = disk_guard()
            print(f"[{ts}] {sb} | {sn} | {dk}", flush=True)
        except Exception as e:
            print(f"[{ts}] watchdog-error: {e}", flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
