#!/usr/bin/env python3
"""Recovery monitor: tracks env-failure rate + env_service health after the
retries=10 / MAX_SLOTS=192 relaunch. Emits one status line per cycle; ALERTs on
env_service down, sustained high failure, or jammed slots."""
import json
import os
import subprocess
import time
import urllib.request

R = "/root/data/training_runs/camel-no-easy-async-2r7t-20260528-215756"
INTERVAL = 300
CPU_QUOTA = 250
start = time.time()


def daytona_cpu():
    """Total CPU of started sandboxes vs the ~200 account quota (the failure
    driver — when this hits the quota, all creates fail 'Total CPU limit')."""
    try:
        key = os.environ["DAYTONA_API_KEY"]; base = os.environ["DAYTONA_API_URL"]
        out = subprocess.run(
            ["curl", "-s", "--max-time", "15", "-H", "Authorization: Bearer " + key,
             base + "/sandbox?limit=200"], capture_output=True, text=True).stdout
        it = json.loads(out)
        it = it.get("items", it)
        st = [s for s in it if s.get("state") == "started"]
        return len(st), sum(s.get("cpu", 0) for s in st)
    except Exception:
        return None, None


def health():
    try:
        return json.load(urllib.request.urlopen("http://127.0.0.1:8002/health", timeout=5))
    except Exception:
        return None


def failrate():
    fs = subprocess.run(
        ["find", R + "/trials", "-name", "run_info.json", "-mmin", "-6"],
        capture_output=True, text=True,
    ).stdout.split()
    n = f = nr = 0
    for fp in fs:
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        n += 1
        ei = d.get("error_info") or {}
        if ei.get("stage") or d.get("reward") is None:
            f += 1
            if "No available runners" in str(ei.get("error_message", "")):
                nr += 1
    return n, (100 * f // max(n, 1)), (100 * nr // max(n, 1))


while True:
    ts = time.strftime("%H:%M")
    h = health()
    if h is None:
        print(f"{ts} ALERT: env_service DOWN (no /health response)", flush=True)
        time.sleep(60)
        continue
    n, fr, nr = failrate()
    nsb, cpu = daytona_cpu()
    bg = h["build_gate"]
    cpu_s = f"cpu={cpu}/{CPU_QUOTA}" if cpu is not None else "cpu=?"
    line = (f"{ts} ms={h['max_slots']} act={h['active_steps']} "
            f"avail={h['available_slots']} built={bg['built']} {cpu_s} | "
            f"fail={fr}% norunner={nr}% (n={n})")
    elapsed_min = (time.time() - start) / 60
    if elapsed_min > 20 and fr > 40 and n >= 20:
        line += "  <<< ALERT: failure still high after grace"
    if cpu is not None and cpu >= CPU_QUOTA - 12:
        line += f"  <<< ALERT: CPU near quota ({cpu}/{CPU_QUOTA}) — failures imminent, trim MAX_SLOTS"
    if h["available_slots"] == 0 and h["active_steps"] == h["max_slots"]:
        line += "  (slots full)"
    print(line, flush=True)
    time.sleep(INTERVAL)
