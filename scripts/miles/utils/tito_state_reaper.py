#!/usr/bin/env python3
"""Continuously reap stale tito_state.json files.

Deletes every file matching the given glob pattern whose mtime is AGE seconds
or older. Active trials keep their state file fresh (mtime < AGE), so only
stale/abandoned ones are removed.

Usage:
    tito_state_reaper.py '<glob-pattern>' [age_sec=300] [interval_sec=60]

The pattern should point at tito_state.json with the per-trial directory
replaced by '*', e.g.:
    /root/data/training_runs/<RUN>/trials/<RUN>/*/tito_state.json
"""
import glob
import os
import sys
import time


def main():
    if len(sys.argv) < 2:
        sys.stderr.write(
            "usage: tito_state_reaper.py '<glob-pattern>' [age_sec] [interval_sec]\n"
        )
        sys.exit(2)
    pattern = sys.argv[1]
    age = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    interval = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0

    print(
        f"[tito-reaper] start pattern={pattern} age>={age:.0f}s "
        f"interval={interval:.0f}s",
        flush=True,
    )
    total = 0
    while True:
        now = time.time()
        deleted = 0
        for f in glob.glob(pattern):
            try:
                if now - os.path.getmtime(f) >= age:
                    os.remove(f)
                    deleted += 1
            except FileNotFoundError:
                pass  # raced with another deleter / trial cleanup
            except OSError as e:
                print(f"[tito-reaper] WARN {f}: {e}", flush=True)
        if deleted:
            total += deleted
            print(
                f"[tito-reaper] {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"reaped {deleted} (total {total})",
                flush=True,
            )
        time.sleep(interval)


if __name__ == "__main__":
    main()
