#!/usr/bin/env python3
"""Prune write-only debug artifacts from a training run dir to reclaim disk.

Removes (none are read by the live trainer; only offline debug tools touch them):
  - trials/**/tito_state.json
  - dump_details/rollout_data/<step>.pt
  - dump_details/train_data/<step>_<rank>.pt

Guards:
  * age  : a file is only eligible if its mtime is older than --age-min (default 30).
  * steps: for rollout_data / train_data, the latest --keep-steps step numbers are
           ALWAYS kept (write-conflict safety for in-flight steps), regardless of age.
           tito_state.json has only the age guard (per-trajectory, no step concept).

Usage:
  python3 cleanup_run_artifacts.py <run_dir> [--age-min 30] [--keep-steps 2] [--dry-run]
"""
import argparse
import os
import re
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="training run directory to clean")
    ap.add_argument("--age-min", type=float, default=30,
                    help="only delete files whose mtime is older than this many minutes")
    ap.add_argument("--keep-steps", type=int, default=2,
                    help="always keep the N highest step numbers of rollout/train data")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be deleted without deleting")
    args = ap.parse_args()

    run = os.path.abspath(args.run_dir)
    if not os.path.isdir(run):
        raise SystemExit(f"not a directory: {run}")

    now = time.time()
    age_cut = args.age_min * 60.0
    stats = {"deleted": 0, "freed": 0, "kept_age": 0, "kept_step": 0}

    def old_enough(path):
        try:
            return (now - os.path.getmtime(path)) > age_cut
        except OSError:
            return False

    def rm(path):
        try:
            sz = os.path.getsize(path)
        except OSError:
            sz = 0
        if not args.dry_run:
            try:
                os.remove(path)
            except OSError as e:
                print(f"  WARN could not remove {path}: {e}")
                return
        stats["deleted"] += 1
        stats["freed"] += sz

    # ── 1) tito_state.json (age guard only) ──────────────────────────────────
    tito = 0
    trials = os.path.join(run, "trials")
    for dirpath, _dirs, files in os.walk(trials):
        for f in files:
            if f == "tito_state.json":
                p = os.path.join(dirpath, f)
                if old_enough(p):
                    rm(p); tito += 1
                else:
                    stats["kept_age"] += 1

    # ── helper: list (name, step) and the latest-N step numbers in a dir ─────
    def scan_steps(dirpath, step_re):
        files, steps = [], set()
        try:
            names = os.listdir(dirpath)
        except OSError:
            return [], set()
        for n in names:
            m = step_re.match(n)
            if m:
                st = int(m.group(1))
                files.append((n, st))
                steps.add(st)
        keep = set(sorted(steps)[-args.keep_steps:]) if steps else set()
        return files, keep

    def prune_step_dir(dirpath, step_re):
        files, keep = scan_steps(dirpath, step_re)
        n = 0
        for name, st in files:
            p = os.path.join(dirpath, name)
            if st in keep:
                stats["kept_step"] += 1
                continue
            if not old_enough(p):
                stats["kept_age"] += 1
                continue
            rm(p); n += 1
        return n, sorted(keep)

    # ── 2) rollout_data/<step>.pt ────────────────────────────────────────────
    rollout, keep_ro = prune_step_dir(
        os.path.join(run, "dump_details", "rollout_data"), re.compile(r"^(\d+)\.pt$"))

    # ── 3) train_data/<step>_<rank>.pt ───────────────────────────────────────
    train, keep_tr = prune_step_dir(
        os.path.join(run, "dump_details", "train_data"), re.compile(r"^(\d+)_\d+\.pt$"))

    tag = "[DRY-RUN] would remove" if args.dry_run else "removed"
    print(f"run: {run}")
    print(f"{tag}: tito_state={tito}  rollout_data={rollout}  train_data={train}")
    print(f"   total {stats['deleted']} files, {stats['freed'] / 1e9:.1f} GB")
    print(f"kept (latest-{args.keep_steps}-steps guard): "
          f"rollout_steps={keep_ro}  train_steps={keep_tr}")
    print(f"kept (younger than {args.age_min}min): {stats['kept_age']} files; "
          f"kept by step-guard: {stats['kept_step']} files")


if __name__ == "__main__":
    main()
