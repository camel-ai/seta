"""Filter dataset tasks based on prior evaluation results.

Reads an ``evaluated_tasks.csv`` (columns ``task_id, traj_0, ..., traj_N``)
and writes a ``task_filter.txt`` to the dataset folder containing one
``task_id`` per line — the set of tasks that should be KEPT during
training/eval. ``load_harbor_dataset`` auto-detects this file and skips
everything not listed.

Filter rules (combinable):
    --drop-missing      drop tasks not present in (or empty in) the CSV
    --drop-too-hard     drop tasks where max(reward) == 0.0 across trajs
    --drop-too-easy     drop tasks where min(reward) == 1.0 across trajs

Or, for a delta-evaluation workflow:
    --only-missing      emit ONLY tasks present in the dataset but absent
                        from (or empty in) the CSV(s). The other --drop-*
                        flags are ignored in this mode.

Pass --csv multiple times to union several CSVs (e.g. a pre-computed
baseline plus a new partial run).

Usage:
    python -m seta_env.dataset.filter_tasks \\
        --csv dataset/evaluated_tasks.csv \\
        --dataset dataset/seta-env-v2 \\
        --drop-missing --drop-too-hard --drop-too-easy
"""

import argparse
import csv
from pathlib import Path


def _parse_csv(csv_path: Path) -> dict[str, list[float]]:
    rewards: dict[str, list[float]] = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        traj_cols = [c for c in fields if c.startswith("traj_")]
        if "task_id" not in fields or not traj_cols:
            raise ValueError(
                f"{csv_path} must have a 'task_id' column and one or more "
                f"'traj_*' columns; got {fields}"
            )
        for row in reader:
            tid = (row.get("task_id") or "").strip()
            if not tid:
                continue
            vals: list[float] = []
            for c in traj_cols:
                v = (row.get(c) or "").strip()
                if not v:
                    continue
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
            rewards[tid] = vals
    return rewards


def filter_tasks(
    csv_paths: list[Path],
    dataset_dir: Path,
    drop_missing: bool,
    drop_too_hard: bool,
    drop_too_easy: bool,
    only_missing: bool = False,
) -> list[str]:
    # Union the rewards from all CSVs (later files extend, not overwrite,
    # the per-task trajectory list — they are independent samples).
    rewards: dict[str, list[float]] = {}
    for cp in csv_paths:
        for tid, vals in _parse_csv(cp).items():
            rewards.setdefault(tid, []).extend(vals)

    dataset_task_ids = sorted(
        p.name for p in dataset_dir.iterdir() if p.is_dir()
    )

    if only_missing:
        kept = [tid for tid in dataset_task_ids if not rewards.get(tid)]
        print(
            f"[filter_tasks] dataset={len(dataset_task_ids)} "
            f"only_missing={len(kept)}"
        )
        return kept

    kept: list[str] = []
    n_missing = n_hard = n_easy = 0
    for tid in dataset_task_ids:
        vals = rewards.get(tid)
        if not vals:  # missing or all-blank row
            if drop_missing:
                n_missing += 1
                continue
            kept.append(tid)
            continue
        if drop_too_hard and max(vals) == 0.0:
            n_hard += 1
            continue
        if drop_too_easy and min(vals) == 1.0:
            n_easy += 1
            continue
        kept.append(tid)

    print(
        f"[filter_tasks] dataset={len(dataset_task_ids)} kept={len(kept)} "
        f"dropped: missing={n_missing} too_hard={n_hard} too_easy={n_easy}"
    )
    return kept


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", required=True, type=Path, action="append",
                   help="Path to evaluated_tasks.csv (task_id + traj_* "
                        "columns). Pass multiple times to union several CSVs.")
    p.add_argument("--dataset", required=True, type=Path,
                   help="Dataset directory containing one subdir per task.")
    p.add_argument("--drop-missing", action="store_true",
                   help="Drop tasks not present in the CSV.")
    p.add_argument("--drop-too-hard", action="store_true",
                   help="Drop tasks where max(reward) == 0.0.")
    p.add_argument("--drop-too-easy", action="store_true",
                   help="Drop tasks where min(reward) == 1.0.")
    p.add_argument("--only-missing", action="store_true",
                   help="Emit ONLY tasks present in the dataset but absent "
                        "from the CSV(s). Useful for delta evaluation.")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output path. Defaults to <dataset>/task_filter.txt")
    args = p.parse_args()

    if not args.dataset.is_dir():
        p.error(f"--dataset is not a directory: {args.dataset}")
    for cp in args.csv:
        if not cp.is_file():
            p.error(f"--csv is not a file: {cp}")

    out = args.output or (args.dataset / "task_filter.txt")
    kept = filter_tasks(
        args.csv, args.dataset,
        args.drop_missing, args.drop_too_hard, args.drop_too_easy,
        only_missing=args.only_missing,
    )
    out.write_text("\n".join(kept) + ("\n" if kept else ""))
    print(f"[filter_tasks] wrote {len(kept)} task ids → {out}")


if __name__ == "__main__":
    main()
