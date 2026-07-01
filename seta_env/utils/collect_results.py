"""Collect and summarize evaluation results from a trial root directory.

Reads ``verifier/ctrf.json`` for reward (passed/total tests) and
``run_info.json`` for metadata, producing results.csv and summary.json
in the same format as ``results.collect_and_summarize``.

Usage:
    # Single trial root → results.csv + summary.json (existing behavior)
    python -m seta_env.utils.collect_results /path/to/trial_root
    python -m seta_env.utils.collect_results /path/to/trial_root --output /path/to/output_dir

    # Merge multiple eval output dirs (each containing trials/ and failed/)
    # into success.csv (per-task per-trajectory rewards) + failed.csv (fully-failed tasks)
    python -m seta_env.utils.collect_results --merge \\
        /path/to/eval_run \\
        /path/to/eval_run_resume \\
        /path/to/eval_run_resume_2 \\
        --output /path/to/merged_out

    # Same as above, but ALSO move every trial subdir and failed/*.txt into
    # the output dir (originals become hollow). Use --collect-trials copy
    # for a non-destructive variant.
    python -m seta_env.utils.collect_results --merge \\
        /path/to/eval_run \\
        /path/to/eval_run_resume \\
        --output /path/to/merged_out \\
        --collect-trials move
"""

import argparse
import csv
import json
import logging
import statistics
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-trial data extraction
# ---------------------------------------------------------------------------

def _read_reward_from_ctrf(trial_dir: Path) -> float | None:
    """Parse reward from verifier/ctrf.json as passed/total tests."""
    ctrf_path = trial_dir / "verifier" / "ctrf.json"
    if not ctrf_path.exists():
        return None
    try:
        ctrf = json.loads(ctrf_path.read_text())
        summary = ctrf["results"]["summary"]
        total = summary["tests"]
        if total == 0:
            return 0.0
        return summary["passed"] / total
    except (json.JSONDecodeError, KeyError, OSError) as e:
        logger.warning("Failed to read reward from %s: %s", ctrf_path, e)
        return None


def collect_trial(trial_dir: Path) -> dict | None:
    """Read verifier/ctrf.json (for reward) and run_info.json (for metadata)
    from a trial directory and return a CSV row dict.

    Returns None if neither file exists.
    """
    run_info_path = trial_dir / "run_info.json"
    ctrf_path = trial_dir / "verifier" / "ctrf.json"

    if not run_info_path.exists() and not ctrf_path.exists():
        return None

    # -- run_info (metadata) --------------------------------------------------
    run_info: dict = {}
    if run_info_path.exists():
        try:
            run_info = json.loads(run_info_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read %s: %s", run_info_path, e)

    summary = run_info.get("agent_summary") or {}
    error_info = run_info.get("error_info") or {}

    # -- reward from ctrf.json ------------------------------------------------
    reward = _read_reward_from_ctrf(trial_dir)

    return {
        "task_name":              run_info.get("task_name"),
        "traj_i":                 run_info.get("traj_i", 0),
        "uid":                    run_info.get("uid", ""),
        "reward":                 reward,
        "error":                  bool(error_info),
        "error_stage":            error_info.get("stage", ""),
        "error_message":          error_info.get("error_message", ""),
        "iteration_count":        summary.get("iteration_count"),
        "termination_reason":     summary.get("termination_reason"),
        "total_tool_calls":       summary.get("total_tool_calls"),
        "max_parallel_tool_call": summary.get("max_parallel_tool_call"),
        "parse_error_count":      summary.get("parse_error_count"),
        "prompt_tokens":          summary.get("prompt_tokens"),
        "completion_tokens":      summary.get("completion_tokens"),
        "total_tokens":           summary.get("total_tokens"),
    }


def collect_from_disk(trial_root: str | Path) -> list[dict]:
    """Walk trial_root and collect results from all trial directories
    that contain a ``run_info.json`` or ``verifier/ctrf.json`` file.

    Skips ``_build_*`` directories (used for image building, not actual trials).
    """
    trial_root = Path(trial_root)
    if not trial_root.is_dir():
        raise FileNotFoundError(f"Trial root not found: {trial_root}")

    rows = []
    for child in sorted(trial_root.iterdir()):
        if not child.is_dir() or child.name.startswith("_build_"):
            continue
        row = collect_trial(child)
        if row is not None:
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# evaluated_tasks.csv (wide format consumed by seta_env.dataset.filter_tasks)
# ---------------------------------------------------------------------------

def _write_evaluated_tasks_csv(
    task_traj_rewards: dict[str, dict[int, float]],
    output_path: Path,
) -> None:
    """Write the wide-format ``evaluated_tasks.csv`` consumed by filter_tasks.

    Columns: ``task_id, traj_0, traj_1, ..., traj_{N-1}`` where N is one
    more than the largest trajectory index seen. Missing trajectories are
    left blank.
    """
    max_traj_idx = -1
    for trajs in task_traj_rewards.values():
        if trajs:
            max_traj_idx = max(max_traj_idx, max(trajs.keys()))
    n_trajs = max_traj_idx + 1 if max_traj_idx >= 0 else 0

    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["task_id"] + [f"traj_{i}" for i in range(n_trajs)])
        for task in sorted(task_traj_rewards.keys()):
            trajs = task_traj_rewards[task]
            writer.writerow(
                [task] + [
                    f"{trajs[i]:.6g}" if i in trajs else ""
                    for i in range(n_trajs)
                ]
            )


def _rows_to_task_traj_rewards(rows: list[dict]) -> dict[str, dict[int, float]]:
    """Pivot per-trajectory rows into ``{task: {traj_i: reward}}``."""
    out: dict[str, dict[int, float]] = {}
    for r in rows:
        task = r.get("task_name")
        reward = r.get("reward")
        if not task or reward is None:
            continue
        try:
            traj_i = int(r.get("traj_i") or 0)
        except (TypeError, ValueError):
            traj_i = 0
        out.setdefault(task, {})[traj_i] = reward
    return out


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(rows: list[dict], trial_name: str = "") -> dict:
    """Compute aggregate summary from collected rows.

    Returns a summary dict compatible with ``results.collect_and_summarize``.
    """
    valid_rewards = [r["reward"] for r in rows if r["reward"] is not None]
    n_total = len(rows)
    n_errors = sum(1 for r in rows if r["error"])
    n_pass = sum(1 for r in rows if r["reward"] == 1.0)

    return {
        "trial_name":  trial_name,
        "total_trajs": n_total,
        "error_count": n_errors,
        "pass_count":  n_pass,
        "pass_ratio":  round(n_pass / n_total, 4) if n_total else 0.0,
        "reward": {
            "mean": round(statistics.mean(valid_rewards), 4) if valid_rewards else None,
            "std":  round(statistics.stdev(valid_rewards), 4) if len(valid_rewards) > 1 else 0.0,
            "min":  min(valid_rewards) if valid_rewards else None,
            "max":  max(valid_rewards) if valid_rewards else None,
        },
    }


def print_summary(summary: dict) -> None:
    """Print a formatted evaluation summary to stdout."""
    r = summary["reward"]
    mean_str = f"{r['mean']:.4f}" if r["mean"] is not None else "N/A"
    std_str  = f"{r['std']:.4f}"  if r["std"]  is not None else "N/A"
    min_str  = f"{r['min']:.4f}"  if r["min"]  is not None else "N/A"
    max_str  = f"{r['max']:.4f}"  if r["max"]  is not None else "N/A"
    ratio = summary["pass_ratio"] * 100

    print()
    print("=" * 64)
    print("  EVAL SUMMARY")
    print("=" * 64)
    print(f"  Trial:         {summary['trial_name']}")
    print(f"  Trajectories:  {summary['total_trajs']}")
    print(f"  Pass (r=1.0):  {summary['pass_count']} / {summary['total_trajs']}"
          f"  ({ratio:.1f}%)")
    print(f"  Mean reward:   {mean_str}  ±  {std_str}"
          f"  [min={min_str}  max={max_str}]")
    print(f"  Errors:        {summary['error_count']}")
    print("=" * 64)
    print()


def collect_and_summarize_from_disk(
    trial_root: str | Path,
    output_dir: str | Path | None = None,
) -> dict:
    """Collect results from a trial root on disk, write CSV + summary, return summary.

    Args:
        trial_root: Directory containing trial subdirectories (e.g.
                     ``<experiment>/trials/``).
        output_dir: Where to write ``results.csv`` and ``summary.json``.
                     Defaults to the parent of trial_root.
    """
    trial_root = Path(trial_root)
    if output_dir is None:
        output_dir = trial_root.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_from_disk(trial_root)
    if not rows:
        logger.warning("No valid trial directories found in %s", trial_root)
        return {}

    trial_name = trial_root.parent.name
    summary = summarize(rows, trial_name=trial_name)

    # Write summary.json
    with open(output_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # Write results.csv
    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d rows to %s", len(rows), csv_path)
    logger.info("Summary written to %s", output_dir / "summary.json")

    # Also emit the wide-format file consumed by seta_env.dataset.filter_tasks
    evaluated_path = output_dir / "evaluated_tasks.csv"
    _write_evaluated_tasks_csv(_rows_to_task_traj_rewards(rows), evaluated_path)
    logger.info("Wrote evaluated_tasks.csv to %s", evaluated_path)

    print_summary(summary)
    return summary


# ---------------------------------------------------------------------------
# Multi-dir merge: success.csv + failed.csv
# ---------------------------------------------------------------------------

import re as _re

# Matches the final exception line of a Python traceback at column 0:
#   "RuntimeError: Docker compose command failed ..."
#   "asyncio.exceptions.TimeoutError: ..."
#   "MyPkg.errors.MyError"  (no colon — bare name)
_EXC_LINE_RE = _re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
    r"(?:Error|Exception|Warning|Interrupt|Exit|Timeout))(?::\s*(.*))?$"
)


def _extract_fail_reason(text: str) -> str:
    """Pull the most informative one-line summary from a Python traceback.

    Strategy:
      1. Find the LAST line at column 0 matching a Python exception pattern
         (e.g. ``RuntimeError: ...``, ``asyncio.TimeoutError``).
      2. The message portion may continue across following lines until the
         next traceback section or EOF — concatenate them.
      3. Trim and squash whitespace for CSV friendliness.
    """
    if not text:
        return ""

    def _trim(s: str, n: int = 500) -> str:
        s = " ".join(s.split())
        return s if len(s) <= n else s[: n - 3] + "..."

    lines = text.splitlines()

    # Find last column-0 exception line
    last_idx = -1
    for i, ln in enumerate(lines):
        if not ln or ln[0] in (" ", "\t"):
            continue
        if _EXC_LINE_RE.match(ln):
            last_idx = i

    if last_idx == -1:
        # No recognizable exception; fall back to last non-empty line
        for ln in reversed(lines):
            s = ln.strip()
            if s:
                return _trim(s)
        return ""

    # Collect the exception line + any continuation lines (indented or
    # otherwise non-empty) until the next blank line block or new traceback.
    pieces = [lines[last_idx].strip()]
    for ln in lines[last_idx + 1:]:
        s = ln.strip()
        if not s:
            # First blank line ends the message
            break
        if s.startswith(("Traceback", "During handling", "The above exception")):
            break
        pieces.append(s)

    return _trim(" ".join(pieces))


def _reason_from_run_info(trial_dir: Path) -> str:
    """Best-effort failure reason from run_info.json when ctrf.json is absent.

    Order of preference:
      1. error_info.error_message (if rollout itself errored mid-flight)
      2. agent_summary.termination_reason (e.g. max_tokens_exceeded)
      3. "no ctrf.json (rollout produced no verifier output)"
    """
    ri = trial_dir / "run_info.json"
    if not ri.exists():
        return "no run_info.json and no ctrf.json"
    try:
        info = json.loads(ri.read_text())
    except (json.JSONDecodeError, OSError):
        return "unreadable run_info.json"
    err = info.get("error_info") or {}
    if err.get("error_message"):
        stage = err.get("stage", "")
        msg = err["error_message"]
        return f"{stage}: {msg}" if stage else msg
    summary = info.get("agent_summary") or {}
    term = summary.get("termination_reason")
    if term:
        return f"no ctrf.json (termination: {term})"
    return "no ctrf.json (no termination reason)"


def _consolidate_trial_artifacts(
    trial_parent_dirs: list[Path],
    output_dir: Path,
    mode: str,
) -> dict:
    """Move or copy every trial subdir + failed/*.txt into a single output dir.

    For each input dir, walks its ``trials/`` and ``failed/`` and places the
    contents under ``<output_dir>/trials/`` and ``<output_dir>/failed/``.

    - Trial dir names are ``<task>_t<i>_<hash>`` so cross-dir collisions are
      vanishingly rare in practice. If one happens, it is skipped with a
      warning (the existing entry wins).
    - ``failed/<task>.txt`` files DO collide on task name across resumes; the
      later input dir wins (matching the merge policy for fail reasons).
    - ``failed/build_*`` files are skipped — they're docker-build artifacts,
      not task-rollout failures.
    - ``mode`` is one of ``"move"`` or ``"copy"``.
    """
    import shutil

    if mode not in ("move", "copy"):
        raise ValueError(f"mode must be 'move' or 'copy', got {mode!r}")

    out_trials = output_dir / "trials"
    out_failed = output_dir / "failed"
    out_trials.mkdir(parents=True, exist_ok=True)
    out_failed.mkdir(parents=True, exist_ok=True)

    stats = {"trials_moved": 0, "trials_skipped": 0,
             "failed_moved": 0, "failed_overwritten": 0}

    def _transfer_dir(src: Path, dst: Path) -> None:
        if mode == "move":
            shutil.move(str(src), str(dst))
        else:
            shutil.copytree(str(src), str(dst))

    def _transfer_file(src: Path, dst: Path) -> None:
        if mode == "move":
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))

    for d in trial_parent_dirs:
        d = Path(d)
        if not d.is_dir():
            continue

        # Skip if a parent dir is the output itself (e.g. user passed the
        # output dir as one of the inputs by mistake) — would self-consume.
        try:
            if d.resolve() == output_dir.resolve():
                logger.warning(
                    "Skipping %s as input: it is the same as the output dir", d
                )
                continue
        except OSError:
            pass

        # 1. trials/
        src_trials = d / "trials"
        if src_trials.is_dir():
            for child in sorted(src_trials.iterdir()):
                if not child.is_dir() or child.name.startswith("_build"):
                    continue
                dst = out_trials / child.name
                if dst.exists():
                    stats["trials_skipped"] += 1
                    logger.debug(
                        "Skipping %s/%s — already in output", d.name, child.name
                    )
                    continue
                try:
                    _transfer_dir(child, dst)
                    stats["trials_moved"] += 1
                except Exception as e:
                    logger.error("Failed to %s %s → %s: %s", mode, child, dst, e)

        # 2. failed/
        src_failed = d / "failed"
        if src_failed.is_dir():
            for f in sorted(src_failed.iterdir()):
                if not f.is_file() or not f.name.endswith(".txt"):
                    continue
                if f.name.startswith("build_"):
                    continue
                dst = out_failed / f.name
                if dst.exists():
                    # Later-wins: the most recent input dir overwrites
                    try:
                        dst.unlink()
                    except OSError:
                        pass
                    stats["failed_overwritten"] += 1
                try:
                    _transfer_file(f, dst)
                    stats["failed_moved"] += 1
                except Exception as e:
                    logger.error("Failed to %s %s → %s: %s", mode, f, dst, e)

    return stats


def merge_trial_dirs(
    trial_parent_dirs: list[str | Path],
    output_dir: str | Path,
    consolidate: str | None = None,
) -> tuple[Path, Path]:
    """Merge multiple eval output dirs into success.csv + failed.csv.

    Each input dir is the parent of ``trials/`` and ``failed/`` (i.e. the dir
    that ``eval.py`` writes to, NOT the inner ``trials/`` subdir).

    Definitions:
        - "successful" task: at least one trajectory across all merged dirs
          has a ``verifier/ctrf.json`` file (and thus a reward).
        - "failed" task: NO trajectory across any merged dir has a ctrf.json,
          AND a traceback exists in at least one ``failed/<task>.txt``.

    Output:
        - success.csv: columns = task_id, traj_0, traj_1, ..., traj_{N-1}
          where N = max(traj_i)+1 across all observed trajectories. Missing
          trajectories are left blank.
        - failed.csv: columns = task_id, fail_reason

    If ``consolidate`` is ``"move"`` or ``"copy"``, every trial subdir from
    each input ``trials/`` and every failed/*.txt is also moved/copied into
    ``<output_dir>/trials/`` and ``<output_dir>/failed/`` after the CSVs are
    written. ``"move"`` is destructive — the original input dirs become
    hollow once consolidation succeeds.

    Later dirs win on duplicate (task, traj_i) pairs — useful when a resume
    re-ran a previously errored trajectory and now has a real ctrf.json.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # task_id -> {traj_i: reward}  (only trajs with ctrf.json)
    task_traj_rewards: dict[str, dict[int, float]] = {}
    # task_id -> reason (latest-dir-wins; derived from failed/*.txt OR run_info.json)
    task_fail_reason: dict[str, str] = {}

    n_dirs_seen = 0
    for d in trial_parent_dirs:
        d = Path(d)
        if not d.is_dir():
            logger.warning("Skipping missing dir: %s", d)
            continue
        n_dirs_seen += 1

        trials_root = d / "trials"
        if trials_root.is_dir():
            for child in sorted(trials_root.iterdir()):
                if not child.is_dir() or child.name.startswith("_build"):
                    continue

                # Extract task_name from the dir name "<task>_t<i>_<hash>"
                m = _re.match(r"^(.+)_t\d+_[0-9a-f]+$", child.name)
                task_from_name = m.group(1) if m else None

                row = collect_trial(child)
                if row is None:
                    continue
                task = row.get("task_name") or task_from_name
                if not task:
                    continue
                reward = row.get("reward")
                traj_i = int(row.get("traj_i") or 0)

                if reward is not None:
                    # Has ctrf.json → successful trajectory
                    task_traj_rewards.setdefault(task, {})[traj_i] = reward
                else:
                    # No ctrf.json → derive a reason from run_info.json so the
                    # task can be reported as failed if all its trajs are like this.
                    reason = _reason_from_run_info(child)
                    if reason:
                        task_fail_reason[task] = reason

        failed_root = d / "failed"
        if failed_root.is_dir():
            for f in sorted(failed_root.iterdir()):
                if not f.is_file() or not f.name.endswith(".txt"):
                    continue
                if f.name.startswith("build_"):
                    # build_* are image-build artifacts, not task-rollout failures
                    continue
                task = f.stem
                try:
                    text = f.read_text(errors="replace")
                except OSError as e:
                    logger.warning("Could not read %s: %s", f, e)
                    continue
                # Tracebacks from failed/ are the most informative — they win
                # over run_info-derived reasons.
                task_fail_reason[task] = _extract_fail_reason(text)

    if n_dirs_seen == 0:
        raise FileNotFoundError("None of the provided trial dirs exist.")

    # Determine trajectory column count
    max_traj_idx = -1
    for trajs in task_traj_rewards.values():
        if trajs:
            max_traj_idx = max(max_traj_idx, max(trajs.keys()))
    n_trajs = max_traj_idx + 1 if max_traj_idx >= 0 else 0

    successful_tasks = sorted(task_traj_rewards.keys())
    # Failed = had a traceback AND never produced any ctrf.json
    failed_tasks = sorted(t for t in task_fail_reason if t not in task_traj_rewards)

    success_csv = output_dir / "success.csv"
    failed_csv = output_dir / "failed.csv"

    with open(success_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        header = ["task_id"] + [f"traj_{i}" for i in range(n_trajs)]
        writer.writerow(header)
        for task in successful_tasks:
            trajs = task_traj_rewards[task]
            row = [task] + [
                f"{trajs[i]:.6g}" if i in trajs else "" for i in range(n_trajs)
            ]
            writer.writerow(row)

    # Same content under the canonical filename consumed by filter_tasks.
    _write_evaluated_tasks_csv(task_traj_rewards, output_dir / "evaluated_tasks.csv")

    with open(failed_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["task_id", "fail_reason"])
        for task in failed_tasks:
            writer.writerow([task, task_fail_reason[task]])

    print()
    print("=" * 64)
    print("  MERGE SUMMARY")
    print("=" * 64)
    print(f"  Merged dirs:       {n_dirs_seen}")
    print(f"  Successful tasks:  {len(successful_tasks)}  (≥1 ctrf.json)")
    print(f"  Failed tasks:      {len(failed_tasks)}      (all trajs missing ctrf.json)")
    print(f"  Trajectory cols:   {n_trajs}")
    print(f"  → {success_csv}")
    print(f"  → {output_dir / 'evaluated_tasks.csv'}")
    print(f"  → {failed_csv}")
    print("=" * 64)
    print()

    if consolidate:
        input_paths = [Path(d) for d in trial_parent_dirs]
        print(f"Consolidating trial artifacts ({consolidate})...")
        stats = _consolidate_trial_artifacts(input_paths, output_dir, consolidate)
        print()
        print("=" * 64)
        print(f"  CONSOLIDATION SUMMARY ({consolidate})")
        print("=" * 64)
        print(f"  trials/ moved:           {stats['trials_moved']}")
        print(f"  trials/ skipped (dup):   {stats['trials_skipped']}")
        print(f"  failed/ moved:           {stats['failed_moved']}")
        print(f"  failed/ overwritten:     {stats['failed_overwritten']}")
        print(f"  → {output_dir / 'trials'}")
        print(f"  → {output_dir / 'failed'}")
        print("=" * 64)
        print()

    return success_csv, failed_csv


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect and summarize evaluation results from a trial root directory.",
    )
    parser.add_argument(
        "trial_root",
        nargs="?",
        default=None,
        help="Path to the directory containing trial subdirectories. "
             "Required unless --merge is used.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory. For single-trial mode, defaults to parent of trial_root. "
             "For --merge mode, defaults to the first merged dir.",
    )
    parser.add_argument(
        "--merge",
        nargs="+",
        default=None,
        metavar="EVAL_DIR",
        help="Merge multiple eval output dirs (each containing trials/ and failed/) "
             "into success.csv (per-task per-trajectory rewards) and failed.csv "
             "(tasks where every trajectory is missing a ctrf.json).",
    )
    parser.add_argument(
        "--collect-trials",
        choices=("move", "copy"),
        default=None,
        help="In --merge mode, also consolidate every trial subdir and "
             "failed/*.txt from each input into the output dir's trials/ "
             "and failed/. 'move' is destructive (originals become hollow); "
             "'copy' is non-destructive but doubles disk usage.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.merge:
        out = Path(args.output) if args.output else Path(args.merge[0])
        merge_trial_dirs(args.merge, out, consolidate=args.collect_trials)
        return

    if not args.trial_root:
        parser.error("trial_root is required unless --merge is used")
    collect_and_summarize_from_disk(args.trial_root, args.output)


if __name__ == "__main__":
    main()
