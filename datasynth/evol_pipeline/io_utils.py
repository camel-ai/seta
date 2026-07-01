"""I/O utilities for the evolution pipeline.

Handles Harbor task loading/validation, synth_info.json read/write,
variant ID generation, rollout gap-finding, and summary CSV.
Fully standalone — no imports from other pipelines.
"""

import csv
import json
import logging
import os
import pathlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Harbor task format
# ---------------------------------------------------------------------------

_ESSENTIAL_HARBOR_FILES = [
    "task.toml",
    "instruction.md",
    "environment/Dockerfile",
    "solution/solve.sh",
]


def is_harbor_task_complete(task_path: str) -> bool:
    """Return True if a folder has all required Harbor task files."""
    base = pathlib.Path(task_path)
    return (
        all((base / p).exists() for p in _ESSENTIAL_HARBOR_FILES)
        and (base / "tests").is_dir()
    )


def load_harbor_files(task_path: pathlib.Path) -> Dict[str, str]:
    """Load all text files from a Harbor task directory.

    Validates that essential files exist, then recursively reads every file
    under *task_path* and returns ``{relative_path: content}``.

    All paths are resolved to absolute to avoid ambiguity.

    Raises:
        FileNotFoundError: if the directory or an essential file is missing.
    """
    task_path = task_path.resolve()
    if not task_path.exists():
        raise FileNotFoundError(f"Task path does not exist: {task_path}")

    for ef in _ESSENTIAL_HARBOR_FILES:
        if not (task_path / ef).exists():
            raise FileNotFoundError(
                f"Essential Harbor file {ef} missing in {task_path}"
            )

    if not (task_path / "tests").is_dir():
        raise FileNotFoundError(f"Required 'tests' directory missing in {task_path}")

    files: Dict[str, str] = {}
    for item in task_path.rglob("*"):
        if item.is_file():
            rel = item.relative_to(task_path).as_posix()
            try:
                files[rel] = item.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                print(f"Warning: failed to read {item}: {e}")
    return files


# ---------------------------------------------------------------------------
# Variant ID helpers
# ---------------------------------------------------------------------------

_STRATEGY_PREFIX = {
    "depth": "d",
    "breadth": "b",
}


def generate_variant_ids(
    task_id: str,
    strategy: str,
    max_variants: int,
) -> List[str]:
    """Generate variant task IDs for a given strategy.

    Examples:
        >>> generate_variant_ids("402", "depth", 2)
        ['402__d1', '402__d2']
        >>> generate_variant_ids("402__d1", "breadth", 1)
        ['402__d1__b1']
    """
    prefix = _STRATEGY_PREFIX.get(strategy)
    if prefix is None:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Expected one of {list(_STRATEGY_PREFIX)}"
        )
    return [f"{task_id}__{prefix}{i}" for i in range(1, max_variants + 1)]


def parse_variant_id(variant_id: str) -> Dict[str, Any]:
    """Parse a variant ID into its components.

    Returns:
        dict with keys: input_task_id, steps (list of step strings),
        root_task_id (the original, un-evolved task id).

    Example:
        >>> parse_variant_id("402__d1__b1")
        {'root_task_id': '402', 'input_task_id': '402__d1',
         'steps': ['d1', 'b1']}
    """
    parts = variant_id.split("__")
    root = parts[0]
    steps = parts[1:] if len(parts) > 1 else []
    input_task_id = "__".join(parts[:-1]) if steps else root
    return {
        "root_task_id": root,
        "input_task_id": input_task_id,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# synth_info.json helpers
# ---------------------------------------------------------------------------

def write_synth_info(
    output_dir: str,
    variant_id: str,
    info: Dict[str, Any],
) -> str:
    """Write synth_info.json for a variant.  Returns the absolute file path."""
    path = os.path.abspath(os.path.join(output_dir, variant_id, "synth_info.json"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=str)
    return path


def read_synth_info(output_dir: str, variant_id: str) -> Optional[Dict[str, Any]]:
    """Read synth_info.json for a variant.  Returns None if not found."""
    path = os.path.abspath(os.path.join(output_dir, variant_id, "synth_info.json"))
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_synth_info(
    variant_id: str,
    input_task_id: str,
    strategy_name: str,
    evol_target: str,
    input_dir: str,
    status: str = "in_progress",
    verdict: str = "",
    **timing,
) -> Dict[str, Any]:
    """Build a synth_info dict with standard fields."""
    info: Dict[str, Any] = {
        "task_id": variant_id,
        "input_task_id": input_task_id,
        "strategy_name": strategy_name,
        "evol_target": evol_target,
        "input_dir": input_dir,
        "status": status,
        "verdict": verdict,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    info.update(timing)
    return info


# ---------------------------------------------------------------------------
# Task listing
# ---------------------------------------------------------------------------

def list_input_tasks(input_dir: str) -> List[str]:
    """List task IDs (top-level directory names) in input_dir.

    Only includes directories that contain a complete Harbor task.
    If a synth_info.json is present (chained run input), the task must
    also have status=done and verdict=PASS.

    All paths resolved to absolute.
    """
    base = pathlib.Path(input_dir).resolve()
    if not base.is_dir():
        return []
    tasks = []
    for entry in sorted(base.iterdir()):
        if entry.is_dir() and not entry.name.startswith((".", "_")):
            if not is_harbor_task_complete(str(entry)):
                continue
            # If synth_info.json exists (output from a previous evol run),
            # only include PASS variants
            info = read_synth_info(str(base), entry.name)
            if info is not None:
                if info.get("status") != "done" or info.get("verdict") != "PASS":
                    continue
            tasks.append(entry.name)
    return tasks


def list_pass_variants(output_dir: str) -> List[str]:
    """List variant IDs in output_dir that have status=done and verdict=PASS.

    Used when chaining: the next run reads these as input tasks.
    All paths resolved to absolute.
    """
    base = pathlib.Path(output_dir).resolve()
    if not base.is_dir():
        return []
    variants = []
    for entry in sorted(base.iterdir()):
        if entry.is_dir() and not entry.name.startswith((".", "_")):
            info = read_synth_info(output_dir, entry.name)
            if info and info.get("status") == "done" and info.get("verdict") == "PASS":
                variants.append(entry.name)
    return variants


# ---------------------------------------------------------------------------
# Filter / root-ID helpers
# ---------------------------------------------------------------------------

def parse_variant_root(variant_id: str) -> str:
    """Extract root seed task_id by stripping evolution suffixes.

    Evolution suffixes match ``__[db]\\d+`` (e.g. ``__d1``, ``__b2``).
    Everything else (including ``__`` in original task names like
    ``kaggle_notebook__author_title``) is preserved.

    Examples:
        >>> parse_variant_root('402__d1__b1')
        '402'
        >>> parse_variant_root('kaggle_notebook__abaojiang_foo__b1__d1')
        'kaggle_notebook__abaojiang_foo'
        >>> parse_variant_root('stack_overflow__12345')
        'stack_overflow__12345'
    """
    import re
    # Strip trailing __d1, __b2, etc. (evolution step suffixes)
    return re.sub(r'(__[db]\d+)+$', '', variant_id)


def load_filter_csv(filter_csv: str) -> Set[str]:
    """Load task_id column from a filter CSV.  Returns a set of IDs."""
    ids: Set[str] = set()
    with open(filter_csv, newline="") as f:
        for row in csv.DictReader(f):
            tid = row.get("task_id", "").strip()
            if tid:
                ids.add(tid)
    return ids


# ---------------------------------------------------------------------------
# Rollout gap-finding
# ---------------------------------------------------------------------------

def _scan_completed_rollouts(model_dir: pathlib.Path) -> Set[str]:
    """Return task_ids that have completed rollout results under *model_dir*."""
    rolled: Set[str] = set()
    if not model_dir.is_dir():
        return rolled
    for entry in model_dir.iterdir():
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        has_results = any(
            (rd / "verifier" / "reward.txt").exists()
            for rd in entry.iterdir()
            if rd.is_dir() and not rd.name.startswith("_")
        )
        if has_results:
            rolled.add(entry.name)
    return rolled


def find_rollout_gaps(
    tasks_dir: str,
    rollout_dir: str,
    model_config_name: str,
    filter_ids: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Find tasks in *tasks_dir* that don't have rollout results yet.

    Works for both raw seeds (no synth_info.json) and evolved tasks
    (requires status=done and verdict=PASS).

    *filter_ids*: if set, only include tasks whose root seed ID is in this set.
    """
    base = pathlib.Path(tasks_dir)
    if not base.is_dir():
        return []

    rolled = _scan_completed_rollouts(pathlib.Path(rollout_dir) / model_config_name)

    gaps: List[Dict[str, Any]] = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or entry.name.startswith((".", "_")):
            continue
        if entry.name in rolled:
            continue
        if filter_ids and parse_variant_root(entry.name) not in filter_ids:
            continue
        # Must be a complete Harbor task
        if not is_harbor_task_complete(str(entry)):
            continue
        # If synth_info exists (evolved task), require PASS
        info = read_synth_info(tasks_dir, entry.name)
        if info is not None:
            if info.get("status") != "done" or info.get("verdict") != "PASS":
                continue
        gaps.append({
            "task_id": entry.name,
            "task_name": entry.name,
            "task_path": str(entry.resolve()),
            "instruction": read_instruction(str(entry)),
        })

    logger.info(
        "[rollout] gaps for %s: %d need rollout (%d already done)",
        model_config_name, len(gaps), len(rolled),
    )
    return gaps


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def read_instruction(task_path: str) -> str:
    """Read instruction.md from a task folder."""
    p = os.path.join(task_path, "instruction.md")
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def write_summary_csv(output_dir: str) -> str:
    """Scan synth_info.json files in *output_dir* and write summary.csv."""
    csv_path = os.path.join(output_dir, "summary.csv")
    base = pathlib.Path(output_dir)
    if not base.is_dir():
        return csv_path

    rows = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or entry.name.startswith((".", "_")):
            continue
        info_path = entry / "synth_info.json"
        if info_path.exists():
            with open(info_path) as f:
                rows.append(json.load(f))

    if not rows:
        return csv_path

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Summary CSV: %s (%d rows)", csv_path, len(rows))
    return csv_path
