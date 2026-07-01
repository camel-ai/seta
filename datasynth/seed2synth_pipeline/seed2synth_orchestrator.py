"""
seed2synth_orchestrator.py
==========================

Orchestrates two concurrent async loops:
  1. Synth loop   — samples seed tasks from CSVs → runs Seed2TaskPipeline
  2. Rollout loop — finds un-rolled synth tasks   → runs GRPORollout

All paths are injected via __init__ / source_configs; none hardcoded here.

Source config shape:
    [
        {
            "source":   "kaggle_notebook",       # dir name under seed_data/ and synth_data/
            "csv_path": "/abs/path/to/kaggle_post_processed.csv",
        },
        ...
    ]

Usage (see run_orchestrator.py for the actual entry point):
    orch = Seed2SynthOrchestrator(
        source_configs=[...],
        seed_data_dir="/home/ubuntu/seed_data_scraping/seed_data/",
        synth_data_dir="/home/ubuntu/seed_data_scraping/synth_data/",
        rollout_dir="/home/ubuntu/seed_data_scraping/synth_data_rollouts/",
        model_url="http://localhost:8000",
    )
    results = asyncio.run(orch.run())
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

try:
    from hf_utils import download_seed_task
except ImportError:
    download_seed_task = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source-type routing for ClaudeSeed2IdeaAgent
# Maps seed_data dir name → source_type key accepted by ClaudeSeed2IdeaAgent.
# None = not yet supported, skip during synth.
# ---------------------------------------------------------------------------
# Sources supported by Seed2TaskPipeline / ClaudeSeed2IdeaAgent.ADAPTER_MAP.
_SUPPORTED_SOURCES = {"nl2bash", "stackoverflow", "stack_overflow", "unix_linux_se", "kaggle_notebook", "nvd"}

# Sentinel passed to update_synth_summary to write status=in_progress
class _InProgress:
    pass
_IN_PROGRESS = _InProgress()

# Sentinel passed to update_synth_summary to write status=timeout
class _Timeout:
    pass
_TIMEOUT = _Timeout()

# ---------------------------------------------------------------------------
# Rate-limit retry settings for the synth worker
# ---------------------------------------------------------------------------
_RATE_LIMIT_POLL_INTERVAL_S = 10 * 60      # re-check every 10 min
_RATE_LIMIT_MAX_WAIT_S      = 5 * 60 * 60  # give up after 5 h

# Per-task synth timeout — kills tasks that exceed this to maintain throughput
_SYNTH_TIMEOUT_S = 60 * 60                 # 60 minutes

# How often the rollout consumer polls for new tasks while synth is still running
_ROLLOUT_POLL_INTERVAL_S = 30


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if the exception looks like a provider rate-limit response."""
    msg = str(exc).lower()
    type_name = type(exc).__name__
    return (
        "rate limit" in msg
        or "rate_limit" in msg
        or "429" in msg
        or "402" in msg
        or "quota" in msg
        or "out of extra usage" in msg
        or "out_of_credits" in msg
        or "overage_disabled_reason" in msg
        or type_name in ("RateLimitError", "APIStatusError")
        and "429" in msg
    )


def _result_is_quota_error(result) -> bool:
    """Return True if a pipeline result (non-exception) contains a quota/rate-limit error."""
    if result is None:
        return False
    result_str = str(getattr(result, "result", "") or "").lower()
    return (
        "quota" in result_str
        or "rate limit" in result_str
        or "402" in result_str
        or "429" in result_str
    )


# Harbor format (task.toml + instruction.md + environment/Dockerfile + solution/solve.sh)
_ESSENTIAL_HARBOR = [
    "task.toml",
    "instruction.md",
    "environment/Dockerfile",
    "solution/solve.sh",
]
# Legacy format kept for backwards compat
_ESSENTIAL_LEGACY = ["Dockerfile", "task.yaml", "run-tests.sh", "solution.sh"]

# ---------------------------------------------------------------------------
# Default agent config for TerminalEnvironment rollouts.
# "prompt" → loads seta_env/agent/prompts/sys_prompt_default.md (takes
#   precedence over "system_message").
# "agent"  → "train_agent" (seta_env.agent.train_agent.AgentTrain).
# tool_names must be a subset of: shell_exec, shell_view, shell_wait,
#   shell_write_to_process, shell_kill_process, shell_write_content_to_file,
#   shell_image_read (plus note-taking tools from NoteTakingToolkit).
# ---------------------------------------------------------------------------
DEFAULT_AGENT_CONFIG = {
    "agent":            "train_agent",
    "prompt":           "sys_prompt_default",
    "max_total_tokens": 32768,
    "max_iteration":    30,
    "thinking":         False,   # False → appends /no_think to system prompt at agent reset
    "tool_names": [
        "shell_exec",
        "shell_view",
        "shell_wait",
        "shell_write_to_process",
        "shell_kill_process",
        "shell_write_content_to_file",
    ],
}


def _resolve_model_name(model_url: str) -> str:
    """Query the SGLang server's /v1/models endpoint and return the first model ID.

    SGLang often reports a local path (e.g. '/root/models/Qwen3-8B') rather than
    the HuggingFace ID.  Using the server's own ID avoids 404 errors.
    """
    import urllib.request, json as _json

    url = model_url.rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"
    try:
        with urllib.request.urlopen(f"{url}/models", timeout=10) as resp:
            data = _json.loads(resp.read())
        model_id = data["data"][0]["id"]
        logger.info("[model] Auto-detected model name from server: %s", model_id)
        return model_id
    except Exception as exc:
        raise RuntimeError(
            f"Could not auto-detect model name from {url}/models: {exc}. "
            "Pass --model-name explicitly."
        ) from exc


def _build_model_config(model_url: str, model_name: str | None = None) -> dict:
    """Return CAMEL model_config dict for an SGLang-served model.

    If model_name is None or empty, queries the server's /v1/models endpoint
    to auto-detect the correct model ID.
    """
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType

    url = model_url.rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"

    if not model_name:
        model_name = _resolve_model_name(model_url)

    return {
        "model": ModelFactory.create(
            model_platform=ModelPlatformType.SGLANG,
            model_type=model_name,
            url=url,
            api_key="EMPTY",
            model_config_dict={"max_tokens": 4096, "stream": False},
        )
    }


# ============================================================================
# PART 1 — Seed Inventory
# ============================================================================

# Target sampling distribution across categories.
# Values are relative weights (need not sum to 1 — normalized at runtime).
# Unlisted categories get CATEGORY_DEFAULT_WEIGHT.
CATEGORY_WEIGHTS: dict[str, float] = {
    "Software Engineering":        0.25,
    "System Administration":       0.30,
    "File Operations":             0.12,
    "Text Processing":             0.08,
    "Network Configuration":       0.07,
    "Cybersecurity":               0.07,
    "Data Analysis & Processing":  0.1,
    "Database Management":         0.05,
    "Scientific Computing":        0.1,
    "Machine Learning & Model Training": 0.05,
}
CATEGORY_DEFAULT_WEIGHT: float = 0.005  # anything not listed above

def _seed_json_path(seed_data_dir: str, source: str, folder: str) -> str:
    """Return path to the primary seed JSON for a task folder.

    kaggle_notebook tasks contain kernel-metadata.json instead of main.json.
    """
    base = Path(seed_data_dir) / source / folder
    if source == "kaggle_notebook":
        return str(base / "kernel-metadata.json")
    return str(base / "main.json")


def load_csv_tasks(csv_path: str, source: str, seed_data_dir: str) -> list[dict]:
    """Read metadata.csv (unified format) and return one task dict per non-filtered row.

    Supports both new unified format (task_id column) and legacy format (folder column).

    Each dict: source, task_id, category, seed_json_path, title
    """
    df = pd.read_csv(csv_path, dtype=str)

    # Filter out tasks marked as filtered=true
    if "filtered" in df.columns:
        df = df[df["filtered"].str.strip().str.lower() != "true"]

    df["category"] = df["category"].fillna("unknown").str.strip()

    tasks = []
    for _, row in df.iterrows():
        # Support both new (task_id) and legacy (folder) column names
        task_id = str(row.get("task_id") or row.get("folder", "")).strip()
        if not task_id:
            continue

        tasks.append(
            {
                "source":         source,
                "task_id":        task_id,
                "category":       row["category"],
                "seed_json_path": _seed_json_path(seed_data_dir, source, task_id),
                "title":          str(row.get("title", "")).strip(),
            }
        )
    return tasks


def get_synth_task_ids(synth_data_dir: str, source: str, skip_timeout: bool = False) -> set[str]:
    """Return set of task_ids to skip based on synth_info.json files.

    Tasks with status='done' and verdict='pass'/'ditch' are always skipped.
    Tasks with status='timeout' are skipped only if skip_timeout=True (default: re-queued).
    Tasks with status='in_progress' are always re-queued (crashed/interrupted run).
    Tasks absent from synth_info.json are queued for synth.

    Falls back to reading summary.csv if synth_info.json files don't exist yet.
    """
    skip_ids = set()
    source_dir = Path(synth_data_dir) / source

    if not source_dir.exists():
        return set()

    # Check per-task synth_info.json files (new approach)
    for task_dir in source_dir.iterdir():
        if not task_dir.is_dir() or task_dir.name.startswith(".") or task_dir.name == "summary.csv":
            continue

        synth_info = _read_synth_info(synth_data_dir, source, task_dir.name)
        if synth_info:
            status = synth_info.get("status")
            verdict = synth_info.get("verdict")

            # Always skip done+pass and done+ditch
            if status == "done" and verdict in ("pass", "ditch"):
                skip_ids.add(task_dir.name)
            # Skip timeout only if requested
            elif status == "timeout" and skip_timeout:
                skip_ids.add(task_dir.name)
            # Re-queue in_progress (crashed run)

    # Fallback: also check summary.csv for compatibility
    csv_path = source_dir / "summary.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
            skip_statuses = {"done", "timeout"} if skip_timeout else {"done"}
            for task_id in df.loc[df["status"].isin(skip_statuses), "task_id"].tolist():
                if task_id not in skip_ids:
                    skip_ids.add(task_id)
        except Exception:
            pass

    return skip_ids


def compute_category_counts(tasks: list[dict], key: str = "category") -> dict[str, int]:
    """Count tasks per category value."""
    counts: dict[str, int] = {}
    for t in tasks:
        counts[t[key]] = counts.get(t[key], 0) + 1
    return counts


def _sample_by_weights(
    remaining: list[dict],
    total: int,
    weights: dict[str, float],
    default_weight: float,
    seed: int = 42,
) -> list[dict]:
    """Sample up to `total` tasks from `remaining` according to category weights.

    Algorithm:
    1. Compute each category's quota = round(normalized_weight * total).
    2. Shuffle each category bucket (deterministically via seed) so tasks are
       drawn evenly across sources regardless of source load order.
    3. Take min(quota, available) tasks from each category bucket.
    4. Fill any leftover slots from remaining tasks (shuffled).
    """
    import random
    from collections import defaultdict

    rng = random.Random(seed)

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for t in remaining:
        by_cat[t["category"]].append(t)

    # Shuffle within each category so source order doesn't bias selection
    for tasks in by_cat.values():
        rng.shuffle(tasks)

    # Normalize weights over categories actually present in the pool
    raw = {c: weights.get(c, default_weight) for c in by_cat}
    total_w = sum(raw.values()) or 1.0
    quotas = {c: max(1, round(raw[c] / total_w * total)) for c in by_cat}

    selected: list[dict] = []
    leftover: list[dict] = []
    for cat, tasks in by_cat.items():
        q = quotas[cat]
        selected.extend(tasks[:q])
        leftover.extend(tasks[q:])

    # If rounding left us short, fill from leftover
    if len(selected) < total:
        rng.shuffle(leftover)
        selected.extend(leftover[: total - len(selected)])

    return selected[:total]


def collect_synth_queue(
    source_configs: list[dict],
    seed_data_dir: str,
    synth_data_dir: str,
    max_tasks: int | None = None,
    skip_timeout: bool = False,
) -> list[dict]:
    """Return a deterministic queue of tasks to synthesize.

    All seed tasks across all sources are shuffled once with a fixed seed (42)
    to produce a stable ordering.  The queue is built by scanning that fixed
    list sequentially and skipping tasks whose status is 'done' in summary.csv.
    Tasks with status 'error', 'in_progress', or absent from the CSV are
    re-queued.  If skip_timeout=True, tasks with status 'timeout' are also
    skipped; by default they are re-queued.  If max_tasks is None, all
    eligible tasks are queued.
    """
    import random

    all_tasks: list[dict] = []
    done_ids_by_source: dict[str, set[str]] = {}

    for cfg in source_configs:
        source   = cfg["source"]
        csv_path = cfg["csv_path"]

        seed_tasks = load_csv_tasks(csv_path, source, seed_data_dir)
        synth_ids  = get_synth_task_ids(synth_data_dir, source, skip_timeout=skip_timeout)
        done_ids_by_source[source] = synth_ids

        logger.info(
            "[inventory] %s: %d seed tasks, %d done, %d eligible",
            source, len(seed_tasks), len(synth_ids),
            len(seed_tasks) - len(synth_ids),
        )
        all_tasks.extend(seed_tasks)

    # Fixed shuffle — stable ordering across all runs
    rng = random.Random(42)
    rng.shuffle(all_tasks)

    # Sequential scan: skip done, queue the rest (up to cap if set)
    queued: list[dict] = []
    for task in all_tasks:
        if max_tasks is not None and len(queued) >= max_tasks:
            break
        if task["task_id"] in done_ids_by_source.get(task["source"], set()):
            continue
        queued.append(task)

    cat_counts = compute_category_counts(queued)
    logger.info(
        "[inventory] total queued: %d tasks — categories: %s",
        len(queued), cat_counts,
    )
    return queued


# ============================================================================
# PART 2 — Rollout Inventory
# ============================================================================

def _read_task_instruction(task_path: Path) -> str:
    # Harbor format: instruction.md
    md_file = task_path / "instruction.md"
    if md_file.exists():
        try:
            return md_file.read_text().strip()
        except Exception as exc:
            logger.debug("Could not read instruction.md at %s: %s", task_path, exc)
    # Legacy format: instruction embedded in task.yaml
    yaml_file = task_path / "task.yaml"
    if yaml_file.exists():
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f) or {}
            for key in ("instruction", "description", "task_description"):
                if key in data:
                    return str(data[key])
        except Exception as exc:
            logger.debug("Could not read task.yaml at %s: %s", task_path, exc)
    return ""


def _is_valid_synth_task(task_path: Path) -> bool:
    harbor = (
        all((task_path / f).exists() for f in _ESSENTIAL_HARBOR)
        and (task_path / "tests").is_dir()
    )
    legacy = (
        all((task_path / f).exists() for f in _ESSENTIAL_LEGACY)
        and (task_path / "tests").is_dir()
    )
    return harbor or legacy


def get_valid_synth_tasks(synth_data_dir: str) -> list[dict]:
    """Scan synth_data_dir/{source}/{task_id}/ and return rollout-ready tasks.

    Each dict: source, task_id, task_name, task_path, instruction
    """
    results: list[dict] = []
    base = Path(synth_data_dir)
    if not base.exists():
        return results

    for source_dir in sorted(base.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        for task_dir in sorted(source_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            if _is_valid_synth_task(task_dir):
                results.append(
                    {
                        "source":      source,
                        "task_id":     task_dir.name,
                        "task_name":   f"{source}__{task_dir.name}",
                        "task_path":   str(task_dir.resolve()),
                        "instruction": _read_task_instruction(task_dir),
                    }
                )
            else:
                logger.debug("Skipping incomplete synth task: %s", task_dir)
    return results


def get_harbor_tasks(tasks_dir: str, source: str = "harbor", limit: int = 0) -> list[dict]:
    """Scan a flat Harbor tasks directory ({tasks_dir}/{task_id}/) directly.

    Use this to run rollouts against an existing Harbor dataset that does not
    follow the synth_data/{source}/{task_id}/ layout.

    Args:
        tasks_dir: path containing task subdirectories (e.g. seta-env-harbor/)
        source:    label used in task_name and rollout output dirs
        limit:     if > 0, return only the first N valid tasks

    Each dict: source, task_id, task_name, task_path, instruction
    """
    results: list[dict] = []
    base = Path(tasks_dir)
    if not base.exists():
        return results

    for task_dir in sorted(base.iterdir(), key=lambda p: p.name):
        if not task_dir.is_dir():
            continue
        if _is_valid_synth_task(task_dir):
            results.append(
                {
                    "source":      source,
                    "task_id":     task_dir.name,
                    "task_name":   f"{source}__{task_dir.name}",
                    "task_path":   str(task_dir.resolve()),
                    "instruction": _read_task_instruction(task_dir),
                }
            )
        else:
            logger.debug("Skipping incomplete Harbor task: %s", task_dir)
        if limit > 0 and len(results) >= limit:
            break

    return results


def get_rolled_out_ids(rollout_dir: str, model_config_name: str) -> set[str]:
    """Return set of task_names (source__task_id) with status='done' in summary.csv.

    Only tasks marked done are skipped; error/missing tasks are re-queued.
    """
    base = Path(rollout_dir) / model_config_name
    if not base.exists():
        return set()

    csv_path = base / "summary.csv"
    if not csv_path.exists():
        return set()

    import csv as _csv
    ids: set[str] = set()
    with open(csv_path, newline="") as f:
        for row in _csv.DictReader(f):
            if row.get("status") == "done":
                ids.add(row["task_name"])
    return ids


def find_rollout_gaps(
    synth_data_dir: str,
    rollout_dir: str,
    model_config_name: str,
) -> list[dict]:
    """Return valid synth tasks with no rollout yet for the given model config."""
    valid  = get_valid_synth_tasks(synth_data_dir)
    rolled = get_rolled_out_ids(rollout_dir, model_config_name)
    gaps   = [t for t in valid if t["task_name"] not in rolled]
    logger.info(
        "[inventory] rollout gaps for %s: %d/%d tasks need rollout",
        model_config_name, len(gaps), len(valid),
    )
    return gaps


# ============================================================================
# PART 2.5 — Summary CSV helpers
# ============================================================================
# Layout:
#   synth_data_dir/{source}/summary.csv          — one row per synth'd task
#   rollout_dir/{model_config_name}/summary.csv  — one row per rolled-out task
#
# Upsert semantics: re-running the same task overwrites the existing row.
# asyncio.Lock per file path prevents concurrent coroutine write collisions.
# ============================================================================

_SYNTH_SUMMARY_COLS = [
    "task_id", "source", "category", "title",
    "status",              # "done" | "error" | "skipped"
    "verdict",             # "PASS" | "FAIL" | "N/A"
    "stage",               # "full" | "idea-only"
    "idea_time_s",         # seconds for idea agent
    "datapoint_time_s",    # seconds for datapoint agent
    "total_synth_time_s",  # wall-clock seconds for full pipeline.run()
    "timestamp",
]
_ROLLOUT_SUMMARY_COLS = [
    "task_name", "source", "task_id",
    "n_total",          # total trajectories run
    "n_passed",         # trajectories with pass_ratio == 1.0
    "n_failed",         # trajectories where evaluation failed entirely (reward is None)
    "mean_pass_ratio",  # mean pass ratio across non-failed trajectories
    "std_pass_ratio",   # std dev of pass ratio across non-failed trajectories
    "rollout_time_s",   # wall-clock seconds for rollout.run()
    "status",           # "done" | "error"
    "timestamp",
]

# Per-file asyncio locks (created on demand)
_summary_locks: dict[str, asyncio.Lock] = {}


def _get_lock(path: str) -> asyncio.Lock:
    if path not in _summary_locks:
        _summary_locks[path] = asyncio.Lock()
    return _summary_locks[path]


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_synth_info(
    synth_data_dir: str,
    source: str,
    task_id: str,
    status: str,
    verdict: str = None,
    stage: str = None,
    total_synth_time_s: float = 0.0,
    idea_time_s: float = None,
    datapoint_time_s: float = None,
    harbor_oracle_passed: bool = None,
    harbor_empty_failed: bool = None,
) -> None:
    """Write synth_info.json status file to task folder.

    This is the source of truth for task status — replaces relying on summary.csv.

    Args:
        synth_data_dir: root of synth_data/
        source: source name
        task_id: task ID
        status: "in_progress" | "timeout" | "done"
        verdict: "ditch" | "pass" | "fail" | None
        stage: "full" | "idea-only"
        total_synth_time_s: wall-clock seconds
        idea_time_s: seconds for idea agent
        datapoint_time_s: seconds for datapoint agent
        harbor_oracle_passed: True/False if harbor validation ran
        harbor_empty_failed: True/False if harbor validation ran
    """
    task_dir = Path(synth_data_dir) / source / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    synth_info_path = task_dir / "synth_info.json"

    synth_info = {
        "task_id": task_id,
        "source": source,
        "status": status,
        "verdict": verdict,
        "stage": stage,
        "total_synth_time_s": round(total_synth_time_s, 1) if total_synth_time_s else None,
        "idea_time_s": round(idea_time_s, 1) if idea_time_s else None,
        "datapoint_time_s": round(datapoint_time_s, 1) if datapoint_time_s else None,
        "timestamp": _now_utc(),
    }

    if harbor_oracle_passed is not None:
        synth_info["harbor_oracle_passed"] = harbor_oracle_passed
    if harbor_empty_failed is not None:
        synth_info["harbor_empty_failed"] = harbor_empty_failed

    with open(synth_info_path, 'w') as f:
        json.dump(synth_info, f, indent=2)

    logger.debug(f"Wrote synth_info.json for {source}/{task_id} (status={status}, verdict={verdict})")


def _read_synth_info(synth_data_dir: str, source: str, task_id: str) -> dict | None:
    """Read synth_info.json from task folder, or None if not found."""
    synth_info_path = Path(synth_data_dir) / source / task_id / "synth_info.json"
    if not synth_info_path.exists():
        return None
    try:
        with open(synth_info_path) as f:
            return json.load(f)
    except Exception as e:
        logger.debug(f"Error reading synth_info.json for {source}/{task_id}: {e}")
        return None


async def update_synth_summary(
    synth_data_dir: str,
    source: str,
    item: dict,
    judge_result,           # SynthResult | None
    stage: str,
    total_synth_time_s: float = 0.0,
) -> None:
    """Upsert one row into synth_data_dir/{source}/summary.csv."""
    csv_path = os.path.join(synth_data_dir, source, "summary.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if isinstance(judge_result, _InProgress):
        status  = "in_progress"
        verdict = "N/A"
        meta    = {}
    elif isinstance(judge_result, _Timeout):
        status  = "timeout"
        verdict = "N/A"
        meta    = {}
    elif judge_result is not None:
        status  = "done"
        verdict = getattr(judge_result, "verdict", "N/A")
        meta    = getattr(judge_result, "metadata", {}) or {}
    else:
        status  = "error"
        verdict = "N/A"
        meta    = {}

    row = {
        "task_id":            item["task_id"],
        "source":             source,
        "category":           item.get("category", ""),
        "title":              item.get("title", ""),
        "status":             status,
        "verdict":            verdict,
        "stage":              stage,
        "idea_time_s":        meta.get("idea_time_s", ""),
        "datapoint_time_s":   meta.get("datapoint_time_s", ""),
        "total_synth_time_s": round(total_synth_time_s, 1) if total_synth_time_s else "",
        "timestamp":          _now_utc(),
    }

    # Also write synth_info.json (source of truth for status)
    _write_synth_info(
        synth_data_dir,
        source,
        item["task_id"],
        status=status,
        verdict=verdict,
        stage=stage,
        total_synth_time_s=total_synth_time_s,
        idea_time_s=meta.get("idea_time_s"),
        datapoint_time_s=meta.get("datapoint_time_s"),
    )

    async with _get_lock(csv_path):
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        else:
            df = pd.DataFrame({c: pd.Series(dtype=str) for c in _SYNTH_SUMMARY_COLS})

        # Upsert: drop existing row for this task_id, append new one
        df = df[df["task_id"] != str(item["task_id"])]
        df = pd.concat([df, pd.DataFrame([{k: str(v) for k, v in row.items()}])], ignore_index=True)
        df.to_csv(csv_path, index=False)

    logger.info("[summary] synth %s/%s → %s", source, item["task_id"], csv_path)


def _read_ctrf_pass_ratio(ctrf_path: str) -> float | None:
    """Parse pass ratio from a ctrf.json written by pytest. Returns None on any error."""
    try:
        with open(ctrf_path) as f:
            data = json.load(f)
        summary = data["results"]["summary"]
        total = summary.get("tests", 0)
        if total == 0:
            return None
        return summary.get("passed", 0) / total
    except Exception:
        return None


async def update_rollout_summary(
    rollout_dir: str,
    model_config_name: str,
    item: dict,
    rollout_results,        # list[(run_info, reward | None)] | None
    rollout_time_s: float = 0.0,
) -> None:
    """Upsert one row into rollout_dir/{model_config_name}/summary.csv."""
    csv_path = os.path.join(rollout_dir, model_config_name, "summary.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if rollout_results:
        trial_root = os.path.join(
            rollout_dir, model_config_name, item["source"], item["task_id"]
        )
        n_total  = len(rollout_results)
        n_failed = sum(1 for _, r in rollout_results if r is None)

        pass_ratios = []
        for run_info, reward in rollout_results:
            if reward is None:
                continue  # infra failure — excluded from pass ratio stats
            uid = (run_info or {}).get("uid", "")
            ctrf_path = os.path.join(trial_root, uid, "verifier", "ctrf.json")
            pr = _read_ctrf_pass_ratio(ctrf_path)
            if pr is None:
                pr = float(reward)  # fallback: binary reward.txt value
            pass_ratios.append(pr)

        n_passed        = sum(1 for pr in pass_ratios if pr == 1.0)
        if pass_ratios:
            mean_pass_ratio = round(sum(pass_ratios) / len(pass_ratios), 4)
            variance = sum((pr - mean_pass_ratio) ** 2 for pr in pass_ratios) / len(pass_ratios)
            std_pass_ratio  = round(variance ** 0.5, 4)
        else:
            mean_pass_ratio = std_pass_ratio = 0.0
    else:
        n_total = n_passed = n_failed = 0
        mean_pass_ratio = std_pass_ratio = 0.0

    row = {
        "task_name":       item["task_name"],
        "source":          item["source"],
        "task_id":         item["task_id"],
        "n_total":         n_total,
        "n_passed":        n_passed,
        "n_failed":        n_failed,
        "mean_pass_ratio": mean_pass_ratio,
        "std_pass_ratio":  std_pass_ratio,
        "rollout_time_s":  round(rollout_time_s, 1) if rollout_time_s else "",
        "status":          "done" if rollout_results else "error",
        "timestamp":       _now_utc(),
    }

    async with _get_lock(csv_path):
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        else:
            df = pd.DataFrame({c: pd.Series(dtype=str) for c in _ROLLOUT_SUMMARY_COLS})

        df = df[df["task_name"] != str(item["task_name"])]
        df = pd.concat([df, pd.DataFrame([{k: str(v) for k, v in row.items()}])], ignore_index=True)
        df.to_csv(csv_path, index=False)

    logger.info("[summary] rollout %s → %s", item["task_name"], csv_path)


# ============================================================================
# PART 3 — Synth Worker
# ============================================================================

def _build_synth_pipeline(output_base: str):
    """Construct a Seed2TaskPipeline with Claude agents (lazy import)."""
    _ap_dir = os.path.dirname(os.path.abspath(__file__))
    if _ap_dir not in sys.path:
        sys.path.insert(0, _ap_dir)

    from seed2task_pipeline import Seed2TaskPipeline
    from agents.claude_agents import (
        ClaudeSeed2IdeaAgent,
        ClaudeDatapointAgent,
    )

    return Seed2TaskPipeline(
        idea_agent=ClaudeSeed2IdeaAgent(),
        datapoint_agent=ClaudeDatapointAgent(),
        output_base=output_base,
    )


async def _synth_worker(
    worker_id: int,
    queue: asyncio.Queue,
    pipeline,
    results: list,
    stage: str = "full",
    synth_data_dir: str = "",
    seed_data_dir: str = "",
    hf_seed_repo: str = "",
    hf_token_env: str = "HF_TOKEN",
    rollout_queue: "asyncio.Queue | None" = None,
    pause_event: "asyncio.Event | None" = None,
) -> None:
    """Consume seed task dicts from queue and run Seed2TaskPipeline on each.

    If *rollout_queue* is provided, pushes successfully synthesised tasks into
    it so the rollout consumer can pick them up immediately (producer-consumer
    mode).  Rate-limit errors trigger an in-place retry every
    _RATE_LIMIT_POLL_INTERVAL_S seconds for up to _RATE_LIMIT_MAX_WAIT_S total;
    each retry runs the full pipeline from scratch.

    If *pause_event* is set, the worker waits while it is cleared (paused) and
    resumes when it is set again.  On quota errors the worker clears the event
    to pause all sibling workers, then waits for it to be set externally.
    """
    while True:
        # Wait if the orchestrator has paused all workers (e.g. quota exceeded)
        if pause_event is not None and not pause_event.is_set():
            logger.info("[synth_worker %d] Paused — waiting for quota to resume...", worker_id)
            await pause_event.wait()
            logger.info("[synth_worker %d] Resumed.", worker_id)

        item = await queue.get()
        if item is None:            # sentinel — shut down
            queue.task_done()
            return

        source    = item["source"]
        task_id   = item["task_id"]
        seed_json = item["seed_json_path"]

        if source not in _SUPPORTED_SOURCES:
            logger.warning(
                "[synth_worker %d] Skipping %s/%s — source '%s' not supported",
                worker_id, source, task_id, source,
            )
            results.append((item, None))
            queue.task_done()
            continue

        # v2 pipeline takes the seed folder (parent of main.json), not the JSON itself
        seed_folder = str(Path(seed_json).parent)

        # If seed folder missing, try to download on-demand from HuggingFace
        if not os.path.isdir(seed_folder):
            if download_seed_task and seed_data_dir and hf_seed_repo:
                logger.info(
                    "[synth_worker %d] Seed folder missing, downloading %s/%s from HF...",
                    worker_id, source, task_id,
                )
                try:
                    from seed2synth_config import Config
                    config = Config(
                        huggingface=type('obj', (object,), {
                            'seed_repo': hf_seed_repo,
                            'token_env': hf_token_env
                        })(),
                        paths=type('obj', (object,), {
                            'seed_data_dir': seed_data_dir
                        })()
                    )
                    success = download_seed_task(source, task_id, Path(seed_data_dir), config)
                    if not success:
                        logger.warning(
                            "[synth_worker %d] Failed to download %s/%s from HF",
                            worker_id, source, task_id,
                        )
                        results.append((item, None))
                        queue.task_done()
                        continue
                except Exception as e:
                    logger.warning(
                        "[synth_worker %d] Error downloading %s/%s: %s",
                        worker_id, source, task_id, e,
                    )
                    results.append((item, None))
                    queue.task_done()
                    continue
            else:
                logger.warning(
                    "[synth_worker %d] Seed folder missing, skipping %s/%s: %s",
                    worker_id, source, task_id, seed_folder,
                )
                results.append((item, None))
                queue.task_done()
                continue

        logger.info("[synth_worker %d] Starting %s/%s", worker_id, source, task_id)
        result = None
        rate_limit_waited = 0
        total_synth_time_s = 0.0

        # Mark as in-progress so other workers / resumed runs skip it
        if synth_data_dir:
            await update_synth_summary(synth_data_dir, source, item, _IN_PROGRESS, stage)

        try:
            while True:
                try:
                    t_synth = time.monotonic()
                    result = await asyncio.wait_for(
                        pipeline.run(seed_data_folder=seed_folder, stage=stage),
                        timeout=_SYNTH_TIMEOUT_S,
                    )
                    total_synth_time_s = time.monotonic() - t_synth

                    # Quota errors arrive as a result (not an exception) — detect and pause
                    if _result_is_quota_error(result):
                        if pause_event is not None:
                            logger.warning(
                                "[synth_worker %d] Quota exceeded on %s/%s — pausing ALL workers. "
                                "Set pause_event to resume.",
                                worker_id, source, task_id,
                            )
                            pause_event.clear()
                        else:
                            logger.warning(
                                "[synth_worker %d] Quota exceeded on %s/%s — no pause_event, "
                                "sleeping %d min before retry.",
                                worker_id, source, task_id,
                                _RATE_LIMIT_POLL_INTERVAL_S // 60,
                            )
                            await asyncio.sleep(_RATE_LIMIT_POLL_INTERVAL_S)
                        result = None
                        break

                    break  # success
                except asyncio.TimeoutError:
                    total_synth_time_s = time.monotonic() - t_synth
                    logger.warning(
                        "[synth_worker %d] Timeout on %s/%s after %.0fs",
                        worker_id, source, task_id, total_synth_time_s,
                    )
                    result = _TIMEOUT
                    break
                except Exception as exc:
                    if _is_rate_limit_error(exc):
                        if rate_limit_waited >= _RATE_LIMIT_MAX_WAIT_S:
                            logger.error(
                                "[synth_worker %d] Rate limit persisted >5 h on %s/%s, giving up",
                                worker_id, source, task_id,
                            )
                            result = None
                            break
                        logger.warning(
                            "[synth_worker %d] Rate limited on %s/%s — sleeping %d min "
                            "(total waited %d min). Will retry from scratch.",
                            worker_id, source, task_id,
                            _RATE_LIMIT_POLL_INTERVAL_S // 60,
                            rate_limit_waited // 60,
                        )
                        await asyncio.sleep(_RATE_LIMIT_POLL_INTERVAL_S)
                        rate_limit_waited += _RATE_LIMIT_POLL_INTERVAL_S
                        logger.info(
                            "[synth_worker %d] Retrying %s/%s from scratch",
                            worker_id, source, task_id,
                        )
                        continue  # retry pipeline.run() from scratch
                    else:
                        raise  # non-rate-limit errors bubble up

            verdict = result.verdict if result and not isinstance(result, _Timeout) else "N/A"
            logger.info(
                "[synth_worker %d] Done %s/%s verdict=%s total=%.0fs",
                worker_id, source, task_id, verdict, total_synth_time_s,
            )
            results.append((item, result))

            # Feed rollout queue if in producer-consumer mode
            if rollout_queue is not None and result is not None and synth_data_dir:
                task_out_path = Path(synth_data_dir) / source / task_id
                if _is_valid_synth_task(task_out_path):
                    await rollout_queue.put({
                        "source":      source,
                        "task_id":     task_id,
                        "task_name":   f"{source}__{task_id}",
                        "task_path":   str(task_out_path.resolve()),
                        "instruction": _read_task_instruction(task_out_path),
                    })
                    logger.info(
                        "[synth_worker %d] Queued %s/%s for rollout",
                        worker_id, source, task_id,
                    )
                else:
                    logger.warning(
                        "[synth_worker %d] Synth output for %s/%s is incomplete, "
                        "skipping rollout",
                        worker_id, source, task_id,
                    )

        except Exception as exc:
            logger.error(
                "[synth_worker %d] Error on %s/%s: %s",
                worker_id, source, task_id, exc,
            )
            results.append((item, None))
        finally:
            if synth_data_dir:
                await update_synth_summary(
                    synth_data_dir, source, item, result, stage, total_synth_time_s
                )
            queue.task_done()


# ============================================================================
# PART 4 — Rollout Worker
# ============================================================================

async def _rollout_worker(
    worker_id: int,
    queue: asyncio.Queue,
    rollout_dir: str,
    model_config_name: str,
    model_url: str,
    model_name: str,
    n_trajs: int,
    results: list,
    write_summary: bool = True,
    agent_config: dict | None = None,
) -> None:
    """Consume synth task dicts from queue and run GRPORollout on each."""
    _repo_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
    )
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    from seta_env.orchestrators.grpo_rollout import GRPORollout

    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return

        source  = item["source"]
        task_id = item["task_id"]
        logger.info("[rollout_worker %d] Starting %s/%s", worker_id, source, task_id)

        trial_root = os.path.join(rollout_dir, model_config_name, source, task_id)
        os.makedirs(trial_root, exist_ok=True)

        rollout_time_s = 0.0
        try:
            # Pass a callable so each trajectory gets its own model instance.
            # Sharing one model object across parallel trajectories causes
            # model._log_dir to be mutated concurrently, losing conv logs.
            rollout = GRPORollout(
                agent_config=agent_config or DEFAULT_AGENT_CONFIG,
                model_config=lambda: _build_model_config(model_url, model_name),
                env_config={"environment_type": "docker"},
                trial_root=trial_root,
            )
            t_rollout = time.monotonic()
            rollout_results = await rollout.run(
                task={
                    "task_name":   item["task_name"],
                    "task_path":   item["task_path"],
                    "instruction": item["instruction"],
                },
                n_trajs=n_trajs,
            )
            rollout_time_s = time.monotonic() - t_rollout
            results.append((item, rollout_results))
            logger.info(
                "[rollout_worker %d] Done %s/%s (%d trajs, %.0fs)",
                worker_id, source, task_id, len(rollout_results), rollout_time_s,
            )
        except Exception as exc:
            logger.error(
                "[rollout_worker %d] Error on %s/%s: %s",
                worker_id, source, task_id, exc,
            )
            rollout_results = None
            results.append((item, None))
        finally:
            if write_summary:
                await update_rollout_summary(
                    rollout_dir, model_config_name, item, rollout_results, rollout_time_s
                )
            queue.task_done()


# ============================================================================
# PART 4.5 — Rollout Consumer Worker (producer-consumer mode)
# ============================================================================

async def _rollout_consumer_worker(
    worker_id: int,
    queue: asyncio.Queue,
    synth_done_event: asyncio.Event,
    rollout_dir: str,
    model_config_name: str,
    model_url: str,
    model_name: str,
    n_trajs: int,
    results: list,
    agent_config: dict | None = None,
) -> None:
    """Consume rollout tasks from *queue*, polling until synth is done and queue is empty.

    Unlike _rollout_worker (which uses sentinel-based shutdown), this worker
    polls the queue every _ROLLOUT_POLL_INTERVAL_S seconds while synth is still
    running, and exits only when both:
      1. synth_done_event is set  (synth loop finished)
      2. queue is empty           (nothing left to roll out)

    Errors on individual tasks are logged and do NOT kill the worker.
    """
    _repo_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
    )
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    from seta_env.orchestrators.grpo_rollout import GRPORollout

    logger.info("[rollout_consumer %d] started", worker_id)

    while True:
        # Non-blocking get so we can check the done-event on empty queue
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            if synth_done_event.is_set():
                logger.info(
                    "[rollout_consumer %d] queue empty + synth done — exiting", worker_id
                )
                return
            await asyncio.sleep(_ROLLOUT_POLL_INTERVAL_S)
            continue

        source  = item["source"]
        task_id = item["task_id"]
        logger.info("[rollout_consumer %d] Starting %s/%s", worker_id, source, task_id)

        trial_root = os.path.join(rollout_dir, model_config_name, source, task_id)
        os.makedirs(trial_root, exist_ok=True)

        rollout_results = None
        rollout_time_s = 0.0
        try:
            # Callable ensures each trajectory gets its own model instance —
            # avoids concurrent mutation of model._log_dir across trajectories.
            rollout = GRPORollout(
                agent_config=agent_config or DEFAULT_AGENT_CONFIG,
                model_config=lambda: _build_model_config(model_url, model_name),
                env_config={"environment_type": "docker"},
                trial_root=trial_root,
            )
            t_rollout = time.monotonic()
            rollout_results = await rollout.run(
                task={
                    "task_name":   item["task_name"],
                    "task_path":   item["task_path"],
                    "instruction": item["instruction"],
                },
                n_trajs=n_trajs,
            )
            rollout_time_s = time.monotonic() - t_rollout
            results.append((item, rollout_results))
            logger.info(
                "[rollout_consumer %d] Done %s/%s (%d trajs, %.0fs)",
                worker_id, source, task_id, len(rollout_results), rollout_time_s,
            )
        except Exception as exc:
            logger.error(
                "[rollout_consumer %d] Error on %s/%s: %s",
                worker_id, source, task_id, exc,
            )
            results.append((item, None))
        finally:
            await update_rollout_summary(
                rollout_dir, model_config_name, item, rollout_results, rollout_time_s
            )
            queue.task_done()


# ============================================================================
# PART 5 — Orchestrator
# ============================================================================

class Seed2SynthOrchestrator:
    """Runs synth and rollout loops concurrently.

    Args:
        source_configs:        list of {"source": str, "csv_path": str}
        seed_data_dir:         root of raw seed data  ({source}/{task_id}/)
        synth_data_dir:        root of synth outputs  ({source}/{task_id}/)
        rollout_dir:           root of rollout outputs ({model_cfg}/{source}/{task_id}/)
        model_url:             vLLM server URL, e.g. "http://localhost:8000"
        model_config_name:     subdir name under rollout_dir, e.g. "Qwen3-8B_thinking"
        model_name:            HF model ID, e.g. "Qwen/Qwen3-8B"
        n_synth_workers:       parallel synth pipeline workers
        n_rollout_workers:     parallel rollout workers
        n_trajs:               trajectories per rollout task
        max_tasks:             max tasks to queue across all sources (None = no cap)
        synth_stage:           "full" or "idea-only"
    """

    def __init__(
        self,
        source_configs: list[dict],
        seed_data_dir: str,
        synth_data_dir: str,
        rollout_dir: str,
        model_url: str,
        model_config_name: str = "Qwen3-8B_thinking",
        model_name: str = "Qwen/Qwen3-8B",
        n_synth_workers: int = 3,
        n_rollout_workers: int = 2,
        n_trajs: int = 8,
        max_tasks: int | None = None,
        synth_stage: str = "full",
        thinking: bool = False,
        skip_timeout: bool = False,
        hf_config: dict | None = None,
    ):
        self.source_configs    = source_configs
        self.seed_data_dir     = seed_data_dir
        self.synth_data_dir    = synth_data_dir
        self.rollout_dir       = rollout_dir
        self.model_url         = model_url
        self.model_config_name = model_config_name
        self.model_name        = model_name
        self.n_synth_workers   = n_synth_workers
        self.n_rollout_workers = n_rollout_workers
        self.n_trajs           = n_trajs
        self.max_tasks         = max_tasks
        self.synth_stage       = synth_stage
        self.skip_timeout      = skip_timeout
        self.hf_config         = hf_config or {}
        self.agent_config      = {**DEFAULT_AGENT_CONFIG, "thinking": thinking}

    async def run_synth_loop(self) -> list:
        """Inventory seed data → queue tasks → drain with N synth workers."""
        tasks = collect_synth_queue(
            self.source_configs,
            self.seed_data_dir,
            self.synth_data_dir,
            max_tasks=self.max_tasks,
            skip_timeout=self.skip_timeout,
        )
        if not tasks:
            logger.info("[synth_loop] No tasks to synthesize.")
            return []

        logger.info("[synth_loop] %d tasks queued across all sources.", len(tasks))

        queue: asyncio.Queue = asyncio.Queue()
        for t in tasks:
            await queue.put(t)
        for _ in range(self.n_synth_workers):
            await queue.put(None)   # sentinels

        pipeline = _build_synth_pipeline(self.synth_data_dir)
        results: list = []
        pause_event = asyncio.Event()
        pause_event.set()  # start unpaused

        workers = [
            _synth_worker(
                i, queue, pipeline, results,
                stage=self.synth_stage,
                synth_data_dir=self.synth_data_dir,
                seed_data_dir=self.seed_data_dir,
                hf_seed_repo=self.hf_config.get("seed_repo", "camel-ai/seta-env-seed2synth-seed"),
                hf_token_env=self.hf_config.get("token_env", "HF_TOKEN"),
                pause_event=pause_event
            )
            for i in range(self.n_synth_workers)
        ]
        await asyncio.gather(*workers)
        return results

    async def run_rollout_loop(self) -> list:
        """Find un-rolled synth tasks → queue → drain with M rollout workers."""
        tasks = find_rollout_gaps(
            self.synth_data_dir,
            self.rollout_dir,
            self.model_config_name,
        )
        if not tasks:
            logger.info("[rollout_loop] No tasks need rollout.")
            return []

        logger.info("[rollout_loop] %d tasks queued for rollout.", len(tasks))

        queue: asyncio.Queue = asyncio.Queue()
        for t in tasks:
            await queue.put(t)
        for _ in range(self.n_rollout_workers):
            await queue.put(None)

        results: list = []
        workers = [
            _rollout_worker(
                i, queue,
                self.rollout_dir,
                self.model_config_name,
                self.model_url,
                self.model_name,
                self.n_trajs,
                results,
                agent_config=self.agent_config,
            )
            for i in range(self.n_rollout_workers)
        ]
        await asyncio.gather(*workers)
        return results

    async def run_rollout_loop_tasks(self, tasks: list[dict]) -> list:
        """Drain a pre-built task list through the rollout workers.

        Use this to run rollouts against an arbitrary task list (e.g. from a
        Harbor dataset dir) without going through the synth inventory.
        """
        logger.info("[rollout_loop] %d tasks queued for rollout.", len(tasks))

        queue: asyncio.Queue = asyncio.Queue()
        for t in tasks:
            await queue.put(t)
        for _ in range(self.n_rollout_workers):
            await queue.put(None)

        results: list = []
        workers = [
            _rollout_worker(
                i, queue,
                self.rollout_dir,
                self.model_config_name,
                self.model_url,
                self.model_name,
                self.n_trajs,
                results,
                write_summary=False,
                agent_config=self.agent_config,
            )
            for i in range(self.n_rollout_workers)
        ]
        await asyncio.gather(*workers)
        return results

    async def run(self) -> dict:
        """Producer-consumer: synth feeds rollout queue; rollout drains until done.

        Design
        ------
        * A shared ``rollout_queue`` connects synth workers (producers) to
          rollout consumer workers.
        * At startup the queue is pre-filled with any tasks that are already
          synthesised but not yet rolled out (gap from a previous run).
        * Each synth worker pushes a valid completed task into ``rollout_queue``
          immediately after synthesis — rollout doesn't wait for all of synth.
        * ``synth_done_event`` is set (in a finally block) once all synth
          workers exit, regardless of success or failure.
        * Rollout consumer workers poll the queue every
          ``_ROLLOUT_POLL_INTERVAL_S`` seconds while synth is still running,
          and exit only when the event is set *and* the queue is empty.
        * Individual task errors in either loop are caught per-worker and do
          NOT kill the other loop.
        """
        synth_done_event: asyncio.Event  = asyncio.Event()
        rollout_queue:    asyncio.Queue  = asyncio.Queue()

        # Pre-fill with tasks that are synth-complete but have no rollout yet
        existing_gaps = find_rollout_gaps(
            self.synth_data_dir, self.rollout_dir, self.model_config_name,
        )
        for t in existing_gaps:
            await rollout_queue.put(t)
        if existing_gaps:
            logger.info(
                "[run] Pre-filled %d existing rollout gap(s) into queue",
                len(existing_gaps),
            )

        synth_results:   list = []
        rollout_results: list = []

        # Build synth pipeline (lazy imports happen inside)
        pipeline = _build_synth_pipeline(self.synth_data_dir)

        # Build seed queue for synth workers
        tasks = collect_synth_queue(
            self.source_configs,
            self.seed_data_dir,
            self.synth_data_dir,
            max_tasks=self.max_tasks,
            skip_timeout=self.skip_timeout,
        )
        seed_queue: asyncio.Queue = asyncio.Queue()
        for t in tasks:
            await seed_queue.put(t)
        for _ in range(self.n_synth_workers):
            await seed_queue.put(None)   # sentinels

        pause_event = asyncio.Event()
        pause_event.set()  # start unpaused

        async def _synth_loop() -> None:
            workers = [
                _synth_worker(
                    i, seed_queue, pipeline, synth_results,
                    stage=self.synth_stage,
                    synth_data_dir=self.synth_data_dir,
                    rollout_queue=rollout_queue,
                    pause_event=pause_event,
                )
                for i in range(self.n_synth_workers)
            ]
            try:
                await asyncio.gather(*workers, return_exceptions=True)
            finally:
                synth_done_event.set()
                logger.info("[synth_loop] finished — signalled rollout consumers")

        rollout_consumers = [
            _rollout_consumer_worker(
                i, rollout_queue, synth_done_event,
                self.rollout_dir,
                self.model_config_name,
                self.model_url,
                self.model_name,
                self.n_trajs,
                rollout_results,
                agent_config=self.agent_config,
            )
            for i in range(self.n_rollout_workers)
        ]

        await asyncio.gather(_synth_loop(), *rollout_consumers, return_exceptions=True)
        return {"synth_results": synth_results, "rollout_results": rollout_results}

    async def run_unified_pipeline(self) -> list:
        """Unified pipeline: idea-only → full generation.

        Early ditching:
        - Stage 1: Run idea-only, agent evaluates and ditches unsuitable tasks
        - Stage 2: Skipped (datasets pre-filtered before pipeline start)
        - Stage 3: For PASS tasks only, run full generation
        """
        import csv
        import os

        logger = logging.getLogger(__name__)

        # ===== STAGE 1: Idea-only evaluation =====
        logger.info("=" * 70)
        logger.info("UNIFIED PIPELINE - STAGE 1: Agent Evaluation (Early Ditch)")
        logger.info("=" * 70)

        self.synth_stage = "idea-only"
        synth_results_1 = await self.run_synth_loop()

        logger.info("Stage 1 complete. Checking verdicts...")

        # ===== STAGE 2: Skipped (datasets pre-filtered before pipeline) =====
        logger.info("=" * 70)
        logger.info("UNIFIED PIPELINE - STAGE 2: Skipped (datasets already validated & sized before pipeline)")
        logger.info("=" * 70)

        # ===== STAGE 3: Full generation (downloaded tasks only) =====
        logger.info("=" * 70)
        logger.info("UNIFIED PIPELINE - STAGE 3: Full Task Generation")
        logger.info("=" * 70)

        self.synth_stage = "full"
        synth_results_2 = await self.run_synth_loop()

        logger.info("=" * 70)
        logger.info("UNIFIED PIPELINE COMPLETE")
        logger.info("=" * 70)

        return synth_results_1 + synth_results_2

    def print_summary(self, results: dict) -> None:
        synth   = results.get("synth_results", [])
        rollout = results.get("rollout_results", [])

        passed   = sum(1 for _, r in synth if r and getattr(r, "verdict", None) == "PASS")
        ditched  = sum(1 for _, r in synth if r and getattr(r, "verdict", None) == "DITCH")
        failed   = sum(1 for _, r in synth if r and getattr(r, "verdict", None) not in ("PASS", "DITCH"))
        skipped  = sum(1 for _, r in synth if r is None)
        print(f"\n{'=' * 60}")
        print(f"SYNTH:   {len(synth)} processed — {passed} PASS / {ditched} DITCH / {failed} FAIL / {skipped} skip")

        rolled    = sum(1 for _, r in rollout if r is not None)
        r_skipped = sum(1 for _, r in rollout if r is None)
        print(f"ROLLOUT: {len(rollout)} processed — {rolled} done / {r_skipped} error")
        print(f"{'=' * 60}\n")
