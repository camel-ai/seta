"""seta_env evaluation — one-click runner.

Usage:
    python scripts/evaluation/eval.py \\
        --config scripts/evaluation/configs/eval_default.yaml \\
        terminal_env.model.model_type=Qwen/Qwen3-32B \\
        terminal_env.model.url=http://localhost:30000/v1 \\
        workers=8

Options:
    --config   Path to YAML config (required).
    --dry-run  Print resolved config and exit.
    --tasks    Space-separated task names to run (default: all).
    --resume   One or more paths to previous trial dirs. Skips tasks that
               already completed in any of them (have n_trajs dirs each
               with verifier/ctrf.json). Outputs to <first_dir>_resume.
"""

import argparse
import asyncio
import glob
import os
import re
import sys
import threading
from dataclasses import replace
from datetime import datetime
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_REPO_ROOT = Path(__file__).resolve().parents[2]

from seta_env.utils.configs import (
    EvalConfig, load_eval_config, save_config, load_tasks,
)
from seta_env.utils.results import collect_and_summarize, print_summary
from seta_env.orchestrators.grpo_rollout import GRPORollout


def _find_completed_tasks(trial_dir: str, n_trajs: int) -> set[str]:
    """Return task names that completed all trajectories in a previous run.

    A task is considered done if it has exactly n_trajs trial directories
    under <trial_dir>/trials/, each containing verifier/ctrf.json.
    """
    trials_root = os.path.join(trial_dir, "trials")
    if not os.path.isdir(trials_root):
        return set()

    # Map task_name -> count of completed trajectories
    # Trial dirs are named <task_name>_t<i>_<hash>
    completed_count: dict[str, int] = {}
    for entry in os.scandir(trials_root):
        if not entry.is_dir():
            continue
        ctrf = os.path.join(entry.path, "verifier", "ctrf.json")
        if not os.path.isfile(ctrf):
            continue
        # Extract task_name: everything before _t<digit>_<hash>
        match = re.match(r"^(.+)_t\d+_[0-9a-f]+$", entry.name)
        if match:
            task_name = match.group(1)
            completed_count[task_name] = completed_count.get(task_name, 0) + 1

    return {name for name, count in completed_count.items() if count >= n_trajs}


def _resume_trial_dir(original: str) -> str:
    """Return a new trial dir path like <original>_resume, _resume_2, etc."""
    candidate = f"{original}_resume"
    if not os.path.exists(candidate):
        return candidate
    i = 2
    while os.path.exists(f"{original}_resume_{i}"):
        i += 1
    return f"{original}_resume_{i}"


def _auto_trial_name(cfg: EvalConfig) -> EvalConfig:
    if not cfg.trial_name:
        slug = cfg.terminal_env.model.model_type.rstrip("/").split("/")[-1] if cfg.terminal_env.model else "nomodel"
        cfg = replace(cfg, trial_name=f"{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    return cfg

async def main_async(cfg: EvalConfig, task_filter: list | None,
                     resume_dirs: list[str] | None = None) -> int:
    tasks = load_tasks(cfg, repo_root=str(_REPO_ROOT))
    if not tasks:
        print("[ERROR] No tasks found.", flush=True); return 1
    if task_filter:
        tasks = [t for t in tasks if t["task_name"] in task_filter]

    if resume_dirs:
        completed: set[str] = set()
        for d in resume_dirs:
            done = _find_completed_tasks(d, cfg.n_trajs)
            print(f"[resume] {d}: {len(done)} completed", flush=True)
            completed |= done
        before = len(tasks)
        tasks = [t for t in tasks if t["task_name"] not in completed]
        print(f"[resume] union: {len(completed)} done, {before - len(tasks)} skipped, {len(tasks)} to run", flush=True)
        if not tasks:
            print("[resume] All tasks completed. Nothing to do.", flush=True); return 0

    print(f"[dataset] {len(tasks)} tasks loaded (rank {cfg.rank}/{cfg.world_size})", flush=True)

    if resume_dirs:
        trial_dir = _resume_trial_dir(resume_dirs[0])
    else:
        trial_dir = os.path.join(cfg.output_dir, cfg.experiment_name, cfg.trial_name)
    trial_root = os.path.join(trial_dir, "trials")
    failed_dir = os.path.join(trial_dir, "failed")
    os.makedirs(trial_root, exist_ok=True)
    save_config(cfg, trial_dir)

    # Set trial_root in the terminal_env config
    te_cfg = cfg.terminal_env
    te_cfg.runtime.trial_root = trial_root

    sem = asyncio.Semaphore(cfg.workers)
    total, done = len(tasks), 0
    accumulated, stop = [], threading.Event()

    def _collect_loop():
        while not stop.wait(60):
            if accumulated:
                collect_and_summarize(list(accumulated), trial_dir, cfg)

    threading.Thread(target=_collect_loop, daemon=True).start()

    async def run_one(task):
        nonlocal done
        async with sem:
            print(f"[START] {task['task_name']}", flush=True)
            rollout = GRPORollout(cfg=te_cfg)
            try:
                results = await rollout.run(task, n_trajs=cfg.n_trajs)
            except Exception as exc:
                import traceback as _tb
                os.makedirs(failed_dir, exist_ok=True)
                Path(failed_dir, f"{task['task_name']}.txt").write_text(_tb.format_exc())
                results = [( {"task_name": task["task_name"], "uid": "", "traj_i": 0,
                               "reward": None, "error_info": {"stage": "rollout", "error_message": str(exc)},
                               "timings": {}, "evaluation": {}, "agent_summary": {}}, None )]
            done += 1
            accumulated.extend(results)
            print(f"[DONE]  {task['task_name']}  rewards={[r for _,r in results]}  ({done}/{total})", flush=True)
            return results

    nested  = await asyncio.gather(*[run_one(t) for t in tasks])
    stop.set()
    results = [item for task_res in nested for item in task_res]

    summary = collect_and_summarize(results, trial_dir, cfg)
    print_summary(summary)
    print(f"Summary → {os.path.join(trial_dir, 'summary.json')}", flush=True)
    print(f"Trials  → {trial_root}", flush=True)
    return 1 if summary["error_count"] > 0 else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="seta_env evaluation")
    parser.add_argument("--config", required=True, help="YAML config file path")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    parser.add_argument("--tasks", nargs="*", default=None, help="Task names to run")
    parser.add_argument("--resume", nargs="+", default=None, metavar="TRIAL_DIR",
                        help="Resume from one or more previous trial dirs, "
                             "skipping tasks completed in any of them")
    script_args, hydra_overrides = parser.parse_known_args()

    cfg, _ = load_eval_config(["--config", script_args.config] + hydra_overrides, EvalConfig)
    cfg = _auto_trial_name(cfg)

    if script_args.dry_run:
        m = cfg.terminal_env.model
        print(f"model={m.model_platform}/{m.model_type}  url={m.url}" if m else "model=external")
        print(f"dataset={cfg.dataset}  workers={cfg.workers}  n_trajs={cfg.n_trajs}")
        if script_args.resume:
            completed: set[str] = set()
            for d in script_args.resume:
                done = _find_completed_tasks(d, cfg.n_trajs)
                print(f"[resume] {d}: {len(done)} completed")
                completed |= done
            print(f"[resume] union: {len(completed)} unique completed tasks")
        print("[dry-run] Exiting.", flush=True); sys.exit(0)

    sys.exit(asyncio.run(main_async(cfg, task_filter=script_args.tasks,
                                    resume_dirs=script_args.resume)))


if __name__ == "__main__":
    main()
