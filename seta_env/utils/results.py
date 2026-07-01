"""Results collection and summary utilities for seta_env evaluations."""

import csv
import json
import os
import statistics
from typing import Any


def collect_and_summarize(results: list, output_dir: str, cfg: Any) -> dict:
    """Aggregate run results, write summary.json and results.csv, return summary dict.

    Args:
        results:    Flat list of (run_info, reward) tuples from GRPORollout.
        output_dir: Directory to write summary.json and results.csv.
        cfg:        EvalConfig instance for metadata fields.

    Returns:
        Summary dict with pass_ratio, reward stats, error count, etc.
    """
    rows = []
    for run_info, reward in results:
        summary = run_info.get("agent_summary") or {}
        error_info = run_info.get("error_info") or {}
        rows.append({
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
        })

    valid_rewards = [r["reward"] for r in rows if r["reward"] is not None]
    n_total  = len(rows)
    n_errors = sum(1 for r in rows if r["error"])
    n_pass   = sum(1 for r in rows if r["reward"] == 1.0)

    summary = {
        "trial_name":      cfg.trial_name,
        "experiment_name": cfg.experiment_name,
        "model_platform":  cfg.terminal_env.model.model_platform if cfg.terminal_env.model else None,
        "model_type":      cfg.terminal_env.model.model_type if cfg.terminal_env.model else None,
        "dataset":         cfg.dataset,
        "total_trajs":     n_total,
        "error_count":     n_errors,
        "pass_count":      n_pass,
        "pass_ratio":      round(n_pass / n_total, 4) if n_total else 0.0,
        "reward": {
            "mean": round(statistics.mean(valid_rewards),  4) if valid_rewards else None,
            "std":  round(statistics.stdev(valid_rewards), 4) if len(valid_rewards) > 1 else 0.0,
            "min":  min(valid_rewards) if valid_rewards else None,
            "max":  max(valid_rewards) if valid_rewards else None,
        },
        "agent": {
            "agent":                 cfg.terminal_env.agent.agent,
            "prompt":                cfg.terminal_env.agent.prompt,
            "max_total_tokens":      cfg.terminal_env.agent.max_total_tokens,
            "max_completion_tokens": cfg.terminal_env.agent.max_completion_tokens,
            "max_iteration":         cfg.terminal_env.agent.max_iteration,
        },
        "eval": {
            "n_trajs": cfg.n_trajs,
            "workers": cfg.workers,
        },
    }

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    if rows:
        csv_path = os.path.join(output_dir, "results.csv")
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return summary


def print_summary(summary: dict) -> None:
    """Print a formatted evaluation summary to stdout."""
    r = summary["reward"]
    mean_str = f"{r['mean']:.4f}" if r["mean"] is not None else "N/A"
    std_str  = f"{r['std']:.4f}"  if r["std"]  is not None else "N/A"
    min_str  = f"{r['min']:.4f}"  if r["min"]  is not None else "N/A"
    max_str  = f"{r['max']:.4f}"  if r["max"]  is not None else "N/A"
    ratio    = summary["pass_ratio"] * 100

    print(flush=True)
    print("=" * 64, flush=True)
    print("  EVAL SUMMARY", flush=True)
    print("=" * 64, flush=True)
    print(f"  Trial:         {summary['trial_name']}", flush=True)
    print(f"  Platform:      {summary['model_platform']}  |  model: {summary['model_type']}", flush=True)
    print(f"  Trajectories:  {summary['total_trajs']}", flush=True)
    print(f"  Pass (r=1.0):  {summary['pass_count']} / {summary['total_trajs']}"
          f"  ({ratio:.1f}%)", flush=True)
    print(f"  Mean reward:   {mean_str}  ±  {std_str}"
          f"  [min={min_str}  max={max_str}]", flush=True)
    print(f"  Errors:        {summary['error_count']}", flush=True)
    print("=" * 64, flush=True)
    print(flush=True)
