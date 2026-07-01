# Results

Each evaluation writes to `outputs/eval/<experiment>/<trial>/`:

- `summary.json` — aggregated stats (pass ratio, reward mean/std, error count)
- `results.csv` — one row per trajectory with the fields below
- `evaluated_tasks.csv` — wide format `task_id, traj_0, ..., traj_{N-1}` consumed by [`seta_env.dataset.filter_tasks`](../seta_env/dataset/filter_tasks.py); see [evaluation.md](evaluation.md#pre-training-evaluation-and-dataset-filtering)
- `trials/` — per-task agent logs and perf traces

## Per-trajectory fields (results.csv)

| Field | Meaning |
|-------|---------|
| `task_name` | Task identifier |
| `traj_i` | Trajectory index (0 to n_trajs-1) |
| `uid` | Unique session ID for this trajectory |
| `reward` | Reward from verifier (0.0–1.0, null on error) |
| `error` | Whether this trajectory hit an error |
| `error_stage` | Which stage failed: `1_reset_env`, `2_run_agent`, `3_evaluate`, `4_calculate_reward` |
| `error_message` | Error details |
| `iteration_count` | How many agent turns ran |
| `termination_reason` | Why the agent stopped: `task_finished`, `max_iteration_reached`, `max_tokens_exceeded`, `max_parse_errors`, `step_timeout` |
| `total_tool_calls` | Total tool calls made across all turns |
| `max_parallel_tool_call` | Largest batch of parallel tool calls in a single turn |
| `parse_error_count` | Times the model output couldn't be parsed as a tool call |
| `prompt_tokens` | Cumulative prompt tokens sent to the model |
| `completion_tokens` | Cumulative completion tokens generated |
| `total_tokens` | prompt_tokens + completion_tokens |
