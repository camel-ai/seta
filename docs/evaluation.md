# Evaluation

```bash
# 1. Start model server
python -m sglang.launch_server --model Qwen/Qwen3-8B --port 30000

# 2. Run eval (dataset auto-downloads if not present)
python scripts/evaluation/eval.py --config scripts/evaluation/configs/eval_default.yaml
```

Override config on the command line:

```bash
python scripts/evaluation/eval.py \
    --config scripts/evaluation/configs/eval_default.yaml \
    terminal_env.model.model_type=Qwen/Qwen3-32B \
    terminal_env.model.url=http://localhost:30000/v1 \
    workers=8 \
    dataset=terminal-bench-2.0
```

For remote Docker execution, see [slot_pool.md](slot_pool.md).

## Pre-training evaluation and dataset filtering

Before launching a long RL training run, run a baseline eval over the
training dataset and use the results to drop tasks that are uninformative
(every trajectory already passes, or none ever passes). The trained model
then only sees tasks where the reward signal can move.

**1. Run a baseline eval** with `n_trajs >= 4` against the training
dataset (any eval entrypoint that writes `trials/` and `failed/` works).

**2. Collect results into `evaluated_tasks.csv`** — the wide-format file
the filter consumes (`task_id, traj_0, ..., traj_{N-1}`):

```bash
# Single eval run:
python -m seta_env.utils.collect_results /path/to/eval_run/trials \
    --output /path/to/eval_run

# Multiple resumed runs (also produces success.csv / failed.csv):
python -m seta_env.utils.collect_results --merge \
    /path/to/eval_run /path/to/eval_run_resume \
    --output /path/to/merged
```

Both modes emit `evaluated_tasks.csv` in the output dir.

**3. Generate a `task_filter.txt`** in the dataset folder. The dataset
loader auto-detects this file and skips everything not listed.

```bash
python -m seta_env.dataset.filter_tasks \
    --csv /path/to/merged/evaluated_tasks.csv \
    --dataset dataset/seta-env-v2 \
    --drop-missing --drop-too-hard --drop-too-easy
```

Flags (combinable):
- `--drop-missing` — task not present in the CSV (or all trajectories blank)
- `--drop-too-hard` — `max(reward) == 0.0` across trajectories
- `--drop-too-easy` — `min(reward) == 1.0` across trajectories

The script writes `<dataset>/task_filter.txt` (one `task_id` per line).
Subsequent calls to `load_harbor_dataset(<dataset>)` automatically honor
it. Delete the file to disable filtering.
