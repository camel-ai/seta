# Base seta-env-v2-03-30 Experiment

**Date:** 2026-03-30
**Model:** Qwen/Qwen3-8B
**Dataset:** [camel-ai/seta-env-v2](https://huggingface.co/datasets/camel-ai/seta-env-v2)
**Hardware:** 1 node × 8 GPUs

Base RL training and evaluation recipe on `seta-env-v2`. Follow the steps
in order — each step assumes the previous one finished successfully.

---

## Step 1 — Download the dataset

```bash
python -m seta_env.dataset.download seta-env-v2
```

This downloads into `dataset/seta-env-v2/`. If the dataset is gated, set
`HF_TOKEN` first:

```bash
export HF_TOKEN="your_token_here"
```

To list other available datasets: `python -m seta_env.dataset.download --list`.

---

## Step 2 — Pre-build Docker images

Every task in `seta-env-v2` runs inside its own Docker image. Building
those on-demand during eval/training serializes a lot of work behind the
GPU and leaks build cache. Pre-build them once up front, then prune the
cache.

```bash
# Build all task images in parallel (tune --concurrency to your machine)
python -m seta_env.utils.prebuild_docker_images dataset/seta-env-v2 \
    --concurrency 8

# Free disk by clearing the build cache (images themselves are kept)
docker builder prune -a -f
```

Run this whenever the dataset is added, updated, or extended (e.g. if
Step 3 / Option C below adds new tasks, re-run prebuild for them).

---

## Step 3 — Filter the dataset (recommended)

RL training learns nothing from tasks that always fail or always pass.
Drop those before training. After this step, `dataset/seta-env-v2/` will
contain a `task_filter.txt` that the loader picks up automatically — no
config changes needed. Delete that file later to disable filtering.

`dataset/seta-env-v2/` currently contains **2350 tasks**. A pre-computed
`evaluated_tasks.csv` covering **1482** of them is available here:

> **[Download evaluated_tasks.csv (Google Drive)](https://drive.google.com/file/d/1dx9Ls-Obqn45JphSQrXcUTh3b2yv2Fhl/view?usp=share_link)**

Save it to `dataset/evaluated_tasks.csv`, then pick **one** of the three
options below. Since the baseline CSV is missing ~868 tasks,
**Option C (delta evaluation)** is recommended — it reuses the 1482
already-covered tasks and only spends GPU time on the remainder.

### Option A — Use the pre-computed CSV (fastest, no GPU needed)

```bash
python -m seta_env.dataset.filter_tasks \
    --csv dataset/evaluated_tasks.csv \
    --dataset dataset/seta-env-v2 \
    --drop-missing
```

### Option B — Run a fresh full evaluation

Use this when the baseline CSV is stale or you want a measurement with
your current model/config.

```bash
# 1. Full eval over dataset/seta-env-v2
python -m areal.launcher.local \
    scripts/areal/eval.py \
    --config scripts/areal/configs/config_eval_local_seta_v2.yaml \
    allocation_mode=sglang:d1p1t1+eval

# 2. Collect results into evaluated_tasks.csv
EVAL_OUT=outputs/areal/experiments/camel-terminal_agent-eval/trial0-seta-env-v2-eval-local-nothink-docker
python -m seta_env.utils.collect_results "$EVAL_OUT/trials" --output "$EVAL_OUT"

# 3. Write dataset/seta-env-v2/task_filter.txt
python -m seta_env.dataset.filter_tasks \
    --csv "$EVAL_OUT/evaluated_tasks.csv" \
    --dataset dataset/seta-env-v2 \
    --drop-missing
```

### Option C — Delta evaluation (only tasks missing from the baseline CSV)

Use this to extend the pre-computed CSV with new tasks added to
`dataset/seta-env-v2` since it was generated, without re-running eval on
the 1482 already-covered tasks.

```bash
# 1. Write a TEMPORARY task_filter.txt containing only the missing tasks
python -m seta_env.dataset.filter_tasks \
    --csv dataset/evaluated_tasks.csv \
    --dataset dataset/seta-env-v2 \
    --only-missing

# 2. Run eval — the loader auto-honors task_filter.txt, so eval covers
#    only the missing tasks
python -m areal.launcher.local \
    scripts/areal/eval.py \
    --config scripts/areal/configs/config_eval_local_seta_v2.yaml \
    allocation_mode=sglang:d1p1t1+eval

# 3. Collect the new partial results
EVAL_OUT=outputs/areal/experiments/camel-terminal_agent-eval/trial0-seta-env-v2-eval-local-nothink-docker
python -m seta_env.utils.collect_results "$EVAL_OUT/trials" --output "$EVAL_OUT"

# 4. Overwrite task_filter.txt from the UNION of both CSVs (--csv repeats)
python -m seta_env.dataset.filter_tasks \
    --csv dataset/evaluated_tasks.csv \
    --csv "$EVAL_OUT/evaluated_tasks.csv" \
    --dataset dataset/seta-env-v2 \
    --drop-missing
```

> Other available drop flags: `--drop-too-hard` (max reward = 0 across
> trajectories), `--drop-too-easy` (min reward = 1). See
> [evaluation.md](../evaluation.md#pre-training-evaluation-and-dataset-filtering)
> for details.

---

## Step 4 — Train

Train with the **full-pass bonus** reward function (`pass_ratio_with_bonus`),
which adds **+1.0** when every unit test passes:

```bash
python -m areal.launcher.local \
    scripts/areal/rl_train.py \
    --config scripts/areal/configs/config_train_local_seta_v2_full_pass_bonus.yaml
```

The training run automatically honors `dataset/seta-env-v2/task_filter.txt`
if Step 3 was performed.

### Alternative variant (optional)

| Variant | Config | What changes |
|---------|--------|-------------|
| Parallel-limit + penalty | [`config_train_local_seta_v2_parallel_limit.yaml`](../../scripts/areal/configs/config_train_local_seta_v2_parallel_limit.yaml) | Caps parallel tool calls at **5/turn** (excess rejected with a stub result). Uses `pass_ratio_parallel_penalty` (full-pass bonus + small soft penalty above the threshold). |

---

## Step 5 — Evaluate the trained model

```bash
python -m areal.launcher.local \
    scripts/areal/eval.py \
    --config scripts/areal/configs/config_eval_local_seta_v2.yaml \
    allocation_mode=sglang:d1p1t1+eval
```

Results are written to
`outputs/areal/experiments/camel-terminal_agent-eval/`.

---

## Reference

### Key parameters

| Parameter             | Train                       | Eval                  |
|-----------------------|-----------------------------|-----------------------|
| Model                 | Qwen/Qwen3-8B               | Qwen/Qwen3-8B         |
| Dataset               | dataset/seta-env-v2         | dataset/seta-env-v2   |
| Epochs                | 40                          | 1                     |
| Batch size            | 16                          | 16                    |
| Max iteration         | 30                          | 30                    |
| Max completion tokens | 4096                        | 4096                  |
| Allocation mode       | `sglang:d4p1t1+fsdp:d4p1t1` | `sglang:d4p1t1+eval`  |
| Cluster               | 1 node, 8 GPUs              | 1 node, 8 GPUs        |
| Reward function       | pass_ratio_with_bonus       | pass_ratio            |
| Prompt                | sys_prompt_base             | sys_prompt_base       |
| Thinking              | false                       | false                 |

### Configs

| Purpose                                       | Config file |
|-----------------------------------------------|-------------|
| Training (full-pass bonus, **default**)       | [`config_train_local_seta_v2_full_pass_bonus.yaml`](../../scripts/areal/configs/config_train_local_seta_v2_full_pass_bonus.yaml) |
| Training (parallel-limit + penalty alternative) | [`config_train_local_seta_v2_parallel_limit.yaml`](../../scripts/areal/configs/config_train_local_seta_v2_parallel_limit.yaml) |
| Evaluation                                    | [`config_eval_local_seta_v2.yaml`](../../scripts/areal/configs/config_eval_local_seta_v2.yaml) |
