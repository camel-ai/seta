# Configuration

## Overview

Every eval config has two parts:

```yaml
# 1. TerminalEnv — HOW to run each task (model, agent, runtime)
terminal_env:
  model:    { ... }    # which LLM to use
  agent:    { ... }    # prompt, tool list, iteration limits
  runtime:  { ... }    # local docker or remote slot pool
  env:      { ... }    # reward function, timeouts

# 2. Eval loop — WHAT to run (dataset, parallelism, output)
n_trajs: 8
workers: 3
dataset: seta-env-v2
output_dir: outputs/eval
```

Between runs you typically only change **model** and **dataset**. Everything else stays the same.

## What to change

### Switch model

```bash
python scripts/evaluation/eval.py \
    --config scripts/evaluation/configs/eval_default.yaml \
    terminal_env.model.model_type=Qwen/Qwen3-32B \
    terminal_env.model.url=http://localhost:30000/v1
```

Or edit the YAML directly:

```yaml
terminal_env:
  model:
    model_type: Qwen/Qwen3-32B          # ← change this
    url: http://localhost:30000/v1       # ← and this
```

### Switch dataset

Use any label from `seta_env/dataset/datasets.yaml`. Auto-downloads on first use.

```bash
python scripts/evaluation/eval.py \
    --config scripts/evaluation/configs/eval_default.yaml \
    dataset=terminal-bench-2.0
```

Or use a local path:

```bash
dataset=/data/my-custom-dataset
```

### Switch between local Docker and remote slot pool

Two example configs are provided — identical except for `runtime`:

| Config | Runtime | When to use |
|--------|---------|-------------|
| [eval_default.yaml](../scripts/evaluation/configs/eval_default.yaml) | `docker` (local) | Single machine, no setup needed |
| [eval_remote.yaml](../scripts/evaluation/configs/eval_remote.yaml) | `remote_docker` (slot pool) | Multiple nodes, requires [slot pool service](slot_pool.md) running |

The only difference:

```yaml
# eval_default.yaml (local)
runtime:
  env_type: docker

# eval_remote.yaml (remote)
runtime:
  env_type: remote_docker
  scheduler_url: "http://127.0.0.1:8000"
  node_api_key: harbor-node-dev-key
```

Or override on the command line without a separate config:

```bash
python scripts/evaluation/eval.py \
    --config scripts/evaluation/configs/eval_default.yaml \
    terminal_env.runtime.env_type=remote_docker \
    terminal_env.runtime.scheduler_url=http://127.0.0.1:8000
```

## Model: internal vs external

The model can be configured in two ways:

- **Internal** (standalone eval): Set `terminal_env.model` in the YAML. TerminalEnvironment creates the model via ModelFactory from the config fields (platform, url, type).
- **External** (AReaL training): Set `terminal_env.model: null` in the YAML. The AReaL workflow builds the model externally and passes the instance into GRPORollout at runtime.

You don't need to think about this for standalone eval — just fill in the model config. It only matters when integrating with a training framework like AReaL that manages its own inference engine.

## What NOT to change (usually)

- `agent` — prompt, tool list, iteration limits. These define the agent's behavior and should stay consistent across evaluations.
- `env` — reward function and timeouts. Change only if you're adding a new reward function or debugging timeout issues.
- `terminal_env.model.model_platform` — stays `sglang` unless you're using a different inference backend.

## Config reference

### terminal_env.model

| Field | Description | Example |
|-------|-------------|---------|
| `model_platform` | Inference backend | `sglang` |
| `model_type` | HuggingFace model path | `Qwen/Qwen3-8B` |
| `url` | Inference server URL | `http://localhost:30000/v1` |

### terminal_env.runtime

| Field | Description | Default |
|-------|-------------|---------|
| `env_type` | `docker` or `remote_docker` | `docker` |
| `scheduler_url` | Slot pool scheduler (remote_docker only) | `http://127.0.0.1:8000` |
| `node_api_key` | Node manager API key (remote_docker only) | `harbor-node-dev-key` |

### Eval loop

| Field | Description | Default |
|-------|-------------|---------|
| `dataset` | Dataset label or local path | `seta-env-v2` |
| `n_trajs` | Trajectories per task (pass@k) | `8` |
| `workers` | Max concurrent tasks | `3` |
| `output_dir` | Results output directory | `outputs/eval` |
