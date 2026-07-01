# Training

Training uses the [AReaL](https://github.com/inclusionAI/AReaL) framework for distributed RL training. All scripts are under `scripts/areal/`.

## AReaL Evaluation (inference only)

Tests the AReaL codepath without gradient updates:

```bash
python -m areal.launcher.local \
    scripts/areal/eval.py \
    --config scripts/areal/configs/config_eval.yaml
```

## AReaL RL Training

```bash
python -m areal.launcher.local \
    scripts/areal/rl_train.py \
    --config scripts/areal/configs/config_eval.yaml
```

The same `config_eval.yaml` can be used for both eval and training. For training-specific options (e.g. `filter_uniform_reward`, `async_training`), override on the command line or create a separate training config.

## Config

AReaL configs extend the seta-env `terminal_env` block (same structure as [evaluation configs](configuration.md)) with AReaL-specific settings (actor, sglang, cluster, etc.). The `terminal_env` section is identical — see [configuration.md](configuration.md) for what to change.

Key AReaL-specific fields:

| Field | Description |
|-------|-------------|
| `actor.path` | Model to train (HuggingFace path) |
| `cluster.n_gpus_per_node` | GPUs per node |
| `allocation_mode` | GPU strategy (e.g. `sglang:d4p1t1`) |
| `train_dataset.path` | Dataset label or path |
