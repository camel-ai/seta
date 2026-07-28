# SETA: Scaling Environments for Terminal Agents


![SETA](assets/TerminalAgent.jpg)

***Designing resilient toolkits and scalable RL environments for CAMEL terminal agents***

[![🤗 Dataset](https://img.shields.io/badge/🤗%20Dataset-camel--ai%2FSETA--Env-yellow)](https://huggingface.co/datasets/camel-ai/SETA-Env) [![🤗 Model](https://img.shields.io/badge/🤗%20Model-Qwen3--8B--SETA--Env--RL-blue)](https://huggingface.co/camel-ai/Qwen3-8B-SETA-Env-RL)

----

## Installation

```bash
git clone --recurse-submodules https://github.com/camel-ai/seta.git
cd seta
bash setup.sh
```

## Quick start

Three runtime options — choose one:

**Option A: Local Docker** (single machine, no extra setup)
```bash
# uses eval_default.yaml (env_type: docker)
--config scripts/evaluation/configs/eval_default.yaml
```

**Option B: Remote Docker** (multiple nodes via [slot pool service](docs/slot_pool.md))
```bash
# 1. start slot pool service first
bash seta_env/runtimes/slot_pool_service/start.sh --dataset seta-env-v2
# 2. uses eval_remote.yaml (env_type: remote_docker)
--config scripts/evaluation/configs/eval_remote.yaml
```

**Option C: Env Service** (remote CPU servers for agent execution, see [env_service](docs/env_service.md))
```bash
# 1. deploy env_service to CPU servers + start scheduler
GH_TOKEN=ghp_xxx HF_TOKEN=hf_xxx bash seta_env/services/start.sh --dataset seta-env-v2
# 2. run eval via AReaL launcher
python -m areal.launcher.local scripts/areal/eval_env_service.py \
    --config scripts/areal/configs/config_eval_env_service_seta_v2.yaml
```

### Evaluation

```bash
# start model server
python -m sglang.launch_server --model Qwen/Qwen3-8B --port 30000

# run eval (dataset auto-downloads on first use)
python scripts/evaluation/eval.py --config scripts/evaluation/configs/eval_default.yaml

# sweep across models and datasets
python scripts/evaluation/sweep_eval.py scripts/evaluation/configs/sweep.yaml

# results → outputs/eval/<experiment>/<trial>/summary.json, results.csv
```

### Training (AReaL)

```bash
# RL training
python -m areal.launcher.local \
    scripts/areal/rl_train.py \
    --config scripts/areal/configs/config_eval.yaml

# eval only (no gradient updates, single GPU)
python -m areal.launcher.local \
    scripts/areal/eval.py \
    --config scripts/areal/configs/config_eval.yaml \
    allocation_mode=sglang:d1p1t1+eval

# results → outputs/areal/experiments/<experiment>/<trial>/
```

### Training (Miles)

RL (GRPO) training on the seta_env env service with the
[miles](https://github.com/radixark/miles) framework — disaggregated, session-server rollout with
Daytona sandboxes. Two models are wired up end-to-end:

```bash
# 1. one-time: download + convert the model to a torch-dist checkpoint
python scripts/miles/run_glm47_flash_seta_session_server.py prepare     # GLM-4.7-Flash
python scripts/miles/run_deepseek_v4_seta_session_server.py prepare     # DeepSeek-V4-Flash-FP8

# 2. launch training (restarts env_service + submits the Ray job across the cluster)
bash scripts/miles/run_glm47_flash_seta_session_server.sh               # GLM-4.7-Flash
bash scripts/miles/run_deepseek_v4_seta_session_server.sh               # DeepSeek-V4-Flash-FP8

# results → /data/training_runs/<run>/ (checkpoints, trials, wandb, ray_job.log)
```

Requires an 8-node Ray cluster, `DAYTONA_*` / `WANDB_API_KEY` / `HF_TOKEN` in `~/.bashrc`, and a task
dataset registered under `DATASET_ROOT`. Full setup, config layout, dataset format, and tuning are in
**[scripts/miles/README.md](scripts/miles/README.md)**.

## Docs

- [Configuration](docs/configuration.md) — what to change (model, dataset, runtime) and what to leave alone
- [Dataset](docs/dataset.md) — download and register datasets
- [Evaluation](docs/evaluation.md) — run eval with local or remote Docker
- [Slot Pool Service](docs/slot_pool.md) — distribute environments across remote nodes
- [Env Service](docs/env_service.md) — remote TerminalEnvironment execution on CPU servers
- [Results](docs/results.md) — what each evaluation records and what the fields mean
- [Training](docs/training.md) — AReaL RL training
- [Miles Training](scripts/miles/README.md) — miles RL training (GLM-4.7-Flash, DeepSeek-V4) via env_service + Daytona

## Experiments

- [Experiments](docs/experiments.md) — log of training and evaluation runs

## Acknowledgements

The miles-based RL training pipeline (`scripts/miles/`, the seta_env session-server wiring, and the
Daytona environment integration) was built in collaboration with the **RadixArk miles team**. Thank
you for the [miles](https://github.com/radixark/miles) framework and for the support throughout.

# Citation

Please cite both the SETA project and the arXiv paper when using this work.

**SETA project**

```
@misc{seta,
  author    = {Qijia Shen and Jay Rainton and Aznaur Aliev and Ahmed Awelkair and Boyuan Ma and Zhiqi (Julie) Huang and Yuzhen Mao and Wendong Fan and Philip Torr and Bernard Ghanem and Changran Hu and Urmish Thakker and Guohao Li},
  title     = {{SETA: Scaling Environments for Terminal Agents}},
  year      = {2026},
  month     = jan,
  url       = {https://github.com/camel-ai/seta},
  note      = {Blog: \url{https://eigent-ai.notion.site/SETA-Scaling-Environments-for-Terminal-Agents-2d2511c70ba280a9b7c0fe3e7f1b6ab8}}
}
```

**SETA paper**

```
@misc{shen2026seta,
  author        = {Qijia Shen and Zhiqi Huang and Vamsidhar Kamanuru and Aznaur Aliev and Jay Rainton and Ahmed Awelkair and Zhichen Zeng and Jiajun Li and Shi Dong and Yueming Yuan and Boyuan Ma and Qizheng Zhang and Jiwei Fu and Yuzhen Mao and Wendong Fan and Ping Nie and Philip Torr and Bernard Ghanem and Changran Hu and Jonathan Lingjie Li and Urmish Thakker and Guohao Li},
  title         = {{SETA: Scaling Environments for Terminal Agents}},
  year          = {2026},
  month         = jul,
  eprint        = {2607.10891},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2607.10891}
}
```
