# SETA: Scaling Environments for Terminal Agents


![SETA](assets/TerminalAgent.jpg)

***Designing resilient toolkits and scalable RL environments for CAMEL terminal agents***

[![🤗 Dataset](https://img.shields.io/badge/🤗%20Dataset-camel--ai%2FSETA--Env-yellow)](https://huggingface.co/datasets/camel-ai/SETA-Env) [![🤗 Model](https://img.shields.io/badge/🤗%20Model-Qwen3--8B--SETA--Env--RL-blue)](https://huggingface.co/camel-ai/Qwen3-8B-SETA-Env-RL) [![arXiv Paper](https://img.shields.io/badge/arXiv-2607.10891-b31b1b.svg)](https://arxiv.org/abs/2607.10891) [![SETA Blog](https://img.shields.io/badge/Notion-SETA%20Blog-black?logo=notion)](https://eigent-ai.notion.site/SETA-Scaling-Environments-for-Terminal-Agents-2d2511c70ba280a9b7c0fe3e7f1b6ab8?source=copy_link)

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

RL training of terminal agents with the [Miles](https://github.com/radixark/miles) framework.
Rollouts run through the Harbor agent server (Harbor's Terminus-2 or CAMEL agent in Daytona, Modal,
GKE or Docker sandboxes) or through the seta env_service. Each example folder has a step-by-step
README covering the container, Ray cluster, model preparation, sandboxes, dataset and launch:

| Example | Model | Algorithm |
|---|---|---|
| [deepseek_v4_grpo](scripts/miles/examples/deepseek_v4_grpo/README.md) | DeepSeek-V4-Flash | GRPO |
| [glm47_flash_grpo](scripts/miles/examples/glm47_flash_grpo/README.md) | GLM-4.7-Flash | GRPO |
| [glm5_2_lora_grpo](scripts/miles/examples/glm5_2_lora_grpo/README.md) | GLM-5.2 | LoRA GRPO |
| [glm5_2_lora_ppo](scripts/miles/examples/glm5_2_lora_ppo/README.md) | GLM-5.2 | LoRA PPO |
| [inkling_grpo](scripts/miles/examples/inkling_grpo/README.md) | Inkling-Small | GRPO |
| [qwen3_8_27b_grpo](scripts/miles/examples/qwen3_8_27b_grpo/README.md) | Qwen3.8-27B | GRPO |
| [qwen3_8_27b_ppo](scripts/miles/examples/qwen3_8_27b_ppo/README.md) | Qwen3.8-27B | PPO |

```bash
cd scripts/miles/examples/qwen3_8_27b_grpo
cp env.example .env            # fill in cluster, model, data, sandbox and W&B settings
DRY_RUN=1 bash run_harbor_terminus2.sh   # print the resolved command
bash run_harbor_terminus2.sh
```

See **[scripts/miles/README.md](scripts/miles/README.md)** for the index, rollout paths and shared code.

## Docs

- [Configuration](docs/configuration.md) — what to change (model, dataset, runtime) and what to leave alone
- [Dataset](docs/dataset.md) — download and register datasets
- [Evaluation](docs/evaluation.md) — run eval with local or remote Docker
- [Slot Pool Service](docs/slot_pool.md) — distribute environments across remote nodes
- [Env Service](docs/env_service.md) — remote TerminalEnvironment execution on CPU servers
- [Results](docs/results.md) — what each evaluation records and what the fields mean
- [Training](docs/training.md) — AReaL RL training
- [Miles Training](scripts/miles/README.md) — Miles RL training examples (DeepSeek-V4, GLM-4.7-Flash, GLM-5.2, Inkling, Qwen3.8) via the Harbor agent server or env_service

## Experiments

- [Experiments](docs/experiments.md) — log of training and evaluation runs

## Acknowledgements

The Miles-based RL training pipeline (`scripts/miles/`, the Harbor agent-server and seta_env session-server wiring, and the
sandbox integrations) was built in collaboration with the **RadixArk miles team**. Thank
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
