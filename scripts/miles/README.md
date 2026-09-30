# Miles training for terminal agents

Reinforcement-learning recipes that train LLMs as terminal agents with the
[Miles](https://github.com/radixark/miles) framework. Every example is a self-contained
folder with a step-by-step README (container, Ray cluster, model, sandboxes, dataset, settings,
launch), an `env.example` for your keys and paths, and the launch script(s).

## Examples

| Example | Model | Algorithm | Agent · rollout path | Hardware |
|---|---|---|---|---|
| [deepseek_v4_grpo](examples/deepseek_v4_grpo/README.md) | DeepSeek-V4-Flash (FP8) | GRPO, full fine-tuning | Terminus-2 · Harbor agent server<br>seta CAMEL agent · seta env_service | 8 × 8 H200 |
| [glm47_flash_grpo](examples/glm47_flash_grpo/README.md) | GLM-4.7-Flash | GRPO, full fine-tuning | seta CAMEL agent · seta env_service | 8 × 8 H200 |
| [glm5_2_lora_grpo](examples/glm5_2_lora_grpo/README.md) | GLM-5.2 (744B-A40B) | GRPO, LoRA | Terminus-2 · Harbor agent server | 4 × 8 H200 (or 8) |
| [glm5_2_lora_ppo](examples/glm5_2_lora_ppo/README.md) | GLM-5.2 (744B-A40B) | PPO (LoRA actor + critic) | Terminus-2 · Harbor agent server | 4 × 8 H200 |
| [inkling_grpo](examples/inkling_grpo/README.md) | Inkling-Small (276B MoE) | GRPO, full fine-tuning | CAMEL or Terminus-2 · Harbor agent server | 8 × 8 H200 |
| [qwen3_8_27b_grpo](examples/qwen3_8_27b_grpo/README.md) | Qwen3.8-27B | GRPO, full fine-tuning, 128k context | Terminus-2 · Harbor agent server | 4 × 8 H200 |
| [qwen3_8_27b_ppo](examples/qwen3_8_27b_ppo/README.md) | Qwen3.8-27B | PPO (actor + critic), full fine-tuning | Terminus-2 · Harbor agent server | 4 × 8 H200 |

Each script records the Docker image, Miles commit and (for Harbor) Harbor commit it was run
with, and warns when your checkout differs (`STRICT_PINS=1` makes that an error).

## How a rollout works

Two rollout paths are supported. Both let Miles train on the exact tokens the model generated
(token-in/token-out through Miles' session server) while an agent solves a task in a sandbox and a
verifier scores it.

```
             Ray cluster (Miles: Megatron training + SGLang engines)
                               │  session server (per-trajectory OpenAI-compatible URL)
                               ▼
   ┌─────────────── Harbor agent server ───────────────┐   or   ┌──── seta env_service ────┐
   │ python -m harbor.agent_server                     │        │ seta_env.services         │
   │ agent: Harbor Terminus-2 or CAMEL                 │        │ agent: seta CAMEL agent   │
   │ sandboxes: Daytona · Modal · GKE · Docker         │        │ sandboxes: Daytona        │
   └───────────────────────────────────────────────────┘        └───────────────────────────┘
                               │  verifier reward → Miles reward
```

- **Harbor agent server**: each sample is posted to `POST /run`; Harbor starts the task's
  sandbox (`SANDBOX_BACKEND=daytona|modal|gke|docker`), runs the agent against the session URL,
  runs the task's tests and returns the reward. Tasks are
  [Harbor task directories](https://github.com/harbor-framework/harbor). GKE support
  (cluster creation, sandbox image, scaling) is in `common/gke_cluster.sh`.
- **seta env_service**: the seta environment service (see [docs/env_service.md](../../docs/env_service.md))
  runs seta's CAMEL terminal agent per `POST /step`.

## Layout

```
scripts/miles/
├── README.md                this index
├── cluster.example.yaml     Ray head/worker IPs (alternative to HEAD_IP)
├── common/                  code shared by all examples
├── data/                    prompt-file format, small example files (examples only)
└── examples/<model>_<algorithm>/
    ├── README.md            full setup and launch walkthrough
    ├── env.example          copy to .env and fill in (keys, paths, cluster)
    ├── run_*.sh             launch script(s), one per rollout path / agent
    └── train.py             builds and submits the Miles job
```

`common/`:

| File | Purpose |
|---|---|
| `launcher.sh` | shell helpers used by every run script: settings file, run folder, dry run, Ray, pins, W&B, Harbor install/configure/start |
| `harbor_agent.py` | Miles agent function (`--custom-agent-function-path common.harbor_agent.run`), posts each sample to the Harbor agent server; `abort` cancels surplus trials |
| `harbor_rollout.py` | reward function, infrastructure-failure filters, rollout metrics |
| `harbor_abort.py` | generate wrapper that cancels surplus Harbor trials when a batch is full |
| `harbor_server.sh`, `harbor_install.sh` | start the agent server; install Harbor at a pinned commit |
| `harbor_gke_environment.py`, `gke_cluster.sh`, `gke_tools_image.sh`, `gke_network_policies.yaml` | GKE sandbox backend |
| `env_service_*.py`, `env_service.sh` | seta env_service rollout path |
| `build_prompt_dataset.py` | build a prompt JSONL from Harbor task directories |
| `export_to_hf.sh`, `complete_hf_export.py` | export a trained checkpoint to Hugging Face format |

## Data

Training needs a prompt file (`PROMPT_DATA`) and the matching task directories (`TASKS_DIR`).
See [data/README.md](data/README.md). The files shipped there come from public benchmarks and are
**examples only**; train on your own task set, for example
[SETA-Env](https://huggingface.co/datasets/camel-ai/SETA-Env).

## Common settings

Every example reads the same core variables (full list in each README):

| Variable | Meaning |
|---|---|
| `HEAD_IP` or `CLUSTER_CONFIG` | Ray head node |
| `MODEL_ROOT` | where the prepared model lives (default `/root/models`) |
| `PROMPT_DATA`, `TASKS_DIR` | training prompts and their Harbor task directories |
| `SANDBOX_BACKEND` | `daytona` (default), `modal`, `gke` or `docker` |
| `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY` | optional W&B logging |
| `RUNS_ROOT`, `RUN_NAME` | outputs go to `$RUNS_ROOT/$RUN_NAME/` (default `./runs/<example>-<timestamp>`) |
| `DRY_RUN=1` | print the resolved training command and exit |

## Acknowledgements

The Miles training pipeline was built in collaboration with the RadixArk
[Miles](https://github.com/radixark/miles) team.
