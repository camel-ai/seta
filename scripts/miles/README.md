# Miles RL training for the Terminal Agent

RL (GRPO) training of terminal/CAMEL agents on the **seta_env** environment using the
[miles](https://github.com/radixark/miles) framework, with a **disaggregated, session-server**
rollout architecture and **Daytona** sandboxes for environment execution.

Two models are wired up end-to-end:

| Model | Launcher | Image |
|---|---|---|
| **GLM-4.7-Flash** | `run_glm47_flash_seta_session_server.{py,sh}` | main miles image |
| **DeepSeek-V4-Flash-FP8** | `run_deepseek_v4_seta_session_server.{py,sh}` | DeepSeek-V4 image (`*_v4docker` variants) |

> Naming convention: `*_v4docker` scripts target the **DeepSeek-V4 image**; the ones without the
> `v4docker` suffix run on the **normal main image**.

---

## Architecture

```
 ┌─────────────── Ray cluster (8 nodes) ───────────────┐
 │  train nodes (Megatron)   ⇄   serve nodes (SGLang)  │   disaggregated
 └───────────────────────────┬─────────────────────────┘
                             │  session server (TITO capture)  ── miles/native router
                             ▼
                        env_service  (FastAPI, :8002)
                             │  POST /step  (per-trajectory session URL)
                             ▼
                        Daytona sandboxes  (one per rollout trajectory)
```

- **env_service** (`core/env_service.sh` → `seta_env.services.env_service`) orchestrates Daytona
  sandboxes, builds task environments from `DATASET_ROOT/<dataset>/<task>`, runs the agent per
  `POST /step`, and grades the trajectory.
- **Session server** (miles) captures token-in/token-out (TITO) so training sees the exact tokens the
  model generated; the agent talks OpenAI-compatible to a per-trajectory session URL.
- **Rollout** is fully-async (1-step-off): training and rollout run concurrently on separate nodes.

## Folder layout

```
scripts/miles/
├── README.md                       # this file
├── PATCHES.md                      # V4-DOCKER-specific in-container patch notes
├── run_*_session_server.{py,sh}    # session-server launchers (GLM-4.7-Flash, DeepSeek-V4)
├── run_deepseek_v4_*               # DeepSeek-V4 variants (aime/seta, sync/async, v4docker)
├── run_glm47_flash_*               # GLM-4.7-Flash variants
├── eval_v4_flash_tb.sh             # terminal-bench eval
├── core/                           # modules the run scripts import + serve config
│   ├── seta_agent_function.py      #   --custom-agent-function-path (session-server agent)
│   ├── fully_async_rollout_seta.py #   --rollout-function-path (async worker)
│   ├── generate_with_camel.py      #   --custom-generate-function-path (sync path)
│   ├── reward_func.py              #   --custom-rm-path
│   ├── group_reward_filter.py      #   --dynamic-sampling-filter-path (zero-std / env-fail drop)
│   ├── camel_rollout_metrics.py    #   --custom-rollout-log-function-path
│   ├── env_service.sh              #   env_service launcher (tmux)
│   └── configs/                    #   seta_env_config_*.yaml (per-model env configs)
└── utils/                          # reusable helpers NOT used by run scripts
    ├── daytona_cleanup_ours.py     #   delete our Daytona sandboxes (DELETE=1)
    ├── daytona_watchdog.py         #   monitor Daytona capacity
    ├── tito_state_cleanup_loop.sh  #   reclaim tito_state.json disk
    ├── recovery_monitor.py, cleanup_run_artifacts.py, ...
```

Run scripts reference core modules by package path (`core.seta_agent_function.run`, …); `scripts/miles`
is on `PYTHONPATH`, so `core` resolves as a package.

---

## Prerequisites

1. **8-node Ray cluster**, already bootstrapped, reachable at `HEAD_IP` (default in each `.sh`).
   External Ray is used (`MILES_SCRIPT_EXTERNAL_RAY=1`).
2. **Credentials in `~/.bashrc`** (sourced by the launchers; never printed):
   - `DAYTONA_API_KEY`, `DAYTONA_API_URL` — Daytona sandboxes
   - `WANDB_API_KEY` — metrics
   - `HF_TOKEN` — model + dataset download
3. **Model prepared** — downloaded + converted to a Megatron torch-dist checkpoint (see below).
4. **Task dataset registered** under `DATASET_ROOT` (default `dataset/`) as
   `DATASET_ROOT/<dataset_name>/<task_name>/` harbor task dirs, plus a parquet listing the tasks.

## 1. Prepare the model (one-time)

Downloads the HF checkpoint and converts it to `_torch_dist`:

```bash
python scripts/miles/run_glm47_flash_seta_session_server.py prepare      # GLM-4.7-Flash
python scripts/miles/run_deepseek_v4_seta_session_server.py prepare      # DeepSeek-V4-Flash-FP8
```

Outputs land in `--model-dir` (default `/data/models`): `<model>/` (HF) and `<model>_torch_dist/`.

## 2. Dataset

Each rollout resolves its environment as `DATASET_ROOT/<dataset_name>/<task_name>`, where
`dataset_name = CAMEL_DATASET_NAME` and `task_name = metadata.instance_id` from the parquet.

- Place harbor task dirs (each with `task.toml`, `instruction.md`, `environment/Dockerfile`,
  `tests/`, `solution/`) under `DATASET_ROOT/<dataset_name>/`.
- Build a parquet with columns `prompt` (the instruction), `label` (the task id/slug), and
  `metadata = {"instance_id": <slug>, "agent_name": "tito_train_agent"}`.
- Point the launcher at it via `seta_env_parquet_path` (`--prompt-data`) and set
  `CAMEL_DATASET_NAME=<dataset_name>`.

## 3. Launch training

The `.sh` restarts `env_service` (Daytona) then submits the Ray job:

```bash
bash scripts/miles/run_glm47_flash_seta_session_server.sh      # GLM-4.7-Flash
bash scripts/miles/run_deepseek_v4_seta_session_server.sh      # DeepSeek-V4-Flash-FP8
```

Common overrides (env vars): `NUM_NODES` (default 8), `HEAD_IP`, `CAMEL_DATASET_NAME`,
`ROLLOUT_CONCURRENCY`, `MAX_SLOTS` (Daytona sandbox cap), `WANDB_PROJECT/GROUP`.

Run artifacts land in `RUN_ROOT` (default `/data/training_runs/<run>`): `checkpoints/`, `trials/`
(per-trajectory transcripts + `run_info.json`), `env_service/env_service.log`, `wandb/`, `ray_job.log`.

## 4. Config knobs

Each model has an env config in `core/configs/` (e.g. `seta_env_config_session_server_glm47.yaml`):
`model_platform: tito`, tool/reasoning parsers, `max_iteration`, `max_parallel_tool_calls`, sandbox
`override_cpus/memory/storage`. Training knobs live in the launcher `.py` (`ScriptArgs`):
node split (`rollout_num_nodes`), `rollout-batch-size` × `n-samples-per-prompt`, `ROLLOUT_CONCURRENCY`,
`max_weight_staleness`, tp/pp/ep parallelism.

**Throughput note:** rollout is Daytona/tool-exec bound (each trajectory is a multi-turn agent loop).
Size the train/serve node split and `ROLLOUT_CONCURRENCY`/`MAX_SLOTS` so the trainer isn't starved; see
the `utils/` monitors and the launcher comments for the current defaults.

## Utilities (`utils/`)

- `daytona_cleanup_ours.py` — `DELETE=1 python utils/daytona_cleanup_ours.py` deletes only our
  sandboxes (by owner label); dry-run by default.
- `tito_state_cleanup_loop.sh` — periodic reclaim of `tito_state.json` under `training_runs/`.
- `daytona_watchdog.py`, `recovery_monitor.py`, `env_service_watchdog.sh`,
  `cleanup_run_artifacts.py` — operational monitors/cleaners.
