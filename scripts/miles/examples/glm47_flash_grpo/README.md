# GLM-4.7-Flash · GRPO · terminal-agent RL

Full fine-tuning of [zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) (MoE,
~30B parameters) with GRPO (16 trajectories per task) as a terminal agent, with
[Miles](https://github.com/radixark/miles). Rollouts go through the **seta env_service**: seta's
CAMEL agent works on each task in a Daytona sandbox, talking to the model through a Miles session
server that records every generated token, and the env_service runs the task's tests to produce
the reward. Rollout and training run concurrently on separate nodes.

Files: `run_env_service.sh` (the launcher), `train.py` (model preparation and the Miles command),
`env_service.yaml` (agent, model client, sandbox size, timeouts), `env.example` (your settings).
Shared code: `../../common/env_service.sh` (starts and stops the env_service) and the Miles plugins
`../../common/env_service_{agent,rollout,reward,filters,metrics}.py`.

## How a rollout works

```
 Ray cluster: 7 rollout nodes (14 SGLang engines, TP4)  |  1 training node (Megatron, TP4 x EP8)
        ^ session server (records token-in/token-out; SGLang parses GLM tool calls and reasoning)
        |  OpenAI-compatible calls on a per-trajectory session URL
 env_service on the head node (:8002) -- POST /step per trajectory, from common.env_service_agent
        |  CAMEL agent (tito_train_agent) + task tests
        v
 Daytona sandboxes, one per trajectory (up to MAX_SLOTS = 400 at a time)
```

A continuous rollout worker (`common.env_service_rollout`) keeps `ROLLOUT_CONCURRENCY` = 30 prompt
groups in flight; every training step takes the next 16 finished groups (16 x 16 = 256
trajectories), after a filter that drops groups whose rewards are all equal or that contain a
failed sandbox, and after re-queuing groups more than 4 weight versions old.

## Requirements

- **GPUs**: 8 nodes x 8 H200. 7 nodes serve (two TP4 SGLang engines per node), 1 node trains.
  Rollout (sandbox and tool time) bounds throughput, so most GPUs serve.
- **Host memory**: the Adam state lives in the training node's host memory
  (`--optimizer-cpu-offload`): plan for several hundred GB of RAM there. The head node also runs
  the env_service with up to 400 concurrent trajectories.
- **Disk** on storage every node can read: ~60 GB (HF model) + ~57 GB (torch_dist copy), plus
  ~57 GB per checkpoint. Checkpoints are written every 50 steps and all are kept (3000 steps: 60
  checkpoints, ~3.4 TB); raise `SAVE_INTERVAL` or delete old ones if that is too much.
- **Daytona**: an organization whose limits hold `MAX_SLOTS` sandboxes of 1 CPU / 2 GB / 10 GB at
  once (400 CPUs, 800 GB RAM, 4 TB disk at the default; lower `MAX_SLOTS` otherwise).
- **Accounts**: Daytona; optionally Weights & Biases and Hugging Face.

## Versions

| component | version |
|---|---|
| Miles | [radixark/miles](https://github.com/radixark/miles) `46847f29378f2e3ff4f29898bddcee35a1f67b8d` (upstream `main`, also in [Michaelsqj/miles](https://github.com/Michaelsqj/miles)); no fork changes needed |
| Docker image | not recorded: a `radixark/miles` image built for the Miles commit above (SGLang `v0.5.13` base with the `sglang-miles` branch, Megatron-LM `miles-main`, transformers with native GLM-4.7-Flash support). See Step 1 |
| SGLang, Megatron-LM | the image's `/sgl-workspace/sglang` and `/root/Megatron-LM` |
| env_service side | this seta checkout with its submodules `external/camel` and `external/harbor` ([Michaelsqj/harbor](https://github.com/Michaelsqj/harbor) `a142e520a43dac51201baeeccd8a381996b49805`: Daytona backend with the create-rate limit, declarative image builds and sandbox-leak fixes the env_service settings rely on) |
| Model | `zai-org/GLM-4.7-Flash` (latest revision at download time) |

The recipe was run with stock upstream Miles, whose exact commit was not recorded;
`46847f29` is upstream `main` from that time and contains every Miles interface the recipe
uses: the agentic generate function with the session server and `--tito-allowed-append-roles`,
`--max-weight-staleness`, and the `sglang_rollout` functions the rollout worker builds on. Later
upstream commits remove `--tito-allowed-append-roles` and move the fully-async rollout into
Miles itself, so a newer `/root/miles` does not run this recipe. The launcher checks for these
features, and warns when `/root/miles` is not at the pinned commit (`STRICT_PINS=1` makes that an
error).

## Step 1: Start the container on every node

Use a `radixark/miles` image built for the pinned Miles. Either build one from Miles itself
(`git -C miles checkout 46847f29378f2e3ff4f29898bddcee35a1f67b8d`, then
`python docker/build.py --help` in that checkout), or take a `radixark/miles` image with the same
SGLang base (`v0.5.13`) and move its `/root/miles` to the pinned commit as shown below.

```bash
MILES_IMAGE=<your radixark/miles image>
docker pull "${MILES_IMAGE}"
docker run -d --name miles --gpus all --network host --ipc host --shm-size 64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /path/to/models:/root/models \
  -v /path/to/shared:/shared \
  -v /path/to/seta:/workspace/seta \
  "${MILES_IMAGE}" sleep infinity
docker exec -it miles bash
```

`/root/models` and `/shared` (runs, prompt data, task directories) must hold the same files on
every node, e.g. a shared file system mounted at the same path. Inside the container, on
**every** node, put Miles at the pinned commit:

```bash
git -C /root/miles fetch https://github.com/radixark/miles.git 46847f29378f2e3ff4f29898bddcee35a1f67b8d
git -C /root/miles checkout --detach 46847f29378f2e3ff4f29898bddcee35a1f67b8d
```

## Step 2: Start Ray

In the container, on the head node first, then on each worker node:

```bash
# head node
ray start --head --node-ip-address <HEAD_IP> --num-gpus 8 --disable-usage-stats
# every worker node
ray start --address <HEAD_IP>:6379 --num-gpus 8 --disable-usage-stats
# check: 8 nodes, 64 GPUs
ray status
```

The launcher runs on the head node and submits the training job to this cluster (it never starts
or stops Ray itself). The Ray workers must reach the head node on port 8002 (the env_service).

## Step 3: Prepare the model

Once, in the container on one node with 8 free GPUs:

```bash
cd /workspace/seta
PYTHONPATH=/root/miles python scripts/miles/examples/glm47_flash_grpo/train.py prepare --model-root /root/models
```

It downloads `zai-org/GLM-4.7-Flash` to `/root/models/GLM-4.7-Flash` (HF) and converts it to
`/root/models/GLM-4.7-Flash_torch_dist` (Megatron torch_dist, the starting weights and the KL
reference). A finished conversion is skipped. Set `HF_TOKEN` in the environment if your Hugging
Face account needs it.

## Step 4: The env_service and Daytona

The env_service (`seta_env/services/env_service.py`, see [docs/env_service.md](../../../../docs/env_service.md))
is a FastAPI service. Here a single instance runs on the head node: the launcher starts it with
this folder's `env_service.yaml`, waits for `/health` and stops it when the launcher exits. The
multi-node scheduler, `nodes.yaml` and FRP tunnel from that document are not needed.

It needs its own Python environment with seta_env, CAMEL and Harbor (Python >= 3.12; kept apart
from the Miles Python). Once, in the container on the head node:

```bash
cd /workspace/seta
git submodule update --init external/camel external/harbor
pip install uv
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e external/camel -e external/harbor -e . \
  fastapi uvicorn aiofiles pandas
.venv/bin/python -c "import seta_env.services.env_service"   # must import cleanly
```

The launcher uses `<seta checkout>/.venv/bin/python`; set `ENV_SERVICE_PYTHON` for another
location. `bash setup.sh` in the seta root builds seta's full development environment instead.

Daytona: create an API key at [app.daytona.io](https://app.daytona.io) and put it in `.env` as
`DAYTONA_API_KEY` (plus `DAYTONA_API_URL` for a non-default endpoint). Every sandbox gets
1 CPU / 2 GB RAM / 10 GB disk (`runtime` in `env_service.yaml`), and `MAX_SLOTS` of them run at
once. Task images are built through Daytona's declarative image cache (`DAYTONA_DECLARATIVE=1`),
so no named snapshots accumulate.

## Step 5: Dataset

Two inputs, readable on the head node:

| variable | what it is |
|---|---|
| `TASKS_DIR` | one Harbor task per sub-directory: `<task>/task.toml`, `instruction.md`, `environment/`, `tests/` |
| `PROMPT_DATA` | JSONL (or parquet with the same columns), one row per task |

A row names its task directory in `metadata.instance_id`; the prompt is the task instruction,
either as text or as a one-message chat list (the format of
[../../data/README.md](../../data/README.md)):

```json
{"label": "<task>", "prompt": "<task>/instruction.md, verbatim", "metadata": {"instance_id": "<task>"}}
```

The prompt files shipped next to that README are examples only (public benchmark tasks: do not train on them).
Build your own, e.g. from [SETA-Env](https://huggingface.co/datasets/camel-ai/SETA-Env):

```bash
cd /workspace/seta
.venv/bin/python -m seta_env.dataset.download --list
.venv/bin/python -m seta_env.dataset.download seta-env-final --output /shared/dataset/seta-env-final
python scripts/miles/common/build_prompt_dataset.py build \
  --tasks-dir /shared/dataset/seta-env-final --agent-name tito_train_agent --output /shared/prompts/train.jsonl
```

Then `TASKS_DIR=/shared/dataset/seta-env-final` and `PROMPT_DATA=/shared/prompts/train.jsonl`.
`build_prompt_dataset.py filter` selects subsets (`--match`, `--include`, `--exclude`). The
env_service looks tasks up as `<parent of TASKS_DIR>/<name of TASKS_DIR>/<instance_id>`; the
launcher checks that every row has its directory. Exclude tasks that need a GPU: sandboxes are
CPU-only.

## Step 6: Fill in your settings

```bash
cd /workspace/seta/scripts/miles/examples/glm47_flash_grpo
cp env.example .env     # then edit .env
```

| variable | required | meaning |
|---|---|---|
| `HEAD_IP` (or `CLUSTER_CONFIG`) | yes | Ray head node; the Ray workers reach the env_service there |
| `MODEL_ROOT` | yes | holds `GLM-4.7-Flash` and `GLM-4.7-Flash_torch_dist` (Step 3) |
| `PROMPT_DATA`, `TASKS_DIR` | yes | Step 5 |
| `DAYTONA_API_KEY` | yes | Daytona API key; `DAYTONA_API_URL` for a non-default endpoint |
| `ENV_SERVICE_PYTHON` | if not at the default | Python of Step 4 |
| `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME` | if needed | interconnect interface when it is not the default route |
| `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY` | no | W&B logging (off without a key) |
| `RUNS_ROOT`, `RUN_NAME` | no | run folder `RUNS_ROOT/RUN_NAME` (default `./runs/glm47-flash-grpo-<UTC time>`) |
| `MILES_DIR`, `MEGATRON_PATH`, `STRICT_PINS` | no | checkouts and pin enforcement |
| training and env_service knobs | no | see Configuration reference |

## Step 7: Launch

On the head node, inside the container, in `tmux` or `screen` (the launcher hosts the env_service
for the whole run):

```bash
cd /workspace/seta
DRY_RUN=1 bash scripts/miles/examples/glm47_flash_grpo/run_env_service.sh   # print the resolved command
bash scripts/miles/examples/glm47_flash_grpo/run_env_service.sh
```

The launcher checks the model, the Miles checkout and the prompt data, starts the env_service,
and submits `train_async.py` to Ray. When the launcher exits it stops the env_service; if the Ray
job outlives it, stop it with `ray job list` / `ray job stop <id>`.

## Monitor

- `$RUNS_ROOT/$RUN_NAME/logs/train.log`: the training job (rewards, losses, step timing, filter
  and staleness counts per step);
- `$RUNS_ROOT/$RUN_NAME/logs/env_service.log` and `trials/$RUN_NAME/`: one folder per trajectory
  (agent log, test output, reward);
- `curl -s http://<HEAD_IP>:8002/health`: busy and free sandbox slots, image builds in flight,
  completed and failed steps;
- Ray dashboard: `http://<HEAD_IP>:8265`;
- W&B, if `WANDB_API_KEY` is set: Miles' rollout and training metrics plus `rollout/agent/*`
  (turns, tool calls, termination reasons, agent timeouts).

## Outputs and resume

Under `$RUNS_ROOT/$RUN_NAME/`: `checkpoints/iter_XXXXXXX/` (weights only, torch_dist, every
`SAVE_INTERVAL` steps), `trials/`, `logs/`, `wandb/`, `run_manifest.md` (commits, data sha256)
and `launcher/` (copies of the launch files); `dump_details/` with `DUMP_DETAILS=1`.

**Resume**: launch again with the same `RUN_NAME`; Miles loads the latest checkpoint from
`checkpoints/`. Optimizer state is not saved, so Adam restarts.

**Export to Hugging Face format** (inside the container; config and tokenizer come from the base
checkpoint):

```bash
MODEL_DIR=/root/models/GLM-4.7-Flash bash scripts/miles/common/export_to_hf.sh \
  $RUNS_ROOT/$RUN_NAME/checkpoints/iter_XXXXXXX /root/models/GLM-4.7-Flash-grpo
```

**Leftover sandboxes**: after an interrupted run, list the Harbor-created Daytona sandboxes and
delete them (only when no other job uses the same Daytona organization):

```bash
.venv/bin/python scripts/miles/common/daytona_cleanup.py            # dry run
.venv/bin/python scripts/miles/common/daytona_cleanup.py --delete
```

## Configuration reference

Set in `.env` or the environment; defaults are the tested configuration.

| variable | default | meaning |
|---|---|---|
| `NUM_NODES` | 8 | nodes in the Ray cluster |
| `ROLLOUT_NUM_NODES` | 7 | nodes serving SGLang (TP4 engines); the rest train (TP4 x EP8, so a multiple of 8 GPUs) |
| `GPUS_PER_NODE` | 8 | |
| `NUM_ROLLOUT` | 3000 | optimizer steps |
| `ROLLOUT_BATCH_SIZE` | 16 | prompt groups per step |
| `N_SAMPLES_PER_PROMPT` | 16 | trajectories per prompt: the GRPO group |
| `ROLLOUT_CONCURRENCY` | 30 | prompt groups in flight (30 x 16 = 480 trajectories for 400 sandbox slots) |
| `MAX_WEIGHT_STALENESS` | 4 | a finished group more than this many weight versions old is re-queued |
| `GROUP_FILTER_MIN_REWARD_STD` | 1e-8 | drop groups whose rewards are all equal (no GRPO signal) |
| `GROUP_FILTER_MAX_ENV_FAILURES` | 1 | drop groups with this many failed trajectories (sandbox or env_service errors) |
| `LR` | 1e-6 | learning rate (constant, Adam, weight decay 0.1) |
| `SAVE_INTERVAL` | 50 | steps between checkpoints (all kept) |
| `DUMP_DETAILS` | 0 | 1: per-step rollout and training tensors under `dump_details/` |
| `EXTRA_ARGS` | | extra Miles flags, appended last |
| `ENV_SERVICE_CONFIG` | `env_service.yaml` here | agent, model client, sandbox size, timeouts |
| `ENV_SERVICE_PORT` | 8002 | env_service port on the head node |
| `MAX_SLOTS` | 400 | concurrent trajectories = live Daytona sandboxes |
| `STEP_TIMEOUT_SECONDS` | 5000 | hard cap on one trajectory; above the 4260 s sum of the `task_timeouts` |
| `STEP_USE_SUBPROCESS` | 1 | run each trajectory in a killable subprocess, so the cap can free a hung slot |
| `BUILD_CONCURRENCY` | 24 | concurrent task-image builds |
| `INFLIGHT_LEAD` | 64 | trajectories prepared ahead of a free slot |
| `HARBOR_DAYTONA_MAX_CREATES` | 24 | concurrent Daytona create calls (keeps bursts under Daytona's create rate limit) |
| `DAYTONA_DECLARATIVE` | 1 | build images through Daytona's declarative image cache instead of named snapshots |
| `DAYTONA_SNAPSHOT_EVICT_AGE_HOURS` | 3 | with named snapshots only: delete ones unused this long |
| `MODEL_TIMEOUT` | 900 | the agent's client timeout for one model call, seconds |
| `WANDB_PROJECT` | `glm47-flash-grpo` | |

In `env_service.yaml`: sampling (temperature 1.0, top_p 0.95, up to 8192 tokens per call), one tool
call per turn, 60 turns, 60000 tokens of context, binary reward (1 when every test passes), and the
per-phase timeouts. Fixed in `train.py`: TP4 / EP8 with full recomputation and 32768 tokens per
GPU, SGLang memory fraction 0.7, GRPO with clip 0.2 / 0.28 and a zero-weighted KL term, CPU-offloaded
Adam, the GLM-4.7 parsers (`glm47` tool calls, `glm45` reasoning).

## Notes

- **Sampling settings come from `env_service.yaml`**: the agent sends its own temperature, top_p
  and max_tokens with every call; Miles' `--rollout-temperature` does not apply on this path.
- **Chat template and parsing are engine-side**: SGLang applies GLM-4.7-Flash's template and
  parses tool calls (`glm47`) and reasoning (`glm45`); the session server accepts user and tool
  turns appended by the agent. Do not add `--apply-chat-template`.
- **Timeouts**: one model call may take up to `MODEL_TIMEOUT` (900 s; the 180 s default is too
  short for 8192-token turns under full load), the agent run up to 2400 s (its result is still
  tested), one trajectory up to `STEP_TIMEOUT_SECONDS`. The rollout worker waits at most 3600 s
  for a trajectory; a longer one is dropped.
- **Throughput** is bound by sandboxes and tool execution. If the trainer waits on rollout, raise
  `MAX_SLOTS` (within your Daytona limits) together with `ROLLOUT_CONCURRENCY`; keep
  `ROLLOUT_CONCURRENCY x N_SAMPLES_PER_PROMPT` above `MAX_SLOTS` so no slot idles.
- **Daytona outages** make the group filter drop most groups: the run slows down instead of
  training on failures. Watch the filter counts in `train.log` and `/health`.
