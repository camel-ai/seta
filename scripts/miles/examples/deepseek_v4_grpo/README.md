# DeepSeek-V4-Flash · GRPO · terminal-agent RL

Full fine-tuning of [DeepSeek-V4-Flash-FP8](https://huggingface.co/sgl-project/DeepSeek-V4-Flash-FP8)
(all parameters, FP8 blockwise training with MoE routing replay) with GRPO on terminal tasks, using
[Miles](https://github.com/radixark/miles). Two rollout paths train the same model:

| script | rollout path | trainer |
|---|---|---|
| `run_harbor_terminus2.sh` | Harbor agent server running Harbor's **Terminus-2** agent in Daytona, Modal, GKE or Docker sandboxes | sync `train.py`, actor and SGLang colocated on every GPU |
| `run_env_service.sh` | seta **env_service** running seta's **CAMEL** agent in Daytona sandboxes | `train_async.py`, 2 SGLang nodes + 6 training nodes, continuous rollout worker |

`train.py` is the one Miles launcher for both (`prepare` for the model, `train --rollout-backend
harbor|env-service`).

## Requirements

- **GPUs**: 8 nodes × 8 × H200, InfiniBand between nodes. The Harbor path also has Megatron
  layouts for 5, 6 and 7 nodes (`NUM_NODES`); the env_service path is set up for 8 (2 serve, 6 train).
- **Disk** (shared storage, same path on every node): about 0.3 TB for the FP8 checkpoint, 0.6 TB
  for the BF16 conversion intermediate (can be deleted afterwards), 0.6 TB for the torch_dist
  weights, and about 0.6 TB per saved checkpoint (291B parameters in BF16, weights only).
- **Host memory**: the Adam state lives in CPU memory (optimizer offload).
- **Accounts**: Hugging Face (the model repo is public), one sandbox provider (Daytona for
  env_service; Daytona, Modal or GKE for Harbor, or a local Docker daemon), optionally W&B.

## Versions

| component | `run_harbor_terminus2.sh` | `run_env_service.sh` |
|---|---|---|
| Docker image | `radixark/miles@sha256:ca0bb593dd6f4011b444f64d478b72c213e4c70421f4d7f94e593a709562429e` | not recorded |
| Miles | [`Michaelsqj/miles`](https://github.com/Michaelsqj/miles) `5d3c77e66281871245ce43e32b3c0c8c96f147e8`: upstream Miles plus a routing-replay fix (a response that ends on a stop token returns one extra routed-experts row; it is dropped instead of failing the rollout) | not recorded. The command uses `--tito-allowed-append-roles`, which upstream Miles removed in `978bddca` (#1818): use a Miles from before that commit |
| Harbor | [`Michaelsqj/harbor`](https://github.com/Michaelsqj/harbor) `1e02f96f00a18b5e9158a4baffbaa2bb5a0b2146`: agent server (`/run`, `/flush`, dashboard) with per-trial subprocess workers | not used (env_service drives the sandboxes itself) |
| seta | this repository | this repository (`seta_env` env_service and CAMEL agent) |

The Harbor script warns when `MILES_DIR` is not at the tested commit; `STRICT_PINS=1` makes that an
error. Harbor itself is installed at `HARBOR_COMMIT` automatically.

## Step 1: Start the container on every node

```bash
IMAGE=radixark/miles@sha256:ca0bb593dd6f4011b444f64d478b72c213e4c70421f4d7f94e593a709562429e
docker pull ${IMAGE}
docker run -d --name miles --gpus all --network host --ipc host --shm-size 64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /path/to/models:/root/models \
  -v /path/to/datasets:/datasets \
  -v /path/to/seta:/workspace/seta \
  ${IMAGE} sleep infinity
docker exec -it miles bash
```

All three mounts must be shared storage visible at the same path on every node. Inside the
container, on every node, check out the tested Miles commit:

```bash
git -C /root/miles fetch https://github.com/Michaelsqj/miles.git 5d3c77e66281871245ce43e32b3c0c8c96f147e8
git -C /root/miles checkout 5d3c77e66281871245ce43e32b3c0c8c96f147e8
```

`run_env_service.sh` needs a Miles from before `978bddca` instead (see Versions); it checks for
`--tito-allowed-append-roles` in `MILES_DIR` before it starts anything.

For the Docker sandbox backend, also pass `-v /var/run/docker.sock:/var/run/docker.sock` on the
head node (see Step 4).

## Step 2: Start Ray

On the head node (inside the container):

```bash
export HEAD_IP=<head node IP>
ray start --head --node-ip-address ${HEAD_IP} --num-gpus 8 --dashboard-host 0.0.0.0 --disable-usage-stats
```

On every other node:

```bash
ray start --address ${HEAD_IP}:6379 --num-gpus 8
```

`ray status` on the head should list all nodes. The launchers submit the job to the dashboard at
`http://${HEAD_IP}:8265`, so it must listen on that address. Run everything below on the head node.

## Step 3: Prepare the model

```bash
cd /workspace/seta
PYTHONPATH=/root/miles python scripts/miles/examples/deepseek_v4_grpo/train.py prepare \
  --model-root /root/models --num-nodes 8
```

It downloads `sgl-project/DeepSeek-V4-Flash-FP8`, casts it to BF16 and converts that to Megatron
torch_dist on all 8 nodes (through Ray: the conversion layout TP1 / PP8 / EP4 is the one verified
for this model). Finished steps are skipped when you run it again. Result:

```
/root/models/DeepSeek-V4-Flash-FP8/             # FP8 Hugging Face checkpoint: SGLang serves it
/root/models/DeepSeek-V4-Flash-FP8-bf16/        # conversion intermediate, not used for training
/root/models/DeepSeek-V4-Flash-FP8_torch_dist/  # Megatron weights: KL reference and initial actor
```

torch_dist checkpoints reshard on load, so training may use a different parallel layout.

## Step 4: Sandbox backend

### Harbor path (`run_harbor_terminus2.sh`)

Set `SANDBOX_BACKEND` and that backend's credentials. Harbor is cloned at `HARBOR_COMMIT` into
`~/.cache/seta/harbor/<commit>` (`HARBOR_HOME`) with its own virtualenv on the first run (needs
`uv` or `python3.12` and network access). To install it ahead of time:

```bash
HARBOR_COMMIT=1e02f96f00a18b5e9158a4baffbaa2bb5a0b2146 bash scripts/miles/common/harbor_install.sh
```

The agent server runs on the head node; Terminus-2 calls the model from there, so the sandboxes
never need to reach the cluster.

#### Daytona

`SANDBOX_BACKEND=daytona` (default), `DAYTONA_API_KEY` (https://app.daytona.io → API keys), and
`DAYTONA_API_URL` for a non-default endpoint.

#### Modal

`SANDBOX_BACKEND=modal`, then either run `modal token new` once in the container or set
`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.

#### GKE

Your own Google Cloud project. Only the sandboxes run in GKE; the agent server stays on the head
node and Cloud Build builds each task's image into Artifact Registry the first time the task runs.

**Prerequisites.** A project with billing, and the [gcloud CLI](https://cloud.google.com/sdk/docs/install)
with `gcloud components install gke-gcloud-auth-plugin kubectl`, logged in (`gcloud auth login`, or
`gcloud auth activate-service-account --key-file=...`) where you run the commands below **and** inside
the container on the head node, where the agent server runs. `create` enables the Kubernetes Engine,
Compute Engine, Artifact Registry, Cloud Build and IAM APIs. IAM: Owner, or the roles listed at the top
of `../../common/gke_cluster.sh` (to set up: Kubernetes Engine Admin, Artifact Registry Admin, Service
Account Admin and User, Project IAM Admin, Service Usage Admin; to train: Kubernetes Engine Developer,
Artifact Registry Reader, Cloud Build Editor, Storage Admin).

Put `SANDBOX_BACKEND=gke`, `GKE_PROJECT_ID`, `GKE_REGION` and `GKE_CLUSTER_NAME` in `.env` (Step 6); the
GKE scripts read them from `./.env` too. Then, from this folder:

```bash
bash ../../common/gke_cluster.sh create                  # registry, node service account, cluster, sandbox pool at 0 nodes
HARBOR_COMMIT=<HARBOR_COMMIT of the run script> bash ../../common/gke_tools_image.sh   # prints GKE_TOOLS_IMAGE=...: add it to .env
bash ../../common/gke_cluster.sh apply-network-policies  # optional: sandboxes get DNS and web only (--no-web-egress: no internet)
bash ../../common/gke_cluster.sh scale 8                 # before a run: 8 n2-standard-16 sandbox nodes
bash ../../common/gke_cluster.sh down --yes              # after the run: sandbox nodes to zero
```

The launcher writes the cluster's kubeconfig on first use (`gke_cluster.sh credentials`, to
`~/.kube/gke_<project>_<region>_<cluster>`). An `n2-standard-16` node holds about 15 one-CPU sandboxes;
scale for `AGENT_MAX_CONCURRENT`. `status` shows nodes and sandbox pods; `clear-sandboxes --yes` deletes
pods left by a crashed run. `DRY_RUN=1` prints every command instead of running it. For an existing
cluster skip `create` (network policies need Dataplane V2) and set `GKE_REGISTRY_NAME` to a Docker
repository in `GKE_REGION`. **Cost:** sandbox nodes bill while they are up, so run `down --yes` after
every run; the control plane and one small system node per zone bill until `gke_cluster.sh delete --yes`.
The first run on a new task set also pays for Cloud Build time and registry storage.

#### Docker

`SANDBOX_BACKEND=docker` runs every sandbox as a container on the head node through its Docker
daemon: start the head container with `-v /var/run/docker.sock:/var/run/docker.sock` and make
the `docker` CLI available in it. Keep `AGENT_MAX_CONCURRENT` within what one machine can run.

### env_service path (`run_env_service.sh`)

The script starts seta's env_service (`seta_env.services.env_service`, port `ENV_SERVICE_PORT`,
default 8002) on the head node with `env_service.yaml`, through `common/env_service.sh`. For
every trajectory it creates a Daytona sandbox from the task directory, runs the CAMEL agent
against the Miles session URL and returns the verified reward (see
[docs/env_service.md](../../../../docs/env_service.md)). It needs:

- **Daytona**: `DAYTONA_API_KEY`, and `DAYTONA_API_URL` for a non-default endpoint.
  `env_service.yaml` gives every sandbox 1 CPU, 2 GB RAM and 6 GB disk, so `MAX_SLOTS=160`
  sandboxes need a Daytona quota of 160 CPUs, 320 GB RAM and 960 GB disk; lower `MAX_SLOTS` for
  a smaller quota.
- **A Python environment for the env_service** with `seta_env` and its CAMEL and Harbor
  dependencies, separate from the Miles environment. By default the script uses
  `/workspace/seta/.venv` (`ENV_SERVICE_PYTHON`); create it once on the head node:

```bash
cd /workspace/seta
git submodule update --init external/camel external/harbor
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e external/camel
uv pip install --python .venv/bin/python -e external/harbor
uv pip install --python .venv/bin/python -e .
```

## Step 5: Dataset

Both paths take `PROMPT_DATA` (one row per task) and `TASKS_DIR` (one task per sub-directory:
`<task>/task.toml`, `instruction.md`, `environment/`, `tests/`). The files in
[`../../data/`](../../data/README.md) are **examples only** (public benchmark tasks: do not train
on them). Train on your own tasks, for example the
[SETA-Env](https://huggingface.co/datasets/camel-ai/SETA-Env) environments:

```bash
cd /workspace/seta
python -m seta_env.dataset.download seta-env-final     # -> dataset/seta-env-final/<task>/
python scripts/miles/common/build_prompt_dataset.py build \
  --tasks-dir dataset/seta-env-final --agent-name terminus-2 --output /datasets/train.jsonl
```

`build_prompt_dataset.py filter` selects subsets (`--match`, `--include`, `--exclude`); see
[`../../data/README.md`](../../data/README.md). Tasks that need a GPU cannot run in these sandboxes.

Both scripts take this JSONL: `PROMPT_DATA=/datasets/train.jsonl`,
`TASKS_DIR=/workspace/seta/dataset/seta-env-final`. The Harbor agent server finds a task at
`TASKS_DIR/<instance_id>`. env_service finds it at `DATASET_ROOT/<dataset>/<instance_id>`;
`run_env_service.sh` derives both from `TASKS_DIR` (its parent directory and its name) and sends
the instruction text of each row to the CAMEL agent.

## Step 6: Fill in your settings

```bash
cd /workspace/seta/scripts/miles/examples/deepseek_v4_grpo
cp env.example .env   # then edit .env
```

Both scripts read `.env` from this folder (or the file in `ENV_FILE`); values there override the
script defaults.

| variable | required | meaning |
|---|---|---|
| `HEAD_IP` | yes (or `CLUSTER_CONFIG`) | Ray head IP; the agent server / env_service and the Ray job use it |
| `CLUSTER_CONFIG` | | a copy of [`../../cluster.example.yaml`](../../cluster.example.yaml), instead of `HEAD_IP` |
| `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME` | | interconnect interface, when it is not the default route |
| `MODEL_ROOT` | yes | where `train.py prepare` wrote the model (default `/root/models`) |
| `MILES_DIR`, `MEGATRON_PATH` | | checkouts in the image (default `/root/miles`, `/root/Megatron-LM`) |
| `PROMPT_DATA` | yes | prompt JSONL (Step 5) |
| `TASKS_DIR` | yes | task directories referenced by `PROMPT_DATA` |
| `SANDBOX_BACKEND` | Harbor | `daytona` (default), `modal`, `gke` or `docker` |
| `DAYTONA_API_KEY` | Daytona | Daytona API key |
| `DAYTONA_API_URL` | | Daytona API endpoint, for a non-default one |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | Modal | or run `modal token new` |
| `GKE_*` | GKE | see Step 4, GKE |
| `WANDB_API_KEY` | | enables W&B logging; passed to Miles as `--wandb-key` |
| `WANDB_PROJECT`, `WANDB_ENTITY` | | W&B project (default `deepseek-v4-grpo`) and team |
| `HF_TOKEN` | | Hugging Face token for `train.py prepare` (the repo is public) |
| `RUNS_ROOT`, `RUN_NAME` | | run folder `RUNS_ROOT/RUN_NAME` (default `./runs`, `deepseek-v4-grpo-<path>-<UTC time>`) |
| `HARBOR_COMMIT`, `HARBOR_HOME`, `AGENT_SERVER_PORT` | | Harbor path: Harbor checkout and agent server port (default 11000) |
| `ENV_SERVICE_PYTHON`, `ENV_SERVICE_PORT` | | env_service path: its Python (default `/workspace/seta/.venv/bin/python`, i.e. `<seta>/.venv`) and port (default 8002) |
| `STRICT_PINS` | | `1`: fail instead of warn when Miles is not at the tested commit |

The training knobs are in the Configuration reference below.

## Step 7: Launch

Run on the head node inside the container, in `tmux` or `screen` (the script stays in the
foreground for the whole run). First print the resolved command:

### Harbor + Terminus-2

```bash
DRY_RUN=1 bash scripts/miles/examples/deepseek_v4_grpo/run_harbor_terminus2.sh
bash scripts/miles/examples/deepseek_v4_grpo/run_harbor_terminus2.sh
```

The script checks the model, Miles pin and data (every `PROMPT_DATA` row needs
`TASKS_DIR/<instance_id>/task.toml`), installs Harbor, checks the Harbor features it relies on,
starts the agent server (stopped again when the script exits) and submits the Ray job.

### env_service + CAMEL

```bash
DRY_RUN=1 bash scripts/miles/examples/deepseek_v4_grpo/run_env_service.sh
bash scripts/miles/examples/deepseek_v4_grpo/run_env_service.sh
```

The script checks the model, the Miles version and the data, starts env_service with
`env_service.yaml` (stopped again when the script exits) and submits the Ray job.

## Monitor

- `$RUNS_ROOT/$RUN_NAME/logs/train.log`: the Ray job output (Miles driver log).
- Harbor path: `logs/agent_server.log`, one directory per trial under `trials/`, and the agent
  server dashboard at `http://${HEAD_IP}:11000/dashboard/`.
- env_service path: `logs/env_service.log`, and one folder per trajectory (agent transcript and
  `run_info.json`) under `trials/$RUN_NAME/`.
- Ray dashboard: `http://${HEAD_IP}:8265`.
- W&B, when `WANDB_API_KEY` is set: project `WANDB_PROJECT`, run named `RUN_NAME`. The Harbor
  path also logs `agent/*` timing and turn metrics.
- `run_manifest.md` records the commits, data (row count and sha256) and backend of the run;
  `launcher/` holds a copy of the launch files.

## Outputs and resume

`RUNS_ROOT` defaults to `./runs` under the directory you launch from; it must be shared storage,
because every training node writes its part of a checkpoint there.

Checkpoints go to `$RUNS_ROOT/$RUN_NAME/checkpoints/iter_*` every `SAVE_INTERVAL` steps, as
Megatron torch_dist weights without optimizer state. The same directory is also the load path:
run the script again with the same `RUN_NAME` and training continues from the latest
checkpoint (with a fresh optimizer state).

## Configuration reference

Environment variables (in `.env` or on the command line). Defaults are the configuration each
script was run with.

| variable | `run_harbor_terminus2.sh` | `run_env_service.sh` | meaning |
|---|---|---|---|
| `NUM_NODES` | 8 | 8 | nodes of `GPUS_PER_NODE` GPUs |
| `ROLLOUT_NUM_NODES` | – (colocated) | 2 | nodes that only serve SGLang |
| `GPUS_PER_NODE` | 8 | 8 | GPUs per node |
| `NUM_ROLLOUT` | 3000 | 3000 | optimizer steps |
| `ROLLOUT_BATCH_SIZE` | 16 | 8 | prompt groups trained on per step |
| `N_SAMPLES_PER_PROMPT` | 8 | 16 | trajectories per prompt (GRPO group) |
| `OVER_SAMPLING_BATCH_SIZE` | 24 | – | prompt groups started per step; the surplus is aborted |
| `ROLLOUT_CONCURRENCY` | – | 12 | prompt groups the rollout worker keeps in flight |
| `MAX_WEIGHT_STALENESS` | – | 4 | drop groups generated this many weight updates ago |
| `TEMPERATURE` | 1.0 | 0.8 | Miles rollout temperature |
| `MAX_RESPONSE_LEN` | 16384 | 8192 | tokens per model call |
| `MAX_SEQ_LEN` | 65536 | – | longest training sample; longer ones are truncated |
| `LR` | 1e-6 | 1e-6 | constant learning rate |
| `SAVE_INTERVAL` | 50 | 50 | checkpoint every N steps |
| `SKIP_SAVING` | 0 | 0 | `1`: no checkpoints |
| `DUMP_DETAILS` | 0 | 0 | `1`: write rollout samples and tensors under `dump_details/` |
| `HARDWARE` | auto | auto | GPU type (`H100`, `H200`, `B200`, ...): selects the FP8 scale format |
| `EXTRA_ARGS` | | | extra Miles arguments, appended last |
| `AGENT_MAX_CONCURRENT` | 128 | – | trials the agent server runs at once |
| `HARBOR_AGENT_MAX_ITERATIONS` | 50 | – | Terminus-2 turns per trial |
| `HARBOR_MAX_SEQ_LEN` | 1048576 | – | Terminus-2 context bound in tokens (effectively none) |
| `HARBOR_AGENT_TIMEOUT_MULTIPLIER` | 12 | – | multiplies each task's agent timeout |
| `HARBOR_AGENT_CALL_TIMEOUT_SEC` | 10800 | – | rollout-side deadline for one trial |
| `MAX_SLOTS` | – | 160 | concurrent env_service sandboxes |
| `STEP_TIMEOUT_SECONDS` | – | 5000 | one trajectory end to end |
| `MODEL_TIMEOUT` | – | 900 | the CAMEL agent's timeout for one model call |

The fixed parts of each path are in `train.py`: the Megatron layout (TP8 / EP8, one pipeline
stage per training node), the SGLang engines (Harbor: TP4 / EP4, memory fraction 0.7;
env_service: TP8 / EP8 on the rollout nodes, memory fraction 0.84), GRPO clipping 0.2 / 0.28 with
KL logged but not in the loss, blockwise FP8 training and deterministic mode. The CAMEL agent's
own limits (200 turns, 60000-token context, 8192 tokens and temperature 1.0 per call, task
timeouts) are in `env_service.yaml`.

## Notes

- **Context lengths.** Harbor path: Terminus-2's own context is effectively unbounded and a trial
  ends on the turn cap or the task timeout; training truncates each sample at `MAX_SEQ_LEN`
  (65536). env_service path: the CAMEL agent stops at 60000 tokens, below the model's original
  65536-token context; longer samples overflow Megatron's RoPE cache and crash training.
- **Over-sampling and `/flush`.** When the Harbor path has its batch, the surplus trials still
  send turns to the engines, so the engines never drain for the weight update. The generate
  wrapper (`common.harbor_abort`) and the agent's abort hook post `/flush` to the agent server,
  which cancels those trials. `logs/train.log` shows `[harbor-abort]` lines for each wave.
- **Terminus-2 options.** This Harbor commit passes unknown agent options straight into the
  model call, so `HARBOR_TERMINUS_PARSER` and `HARBOR_TERMINUS_ENABLE_SUMMARIZE` are unset by the
  script; the pinned server also has no `--agent-timeout-sec`, `--artifact` or `--collect`, so the
  script refuses `HARBOR_AGENT_TIMEOUT_SEC`, `HARBOR_EXTRA_ARTIFACTS` and `HARBOR_EXTRA_COLLECT`.
- **Agent settings reach Ray explicitly.** The rollout workers run in the Ray job, not in the
  launching shell, so `train.py` forwards `AGENT_SERVER_URL`, `HARBOR_*` and the `CAMEL_*`
  settings through the job's runtime environment. Changing them in your shell after the job
  started has no effect.
- **Model-call timeout (env_service).** DeepSeek-V4 reasoning turns of up to 8192 tokens take
  minutes under load; with the agent's default 180 s client timeout most trajectories fail, hence
  `MODEL_TIMEOUT=900`.
- **Daytona capacity (env_service).** The sandbox create timeout in `env_service.yaml` (720 s)
  is deliberately moderate: creates that only wait on a full runner pool never trigger a
  scale-up, while failing and retrying does.
