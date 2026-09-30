# Qwen3.8-27B · GRPO · terminal-agent RL

Full fine-tuning of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with GRPO (8 trials
per task, no critic) as a terminal agent, with a 131072-token context. Rollouts go through the
Harbor agent server: Harbor's Terminus-2 agent works on each task in a Daytona, Modal, GKE or
Docker sandbox, talking to the model through a Miles session server, and Harbor runs the task's
tests to produce the reward. Rollout is fully asynchronous: up to 128 trials stay in flight while
the trainer consumes finished groups.

Files: `run_harbor_terminus2.sh` (the launcher), `train.py` (builds the Miles command),
`env.example` (your settings), `prepare_model.sh` (download + convert the base model).
Exporting a trained checkpoint to Hugging Face format uses the shared `scripts/miles/common/export_to_hf.sh`.

## Requirements

- **GPUs**: 4 nodes x 8 H200 (141 GB). 2 nodes train (TP4 x PP1 x CP4), 2 nodes run 16
  single-GPU SGLang engines. The memory settings are sized for this card.
- **Host memory**: the optimizer state is offloaded to CPU, so the training nodes need a few
  hundred GB of RAM each; the head node also hosts the agent server and the session servers.
- **Disk** on storage every node can read: ~56 GB (HF model) + ~51 GB (torch_dist copy), plus
  ~54 GB per checkpoint (100 checkpoints at the default settings; see Notes).
- **Accounts**: a sandbox provider (Daytona or Modal, or your own GKE cluster; or a local Docker
  daemon), optionally Weights & Biases and Hugging Face.

## Versions

| component | version |
|---|---|
| Docker image | `radixark/miles@sha256:616ac8b9fd1e51fabcafefa47faddb67fdac3148f9d53a631dcc9606c88300cd` |
| Miles | [Michaelsqj/miles](https://github.com/Michaelsqj/miles) `1c1ff6b4383923e67263bdad0359cb1792fd63fb`: upstream plus the allocator-cache release after each training phase and `--session-server-startup-timeout-secs` |
| Harbor | [Michaelsqj/harbor](https://github.com/Michaelsqj/harbor) `ab9eaefa3c8c24777e5dd5ff2f0c615694573393` (Terminus-2 format-tolerance fixes, sandbox exec-handshake retry) |
| SGLang | `cb05a44f35a7c9e27e46d74112cc841ca674ef43` (`sglang-miles` branch, shipped in the image) |
| Megatron-LM | the image's `/root/Megatron-LM` |
| Model | `Qwen/Qwen3.8-27B` revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (Apache-2.0) |

The pinned Miles has no `qwen3.8-27B` model type; Qwen3.8-27B is architecturally identical to
Qwen3.5-27B (`model_type: qwen3_5`), so the scripts use the `qwen3.5-27B` Megatron type and the
`qwen35` chat parser. The launcher warns when `/root/miles` or `/sgl-workspace/sglang` is not at
the tested commit (`STRICT_PINS=1` makes that an error), and stops early when a Miles or Harbor
feature it depends on is missing.

## Step 1: Start the container on every node

```bash
docker pull radixark/miles@sha256:616ac8b9fd1e51fabcafefa47faddb67fdac3148f9d53a631dcc9606c88300cd
docker run -d --name miles --gpus all --network host --ipc host --shm-size 64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /path/to/models:/root/models \
  -v /path/to/shared:/shared \
  -v /path/to/seta:/workspace/seta \
  radixark/miles@sha256:616ac8b9fd1e51fabcafefa47faddb67fdac3148f9d53a631dcc9606c88300cd \
  sleep infinity
docker exec -it miles bash
```

`/root/models` and `/shared` (runs, prompt data, task directories) must hold the same files on
every node, e.g. a shared file system mounted at the same path. Inside the container, on
**every** node, put Miles at the tested commit:

```bash
git -C /root/miles fetch https://github.com/Michaelsqj/miles.git 1c1ff6b4383923e67263bdad0359cb1792fd63fb
git -C /root/miles checkout --detach 1c1ff6b4383923e67263bdad0359cb1792fd63fb
```

## Step 2: Start Ray

In the container on the head node, then on each worker node. `MALLOC_ARENA_MAX=2` keeps the
session servers' host memory in check over long runs (see Notes); the launcher warns when the
local raylet was started without it.

```bash
# head node
MALLOC_ARENA_MAX=2 ray start --head --node-ip-address <HEAD_IP> --num-gpus 8 --disable-usage-stats
# every worker node
MALLOC_ARENA_MAX=2 ray start --address <HEAD_IP>:6379 --num-gpus 8 --disable-usage-stats
# check: 4 nodes, 32 GPUs
ray status
```

The launcher runs on the head node and submits the training job to this cluster (it never starts
or stops Ray itself).

## Step 3: Prepare the model

Once, in the container on one node with free GPUs:

```bash
cd /workspace/seta
MODEL_ROOT=/root/models bash scripts/miles/examples/qwen3_8_27b_grpo/prepare_model.sh
```

It downloads the pinned revision to `/root/models/Qwen3.8-27B` (HF, ~56 GB) and converts it to
`/root/models/Qwen3.8-27B_torch_dist` (Megatron torch_dist, ~51 GB). It is idempotent: a finished
conversion is skipped.

## Step 4: Sandbox backend

Pick one with `SANDBOX_BACKEND` and put its credentials in `.env` (Step 6).

#### Daytona

The default. Create an API key at https://app.daytona.io and set `DAYTONA_API_KEY` (and
`DAYTONA_API_URL` for a non-default endpoint).

#### Modal

Run `modal token new` on the head node, or set `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.

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

This recipe sizes each GKE sandbox at 1 CPU, 2 GiB memory and 8 GiB ephemeral storage
(`SANDBOX_CPUS`, `SANDBOX_MEMORY_MB`, `SANDBOX_STORAGE_MB`) and waits up to 1200 s for a pod to
become ready (`GKE_POD_READY_TIMEOUT_SEC`), because many sandboxes cold-start at once. Other
backends use each task's own resources.

#### Docker

A Docker daemon on the head node; every sandbox runs there, so size the node for
`AGENT_MAX_CONCURRENT` of them.

Harbor is installed automatically on the first run, at the pinned commit, into
`~/.cache/seta/harbor/<commit>` (`HARBOR_HOME`; needs `python3.12` and GitHub access). To install
it ahead of time:

```bash
HARBOR_COMMIT=ab9eaefa3c8c24777e5dd5ff2f0c615694573393 bash scripts/miles/common/harbor_install.sh
```

## Step 5: Dataset

The launcher needs `PROMPT_DATA`, a JSONL with one row per task, and `TASKS_DIR`, the Harbor task
directories those rows name. The format is in [`../../data/README.md`](../../data/README.md). The
files in that folder are **examples only** (taken from public benchmarks): do not train on them.
Build the prompt file from your own tasks, for example the
[SETA-Env](https://huggingface.co/datasets/camel-ai/SETA-Env) environments:

```bash
python -m seta_env.dataset.download seta-env-final        # -> dataset/seta-env-final/<task>/
python scripts/miles/common/build_prompt_dataset.py build \
  --tasks-dir dataset/seta-env-final --agent-name terminus-2 --output /shared/prompts/train.jsonl
# then TASKS_DIR=$PWD/dataset/seta-env-final (move it to shared storage if needed)
```

GPU tasks cannot run in CPU sandboxes; leave them out (`build_prompt_dataset.py filter --exclude`).
The launcher checks that every row has a task directory.

## Step 6: Fill in your settings

```bash
cd scripts/miles/examples/qwen3_8_27b_grpo
cp env.example .env      # then edit .env
```

`.env` is read by the launcher (`ENV_FILE=/path/to/file` selects another file) and is gitignored.

| variable | required | meaning |
|---|---|---|
| `HEAD_IP` or `CLUSTER_CONFIG` | yes | Ray head address (the node you launch on) |
| `MODEL_ROOT` | yes | where `prepare_model.sh` put the model |
| `PROMPT_DATA`, `TASKS_DIR` | yes | Step 5 |
| `SANDBOX_BACKEND` + its credentials | yes | Step 4 |
| `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME` | no | interconnect interface when it is not the default route |
| `MODEL_DIR`, `INIT_LOAD` | no | HF checkpoint and starting torch_dist checkpoint (defaults under `MODEL_ROOT`) |
| `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY` | no | W&B logging is on when the key is set |
| `HF_TOKEN` | no | Hugging Face token for `prepare_model.sh` |
| `RUNS_ROOT`, `RUN_NAME` | no | output folder (`./runs/<name>-<UTC time>`); reuse a `RUN_NAME` to resume |
| `MILES_DIR`, `MEGATRON_PATH`, `SGLANG_DIR`, `STRICT_PINS`, `HARBOR_HOME`, `AGENT_SERVER_PORT` | no | checkouts and ports |
| training knobs | no | see Configuration reference |

## Step 7: Launch

On the head node, inside the container, in `tmux` or `screen` (the launcher is the Ray job's client
and hosts the agent server for the whole run):

```bash
cd /workspace/seta
DRY_RUN=1 bash scripts/miles/examples/qwen3_8_27b_grpo/run_harbor_terminus2.sh   # print the resolved command
bash scripts/miles/examples/qwen3_8_27b_grpo/run_harbor_terminus2.sh
```

The launcher checks the model, the Miles checkout and the prompt data, installs Harbor if needed,
starts the Harbor agent server, and submits `train_async.py` to Ray. When the launcher exits it
cancels the in-flight trials and stops the agent server; if the Ray job outlives it, stop it with
`ray job list` / `ray job stop <id>`.

## Monitor

- `$RUNS_ROOT/$RUN_NAME/logs/train.log`: the training job (rewards, losses, step timing);
- `$RUNS_ROOT/$RUN_NAME/logs/agent_server.log` and `trials/`: one folder per Harbor trial (agent
  trajectory, test output, reward);
- Ray dashboard: `http://<HEAD_IP>:8265`;
- Harbor rollout dashboard: `http://<HEAD_IP>:11000/dashboard/`;
- W&B, if `WANDB_API_KEY` is set.

## Outputs and resume

Under `$RUNS_ROOT/$RUN_NAME/`: `checkpoints/iter_XXXXXXX/` (actor weights, torch_dist, every
`SAVE_INTERVAL` steps), `trials/`, `logs/`, `dump_details/` (rollout data per step), `wandb/`,
`run_manifest.md` (commits, backend, data sha256) and `launcher/` (copies of the launch files).

**Resume**: launch again with the same `RUN_NAME`; Miles loads the latest checkpoint from
`checkpoints/`. Optimizer state is not saved (weights only), so it restarts.

**Start from another checkpoint**: set `INIT_LOAD` to a torch_dist checkpoint directory, e.g. an
earlier run's `checkpoints/` (it must contain `latest_checkpointed_iteration.txt`). See Notes.

**Export for serving** (inside the container):

```bash
MODEL_DIR=/root/models/Qwen3.8-27B EXPORT_PREFIXES="model.visual. mtp." \
bash scripts/miles/common/export_to_hf.sh \
  $RUNS_ROOT/$RUN_NAME/checkpoints/iter_0000029 /root/models/Qwen3.8-27B-grpo-step30
```

Miles' converter writes only the trained language model; `complete_hf_export.py` then copies the
untrained vision tower (`model.visual.*`) and MTP head (`mtp.*`) from the base checkpoint, which
SGLang's `Qwen3_5ForConditionalGeneration` loader requires, and checks the tensor names and shapes
against the base.

## Configuration reference

Set in `.env` or the environment; defaults are the tested configuration.

| variable | default | meaning |
|---|---|---|
| `NUM_NODES` | 4 | nodes in the Ray cluster |
| `ROLLOUT_NODES` | 2 | nodes running SGLang engines (one GPU each); the rest train |
| `GPUS_PER_NODE` | 8 | |
| `PIPELINE_PARALLEL_SIZE` | 1 | actor pipeline parallelism (tensor parallelism is 4) |
| `CONTEXT_PARALLEL_SIZE` | 4 | actor context parallelism: each sample is split over this many GPUs |
| `MAX_SEQ_LEN` | 131072 | tokens per training sample (prompt + all turns) |
| `MAX_RESPONSE_LEN` | 32768 | tokens per model turn |
| `MAX_TOKENS_PER_GPU` | `MAX_SEQ_LEN / CONTEXT_PARALLEL_SIZE` | micro-batch budget; x CP it must hold one full sample |
| `NUM_ROLLOUT` | 3000 | optimizer steps |
| `ROLLOUT_BATCH_SIZE` | 16 | prompts per step |
| `N_SAMPLES_PER_PROMPT` | 8 | trials per prompt: the GRPO group |
| `GLOBAL_BATCH_SIZE` | 128 | must equal `ROLLOUT_BATCH_SIZE x N_SAMPLES_PER_PROMPT` (one step per rollout) |
| `AGENT_MAX_CONCURRENT` | 128 | trials in flight (Miles cap and agent-server concurrency) |
| `MAX_WEIGHT_STALENESS` | 2 | weight updates a trial may lag and still be trained on |
| `DYNAMIC_SAMPLING_FILTER` | `check_no_infra_failure` | `none`, `check_no_infra_failure`, or `check_no_infra_failure_and_nonzero_std` (also replace groups whose trials all scored the same) |
| `LR` | 1e-6 | learning rate (constant, Adam, weight decay 0.1) |
| `SAVE_INTERVAL` | 30 | steps between checkpoints |
| `INIT_LOAD` | `$MODEL_ROOT/Qwen3.8-27B_torch_dist` | torch_dist checkpoint the actor starts from; also the KL reference |
| `MODEL_DIR` | `$MODEL_ROOT/Qwen3.8-27B` | HF checkpoint: tokenizer, config, engine boot weights |
| `HARBOR_AGENT_MAX_ITERATIONS` | 10000 | agent turn cap (effectively off) |
| `HARBOR_MAX_SEQ_LEN` | `MAX_SEQ_LEN` | agent context bound; Terminus-2 summarizes before reaching it |
| `HARBOR_AGENT_TIMEOUT_SEC` | 28800 | agent-phase budget per trial; the trial is still tested and trained on |
| `HARBOR_AGENT_CALL_TIMEOUT_SEC` | 36000 | client deadline per trial; when it fires the sample is dropped |
| `HARBOR_TERMINUS_PARSER` | `xml` | Terminus-2 response format |
| `HARBOR_INTERLEAVED_THINKING` | `true` | keep the model's reasoning between turns |
| `HARBOR_TERMINUS_ENABLE_SUMMARIZE` | `true` | summarize the history near the context bound |
| `SANDBOX_CPUS`, `SANDBOX_MEMORY_MB`, `SANDBOX_STORAGE_MB` | 1, 2048, 8192 | GKE sandbox size (other backends use the task's own) |
| `GKE_POD_READY_TIMEOUT_SEC` | 1200 | GKE pod readiness deadline |
| `HARBOR_COMMIT` | `ab9eaefa...` | Harbor pin |
| `EXTRA_ARGS` | | extra Miles/Megatron flags, appended last |

Fixed in `train.py`: temperature 1.0, SGLang context 262144 and memory fraction 0.8, full
recomputation, bf16 gradient reduction, CPU-offloaded Adam, `--use-rollout-logprobs`, pause mode
`in_place`.

## Notes

- **Known issue: `train/grad_norm` is NaN, so gradient clipping is inactive.** The logged gradient
  norm is NaN (occasionally inf) from the first step. Clipping applies only to a finite norm, so
  `--clip-grad` has no effect (an inf norm scales that step's update to zero). Adam does not read
  the norm, so training otherwise proceeds. The configuration involved is
  `--optimizer-cpu-offload` + `--use-precision-aware-optimizer` + `--main-grads-dtype bf16`; the
  cause is not isolated yet.
- **Chat template "reasoning effort" line.** By default the Qwen3.8 chat template adds a system
  line "Reasoning effort is set to xhigh. Please think carefully ...". Training prompts (rendered
  by the Miles session server) have no such line. When you serve a trained checkpoint, send
  `chat_template_kwargs={"reasoning_effort": "medium"}` (the branch that emits no system line) or
  the model sees a prompt it never trained on and reasons far longer.
- **`INIT_LOAD`, not `MODEL_DIR`, decides the starting weights.** A fresh run initialises the actor
  from `--ref-load` (`INIT_LOAD`): with nothing under its own `checkpoints/`, Miles loads it
  weights-only. Pointing `MODEL_DIR` at an HF export of a trained model is not enough and silently
  trains from the base model. `INIT_LOAD` is also the KL reference, which here only changes the
  logged KL (the KL coefficients are 0). The train log names the checkpoint it loaded.
- **Out of memory in the first actor phase**: lower `MAX_RESPONSE_LEN`; add recomputation
  (`EXTRA_ARGS="--recompute-num-layers 2"`); or set `MAX_SEQ_LEN=98304` (with it
  `HARBOR_MAX_SEQ_LEN`), accepting that the longest trajectories are truncated. Do not enable
  `--offload-train`: with a full-weight actor Miles keeps no CPU copy of the parameters, so the
  actor wakes with garbage weights and nothing gets solved.
- **Checkpoints and disk.** Every save is kept, about 54 GB each. To keep only the latest, pass
  `EXTRA_ARGS="--save-retain-interval $SAVE_INTERVAL"`: Megatron then deletes the previous save
  unless its iteration is a multiple of the retain interval, and saves land on iterations 29, 59, ...
- **Host memory on the head node.** The session servers run as children of Ray workers and inherit
  the raylet's environment. With glibc's default arena count their freed heap is not returned to
  the OS, and a long run can exhaust the head node's memory. Start Ray with `MALLOC_ARENA_MAX=2`.
- **Sandbox network.** Some tasks' tests download tools when they run (for example `uvx` fetching
  a Python build and pytest). With sandbox egress blocked their reward is always 0, whatever the
  agent did; with egress open the agent has internet for the whole trial. The GKE configuration
  this was run with blocked egress (`GKE_CLUSTER_DEFAULT_DENY_EGRESS=true` asserts a namespace
  default-deny policy). Check your tasks' tests before choosing.
- **Timeouts.** `HARBOR_AGENT_TIMEOUT_SEC` (8 h) ends the agent phase; the trial is still tested and
  trained on. It is a cap: most trials end far earlier. `HARBOR_AGENT_CALL_TIMEOUT_SEC` is the
  deadline for the whole trial (sandbox start, agent, tests); when it fires the sample is lost, so
  keep it above the agent budget.
- **Uniform groups.** A group whose 8 trials all scored the same has zero advantage and adds no
  gradient. `DYNAMIC_SAMPLING_FILTER=check_no_infra_failure_and_nonzero_std` replaces such groups,
  at the cost of more rollouts per step; removing tasks the base model always or never solves from
  `PROMPT_DATA` is cheaper.
- **Infrastructure failures.** Trials that never reach the tests (`AgentError`, `Flushed`,
  `Unknown`) would otherwise train as reward 0; `check_no_infra_failure` replaces their group.
  Time-limit and length-limit outcomes still count.
- **Concurrency.** A multi-turn trial keeps its prefix in the engine's KV cache for its whole life,
  including while its sandbox runs commands. Before raising `AGENT_MAX_CONCURRENT`, check the
  per-engine KV capacity in the engines' `KV Cache is allocated` log line.
