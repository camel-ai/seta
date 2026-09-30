# Inkling-Small · GRPO · terminal-agent RL

Full-parameter GRPO of [Inkling-Small](https://huggingface.co/thinkingmachines/Inkling-Small)
(Thinking Machines Lab; 276B-parameter MoE with about 12B active, 42 layers) with
[Miles](https://github.com/radixark/miles). Every rollout is one terminal task: the
[Harbor](https://github.com/Michaelsqj/harbor) agent server starts the task's sandbox (Daytona,
Modal, GKE or Docker), runs the agent against the policy and verifies the result; the verifier's
score is the reward. Two agents are supported, one script each:

| script | agent | how the model's output is read |
|---|---|---|
| `run_harbor_camel.sh` | Harbor's **CAMEL** agent, Inkling's native tool calls | the Miles session server parses Inkling's completions; SGLang's parsers are off |
| `run_harbor_terminus2.sh` | Harbor's **Terminus-2** agent, JSON commands in plain text, interleaved thinking | SGLang's Inkling tool and reasoning parsers split the output; needs a patched SGLang (Step 1) |
| `train.py` | the Miles launcher both scripts call (`python train.py train --help`) | |
| `prepare_model.py` | downloads the model and converts it for Megatron (Step 3) | |

Everything else (parallelism, serving, GRPO settings, batch shape) is the same for both.

Inkling-Small is released under the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0)
together with the [Thinking Machines Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy);
your use of the model and of anything you train from it must follow both.

## Requirements

- **GPUs: 8 nodes x 8 H200 (141 GB).** Training and rollout share every GPU. The actor runs
  tensor parallel 8 with sequence parallel, expert parallel 8 and one pipeline stage per node
  (TP8/PP8/EP8, no context or data parallelism); each node also serves one TP8 SGLang engine
  (8 engines). 6 or 7 nodes also work (`NUM_NODES`, 6 or 7 pipeline stages) with less memory
  headroom.
- **Host memory:** the training weights and the Adam state are offloaded to CPU memory while
  SGLang serves, so plan for several hundred GB of free RAM per node.
- **Disk**, on storage every node can read at the same path: the Hugging Face checkpoint and its
  Megatron conversion (about 550 GB each: 276B parameters in BF16), plus about 550 GB per saved
  checkpoint under `RUNS_ROOT`.
- **Head node:** also runs the Harbor agent server and the Miles session servers (CPU).
- **Accounts:** Hugging Face (the model is not gated), one sandbox provider (Daytona, Modal or
  your own GKE cluster; or a local Docker daemon), optionally Weights & Biases.

## Versions

| component | CAMEL (`run_harbor_camel.sh`) | Terminus-2 (`run_harbor_terminus2.sh`) |
|---|---|---|
| Docker image | `radixark/miles@sha256:946f29396ac313d03b50ac0c0c8930b7eaaa403ae9e7d6115595d57e59cdad12` | same |
| Miles | [`Michaelsqj/miles@38bef2a605191c137323976c211ebdac467b3295`](https://github.com/Michaelsqj/miles/commit/38bef2a605191c137323976c211ebdac467b3295): the session server parses raw Inkling completions (tool calls, reasoning), so SGLang's parsers can stay off (later merged upstream as radixark/miles#2302); session servers get 180 s to start | [`Michaelsqj/miles@80a25cb568982b8e445498ab7d3a0fc3d5d3670e`](https://github.com/Michaelsqj/miles/commit/80a25cb568982b8e445498ab7d3a0fc3d5d3670e): keeps Inkling's ordered thinking/text blocks when the agent replays its history, so the replayed turns re-tokenize to the generated tokens; session-server startup time scales with the pool size |
| SGLang | `sgl-project/sglang@cb05a44f35a7c9e27e46d74112cc841ca674ef43` (`sglang-miles` branch) | [`michaelsqj/sglang@baccf651fe984a825c35b2554db3572b31e9e25a`](https://github.com/michaelsqj/sglang/commit/baccf651fe984a825c35b2554db3572b31e9e25a): `cb05a44f` plus an ordered `content_blocks` field in chat responses for Inkling's thinking/text blocks (not merged upstream) |
| Harbor | [`Michaelsqj/harbor@acac1c20e0350f70c60fc6a0755a99d9302b90dd`](https://github.com/Michaelsqj/harbor/commit/acac1c20e0350f70c60fc6a0755a99d9302b90dd): CAMEL's `response_feedback` switches reach the agent, CAMEL library logging off; pins CAMEL to [`Michaelsqj/camel@1c42729b29b6f9f216bab2d08a5e7a52bbb6cf8e`](https://github.com/Michaelsqj/camel/commit/1c42729b29b6f9f216bab2d08a5e7a52bbb6cf8e) in its `uv.lock` | [`Michaelsqj/harbor@1e02f96f00a18b5e9158a4baffbaa2bb5a0b2146`](https://github.com/Michaelsqj/harbor/commit/1e02f96f00a18b5e9158a4baffbaa2bb5a0b2146): agent server with one subprocess worker per trial |
| Model | `thinkingmachines/Inkling-Small@8cc5877b44d343f88b92086aa1fb72897950f06a` | same |

`HARBOR_COMMIT` is installed automatically (Step 4). The scripts warn when Miles or SGLang (head
node) is not at the commit above; `STRICT_PINS=1` turns that into an error. They refuse to start
when a feature they depend on is missing from the Miles, SGLang or Harbor checkout (see Notes).

## Step 1: Start the container on every node

```bash
docker pull radixark/miles@sha256:946f29396ac313d03b50ac0c0c8930b7eaaa403ae9e7d6115595d57e59cdad12
docker run -d --name miles --gpus all --network host --ipc host --shm-size 64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /path/to/models:/root/models \
  -v /path/to/data:/data \
  -v /path/to/seta:/workspace/seta \
  radixark/miles@sha256:946f29396ac313d03b50ac0c0c8930b7eaaa403ae9e7d6115595d57e59cdad12 sleep infinity
docker exec -it miles bash
```

Add your cluster's InfiniBand/RDMA device flags if it needs them, and
`-v /var/run/docker.sock:/var/run/docker.sock` on the head node for `SANDBOX_BACKEND=docker`.

Inside the container on **every node**, check out the Miles and SGLang commits of the script you
will run. Miles runs from `/root/miles` (the scripts put it on `PYTHONPATH`) and SGLang is
imported from `/sgl-workspace/sglang`, so a checkout is all it takes:

```bash
# run_harbor_camel.sh
git -C /root/miles fetch https://github.com/Michaelsqj/miles.git 38bef2a605191c137323976c211ebdac467b3295
git -C /root/miles checkout --detach 38bef2a605191c137323976c211ebdac467b3295
git -C /sgl-workspace/sglang fetch https://github.com/sgl-project/sglang.git cb05a44f35a7c9e27e46d74112cc841ca674ef43
git -C /sgl-workspace/sglang checkout --detach cb05a44f35a7c9e27e46d74112cc841ca674ef43

# run_harbor_terminus2.sh
git -C /root/miles fetch https://github.com/Michaelsqj/miles.git 80a25cb568982b8e445498ab7d3a0fc3d5d3670e
git -C /root/miles checkout --detach 80a25cb568982b8e445498ab7d3a0fc3d5d3670e
git -C /sgl-workspace/sglang fetch https://github.com/michaelsqj/sglang.git baccf651fe984a825c35b2554db3572b31e9e25a
git -C /sgl-workspace/sglang checkout --detach baccf651fe984a825c35b2554db3572b31e9e25a

# check: SGLang is imported from the checkout
python -c "import os, sglang; print(os.path.dirname(sglang.__file__))"   # /sgl-workspace/sglang/python/sglang
```

If the check prints another path, install the checkout with
`pip install --no-deps -e /sgl-workspace/sglang/python`. The Terminus-2 SGLang commit changes
only Python files on top of the CAMEL one, so no kernels need rebuilding.

To switch between the two scripts later, stop the running job and repeat this on every node.

## Step 2: Start Ray

```bash
# head node
ray start --head --node-ip-address "$HEAD_IP" --port 6379 \
  --dashboard-host 0.0.0.0 --dashboard-port 8265 --num-gpus 8 --disable-usage-stats
# every other node
ray start --address "$HEAD_IP:6379" --node-ip-address "$NODE_IP" --num-gpus 8 --disable-usage-stats
# head node: expect 64 GPUs
ray status
```

The run scripts connect to this cluster (they never start or stop Ray) and must run on the head
node, inside the container. Before submitting, Miles kills leftover `sglang` and `miles`
processes on the head node.

## Step 3: Prepare the model

On one node, inside the container, with 8 free GPUs:

```bash
cd /workspace/seta
PYTHONPATH=/root/miles python scripts/miles/examples/inkling_grpo/prepare_model.py --model-root /root/models
```

This downloads `thinkingmachines/Inkling-Small` at the tested revision with the `hf` CLI
(`pip install -U huggingface_hub` if it is missing) and converts it to Megatron's `torch_dist`
format (TP8/EP8; the actor re-shards it on load). Both steps can be re-run and resume. Output:

```
/root/models/Inkling-Small/              Hugging Face checkpoint (SGLang serves it; Miles reads its config)
/root/models/Inkling-Small_torch_dist/   Megatron checkpoint (actor start weights and KL reference)
```

Every node must see both directories at the same path (shared storage, or copy them).

## Step 4: Sandbox backend

Pick one with `SANDBOX_BACKEND` (in `.env`, Step 6). Trials run in sandboxes of that backend;
the agent itself runs in the Harbor agent server on the head node.

#### Daytona

The default. Set `DAYTONA_API_KEY` (https://app.daytona.io -> API keys), and `DAYTONA_API_URL`
for a non-default endpoint.

#### Modal

Set `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`, or run `modal token new` in the container on the head
node (writes `~/.modal.toml`).

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

Uses the Docker daemon of the head node: mount its socket into the container (Step 1). Only
practical with a small `AGENT_MAX_CONCURRENT`.

#### Sandbox size

On GKE these scripts size every sandbox at 1 CPU, 2048 MiB memory and 6144 MiB storage
(`SANDBOX_CPUS`, `SANDBOX_MEMORY_MB`, `SANDBOX_STORAGE_MB`), the size they were run with; size
the cluster for `AGENT_MAX_CONCURRENT` (160) sandboxes. The other backends use each task's own
resources.

#### Harbor

Harbor is installed by the run script on first use, at `HARBOR_COMMIT`, into its own virtualenv
under `~/.cache/seta/harbor/<commit>` (`HARBOR_HOME`) on the head node. It needs Python 3.12 (uv
fetches it when installed: `pip install uv`). To install ahead of time:

```bash
HARBOR_COMMIT=acac1c20e0350f70c60fc6a0755a99d9302b90dd bash scripts/miles/common/harbor_install.sh   # CAMEL
HARBOR_COMMIT=1e02f96f00a18b5e9158a4baffbaa2bb5a0b2146 bash scripts/miles/common/harbor_install.sh   # Terminus-2
```

## Step 5: Dataset

No data ships with this recipe. It needs two inputs, described in
[`../../data/README.md`](../../data/README.md):

- `PROMPT_DATA`: a JSONL with one row per task (`metadata.instance_id` names the task).
- `TASKS_DIR`: the Harbor task directories those rows name (`<task>/task.toml`, `instruction.md`,
  `environment/`, `tests/`).

The example files in `scripts/miles/data` are Terminal-Bench examples for smoke tests only; do not
train on them. Train on your own task set, for example the
[SETA-Env](https://huggingface.co/datasets/camel-ai/SETA-Env) environments:

```bash
# download the tasks as described in ../../data/README.md, e.g. into dataset/seta-env-final/<task>/
python scripts/miles/common/build_prompt_dataset.py build \
  --tasks-dir dataset/seta-env-final --agent-name camel --output train.jsonl
```

`metadata.agent_name` is informational; the script decides the agent. The scripts check that
every row has a task directory before they start. GRPO learns only from prompt groups whose
samples get different rewards, so tasks the model sometimes solves are the useful ones.

## Step 6: Fill in your settings

```bash
cd scripts/miles/examples/inkling_grpo
cp env.example .env     # then edit .env
```

Both scripts read `.env` from this folder (or the file named by `ENV_FILE`); values there and in
the environment override the script defaults.

| variable | required | meaning |
|---|---|---|
| `HEAD_IP` | yes (or `CLUSTER_CONFIG`) | Ray head node IP; the scripts run on this node |
| `CLUSTER_CONFIG` | | a copy of [`../../cluster.example.yaml`](../../cluster.example.yaml) with your IPs, instead of `HEAD_IP` |
| `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME` | | interconnect interface, when it is not the default route |
| `MODEL_ROOT` | yes | where `prepare_model.py` wrote the model (default `/root/models`) |
| `MODEL_DIR`, `TORCH_DIST_DIR` | | the two checkpoints, if not under `MODEL_ROOT` |
| `LOAD_DIR` | | start from an earlier run's `checkpoints/` (see Outputs & resume) |
| `PROMPT_DATA`, `TASKS_DIR` | yes | Step 5 |
| `SANDBOX_BACKEND` | yes | `daytona` (default), `modal`, `gke` or `docker` |
| `DAYTONA_API_KEY`, `DAYTONA_API_URL` | daytona | Step 4 |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | modal | Step 4 (or `~/.modal.toml`) |
| `GKE_*` | gke | Step 4 |
| `WANDB_API_KEY` | | turns W&B logging on; passed to Miles as `--wandb-key` |
| `WANDB_PROJECT`, `WANDB_ENTITY` | | W&B project (default `seta`) and team (default: your user) |
| `RUNS_ROOT`, `RUN_NAME` | | outputs go to `$RUNS_ROOT/$RUN_NAME` (default `./runs/inkling-grpo-<agent>-<UTC time>`); `RUNS_ROOT` must be shared storage, every node writes checkpoint shards there |
| `TORCHINDUCTOR_CACHE_DIR` | | compile cache, default `$RUNS_ROOT/torchinductor_cache`; keep it across runs |
| `MILES_DIR`, `SGLANG_DIR`, `MEGATRON_PATH` | | checkouts in the container (defaults `/root/miles`, `/sgl-workspace/sglang`, `/root/Megatron-LM`) |
| `STRICT_PINS` | | `1`: fail instead of warn when Miles/SGLang are not at the tested commits |
| `HARBOR_COMMIT`, `HARBOR_HOME` | | Harbor commit and install location (defaults: the script's pin, `~/.cache/seta/harbor`) |

The training settings are listed in the [Configuration reference](#configuration-reference).

## Step 7: Launch

Run on the head node inside the container, in `tmux` or `screen` (a run takes many hours).
`DRY_RUN=1` prints the resolved `train.py` command and exits before anything is installed or
started.

### CAMEL agent

```bash
cd /workspace/seta
DRY_RUN=1 bash scripts/miles/examples/inkling_grpo/run_harbor_camel.sh
bash scripts/miles/examples/inkling_grpo/run_harbor_camel.sh
```

### Terminus-2 agent

Needs the Miles and SGLang commits of this script on every node (Step 1).

```bash
cd /workspace/seta
DRY_RUN=1 bash scripts/miles/examples/inkling_grpo/run_harbor_terminus2.sh
bash scripts/miles/examples/inkling_grpo/run_harbor_terminus2.sh
```

Each script checks the model, the checkouts and the prompt data, installs Harbor, starts the
Harbor agent server on the head node (stopped again when the script exits) and submits the Miles
job to Ray. A quick end-to-end check with a small batch:

```bash
NUM_ROLLOUT=3 ROLLOUT_BATCH_SIZE=4 N_SAMPLES_PER_PROMPT=2 OVER_SAMPLING_BATCH_SIZE=6 \
  bash scripts/miles/examples/inkling_grpo/run_harbor_camel.sh
```

(`GLOBAL_BATCH_SIZE` and `AGENT_MAX_CONCURRENT` follow the batch shape.)

## Monitor

- `$RUNS_ROOT/$RUN_NAME/logs/train.log`: the Miles job (rollout progress, rewards, losses).
- `$RUNS_ROOT/$RUN_NAME/logs/agent_server.log`: the Harbor agent server.
- `$RUNS_ROOT/$RUN_NAME/trials/`: one folder per trial (agent log, trajectory, verifier output).
- Ray dashboard: `http://<HEAD_IP>:8265`.
- Harbor dashboard: `http://<HEAD_IP>:11000/dashboard/` (`AGENT_SERVER_PORT`).
- W&B, when `WANDB_API_KEY` is set: KL to the reference model and the entropy are logged every
  step (the KL coefficient is 0, so they are only observed).

## Outputs and resume

```
$RUNS_ROOT/$RUN_NAME/
  logs/train.log, logs/agent_server.log
  trials/                  Harbor trial artifacts
  checkpoints/             Megatron torch_dist checkpoints, weights only
  dump_details/            per-rollout samples and token ids (DUMP_DETAILS=1)
  wandb/
  run_manifest.md          commits, backend, prompt data (row count, sha256)
  launcher/                copies of the run script and train.py
$RUNS_ROOT/torchinductor_cache/   compile cache shared by runs
```

A checkpoint is written every `SAVE_INTERVAL` rollouts and after the last one. Checkpoints hold
the model weights only (with the Adam state a checkpoint of this model is several TB), so a run
cannot be resumed exactly. To continue from saved weights, start a new run with
`LOAD_DIR=$RUNS_ROOT/<old run>/checkpoints`: the actor loads the latest checkpoint there, the
optimizer state and the rollout count start fresh, and the KL reference stays the base model.
Miles' `tools/convert_torch_dist_to_hf.py` converts a checkpoint to Hugging Face format; these
scripts do not wrap it.

## Configuration reference

Environment variables (or `.env`) read by both scripts; defaults are the configuration these
recipes were run with.

| variable | default | meaning |
|---|---|---|
| `NUM_NODES` | `8` | nodes of 8 GPUs; 6, 7 or 8 (one pipeline stage and one SGLang engine per node) |
| `GPUS_PER_NODE` | `8` | only 8 is supported |
| `LR` | `1e-5` | constant learning rate (Adam, weight decay 0.1) |
| `NUM_ROLLOUT` | `100` | rollouts, one optimizer step each |
| `OVER_SAMPLING_BATCH_SIZE` | `20` | prompt groups started per rollout |
| `ROLLOUT_BATCH_SIZE` | `16` | complete groups kept per rollout (groups with an aborted sample are dropped) |
| `N_SAMPLES_PER_PROMPT` | `8` | trials per prompt (the GRPO group) |
| `GLOBAL_BATCH_SIZE` | `128` | `ROLLOUT_BATCH_SIZE x N_SAMPLES_PER_PROMPT`: one step per rollout |
| `SAVE_INTERVAL` | `99` (CAMEL), `50` (Terminus-2) | checkpoint every N rollouts (plus the last) |
| `MAX_SEQ_LEN` | `32768` | tokens of one training sample (prompt plus all turns); longer samples are truncated |
| `MAX_RESPONSE_LEN` | `16384` | tokens of one model turn |
| `SESSION_SERVERS` | `64` | Miles session-server processes on the head node |
| `SESSION_SERVER_PORT` | `30000` | first port; they use `[port, port + SESSION_SERVERS)` |
| `DUMP_DETAILS` | `1` | write `dump_details/` |
| `EXTRA_ARGS` | | extra Miles/Megatron/SGLang flags, appended last (the last occurrence of a flag wins) |
| `AGENT_MAX_CONCURRENT` | `160` | trials Harbor runs at once; default `OVER_SAMPLING_BATCH_SIZE x N_SAMPLES_PER_PROMPT` |
| `AGENT_SERVER_PORT` | `11000` | Harbor agent server port on the head node |
| `HARBOR_AGENT_MAX_ITERATIONS` | `50` | agent turns per trial |
| `HARBOR_MAX_SEQ_LEN` | `1048576` | the agent's context bound (separate from `MAX_SEQ_LEN`) |
| `HARBOR_AGENT_CALL_TIMEOUT_SEC` | `10800` | client deadline for one whole trial (queue, sandbox, agent, verification) |
| `HARBOR_AGENT_TIMEOUT_MULTIPLIER` | `12` | scales each task's own agent timeout |
| `AGENT_MODEL_NAME` | `inkling-small` | model name the agent sends |
| `SANDBOX_CPUS` | `1` | GKE only: CPUs per sandbox (other backends use the task's value) |
| `SANDBOX_MEMORY_MB` | `2048` | GKE only: memory per sandbox |
| `SANDBOX_STORAGE_MB` | `6144` | GKE only: ephemeral storage per sandbox |
| `HARBOR_INTERLEAVED_THINKING` | `true` | Terminus-2 only: earlier turns keep their reasoning |

Fixed in `train.py` (edit it, or override with `EXTRA_ARGS`): GRPO with clip 0.2/0.28 and
truncated importance sampling, rollout routing replay for the MoE layers, rollout temperature 1,
full activation recomputation, micro-batch 1, and SGLang at memory fraction 0.75, 256 running
requests and 1,048,576 KV tokens per engine.

Not available at these Harbor commits: `HARBOR_TERMINUS_PARSER`,
`HARBOR_TERMINUS_ENABLE_SUMMARIZE`, `HARBOR_CAMEL_MAX_COMPACTIONS` (the scripts unset them),
`HARBOR_AGENT_TIMEOUT_SEC`, `HARBOR_EXTRA_ARTIFACTS` and `HARBOR_EXTRA_COLLECT` (leave unset).

## Notes

- **Learning rate.** The default is `1e-5`. For full fine-tuning of this model `3e-5` was the
  highest learning rate that trained stably; `5e-5` and `1e-4` collapsed. Watch the logged KL to
  the reference and the entropy even though neither is in the loss, and watch for the policy
  learning to end hard tasks early.
- **Serving on the training GPUs.** One TP8/EP1 engine per node at memory fraction 0.75. TP4
  engines leave no room for the weight update after each step, and SGLang expert parallelism
  does not shrink the per-GPU footprint. The large KV pool (1M tokens per engine) avoids request
  retractions with 160 concurrent multi-turn trials. The router keeps each session on one
  engine (prefix cache) and places new sessions on the least-loaded engine.
- **Training mesh and memory.** TP8/PP8/EP8 without context parallelism, with `MAX_SEQ_LEN`
  capping each sample: without the cap, a single long trajectory can run the MoE layers out of
  memory. Context parallel 2 with all-gather (`--context-parallel-size 2 --allgather-cp`) worked
  in offline replays but hit NCCL timeouts in live runs on H200; avoid it. The offload flags in
  `train.py` (`--offload-train-target cpu --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer`) are required: without them
  creating the Adam state runs out of GPU memory. Keep `TORCHINDUCTOR_CACHE_DIR` across runs:
  with a cold cache the first actor step spends a long time compiling.
- **Storage.** Checkpoints are weights only (`--no-save-optim`); keep `SAVE_INTERVAL` sparse and
  `RUNS_ROOT` on a large volume. `DUMP_DETAILS=1` and the trial folders also grow with every
  rollout.
- **Session servers.** 64 session-server processes start at once and each imports the tokenizer
  stack; Miles' old 30 s readiness deadline rejects the pool before the first rollout. Both Miles
  commits give them longer, and the scripts refuse a Miles with the 30 s deadline.
- **Token-in/token-out.** Training uses the exact tokens the engine produced, so the agent's
  replayed history must re-tokenize to them. CAMEL: SGLang's Inkling parsers must stay off
  (`train.py` refuses them) because the Miles session server parses Inkling's completions. Terminus-2:
  the SGLang and Miles commits above keep Inkling's interleaved thinking/text blocks in order;
  the scripts refuse checkouts without them.
- **Terminus-2 protocol.** Inkling sometimes emits its native tool-call blocks inside Terminus-2's
  JSON protocol. When such trajectories still succeed they get a positive advantage and the
  habit is reinforced; at `5e-5` this degraded the Terminus-2 run quickly. Check the trials for
  native tool blocks, and prefer the CAMEL script, which uses the native format.
- **CAMEL behavior.** A malformed or length-cut turn ends the trajectory as the policy wrote it:
  no corrective feedback, no retry. The Harbor commit routes these switches to the agent (with
  older commits they are silently ignored), and it turns off CAMEL's library logging, which
  otherwise logs every request payload and can fill the run disk. The script checks both.
- **Head-node processes.** Miles kills leftover `sglang` and `miles` processes on the head node
  before it submits a job; do not share the head node with another Miles job.
