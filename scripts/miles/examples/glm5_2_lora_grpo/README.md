# GLM-5.2 · LoRA GRPO · terminal-agent RL

Trains a LoRA adapter (rank 16, alpha 32, on the attention and MLA projections)
on [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) (744B total / 40B active
parameters, MoE with DeepSeek sparse attention) with GRPO in
[Miles](https://github.com/radixark/miles). Rollouts run through the
[Harbor](https://github.com/Michaelsqj/harbor) agent server: for every sample it
starts the task's sandbox (Daytona, Modal, GKE or Docker), runs Harbor's
**Terminus-2** agent (XML actions, interleaved thinking, summarization when the
context fills) against the policy, and returns the verifier's score as the reward.
The BF16 base weights stay frozen for training; the rollout engines serve the FP8
checkpoint plus the current adapter.

| file | what it is |
|---|---|
| `run_harbor_terminus2.sh` | the training entry point |
| `train.py` | the Miles launcher it calls (`python train.py train --help`) |
| `env.example` | every setting you fill in; copy to `.env` |
| `prepare_model.sh` | downloads the BF16 and FP8 checkpoints at pinned revisions |
| `setup_megatron_bridge.sh` | checks out the Megatron-Bridge commit this recipe needs |
| `megatron_bridge_sparse_mla_bwd.patch` | the same Megatron-Bridge fix, for a checkout of your own |

Only the Terminus-2 agent is supported here (`HARBOR_AGENT_NAME=terminus-2`).

## Requirements

- **GPUs:** 4 nodes x 8 H200 (141 GB), or 8 nodes with `NUM_NODES=8`. Training and
  rollout are colocated: every GPU alternates between Megatron training (tensor
  parallel 8, expert parallel across all GPUs) and the SGLang engines (one FP8
  engine per 16 GPUs: 2 engines on 4 nodes, 4 on 8 nodes).
- **Host memory:** large. The optimizer runs on the CPU, and the engines keep a CPU
  copy of their FP8 base weights so that only the adapter is synced after each step.
- **Disk:** about 2.3 TB for the two checkpoints on storage every node can read;
  about 500 GB of fast node-local disk per node for `OFFLOAD_DIR` (the training
  state is parked there during rollout); room for checkpoints and trial logs under
  `RUNS_ROOT`, also on shared storage.
- **Shared paths:** this repository, `MODEL_ROOT`, `RUNS_ROOT` and
  `MEGATRON_BRIDGE_DIR` must resolve to the same content at the same path on
  every node (Ray workers import this repository's `common/` modules).
- **Accounts:** a sandbox provider (Daytona or Modal, or your own GKE cluster);
  optionally Weights & Biases. The GLM-5.2 weights are public on Hugging Face.

## Versions

| component | version |
|---|---|
| Docker image | `radixark/miles@sha256:616ac8b9fd1e51fabcafefa47faddb67fdac3148f9d53a631dcc9606c88300cd` (`radixark/miles:release-v0.1.0-ci`) |
| Miles | [`Michaelsqj/miles@5479631b81d0747607cf424439299b766fdfe79d`](https://github.com/Michaelsqj/miles/commit/5479631b81d0747607cf424439299b766fdfe79d): `radixark/miles@22f268ef` plus a configurable session-server startup timeout (`--session-server-startup-timeout-secs`), session-server stage timing, incremental rollout routing replay outside the `retract` pause mode (radixark/miles#2834), and a `flush_cache` that waits for the engines to drain instead of failing after 60 s |
| SGLang | `sgl-project/sglang@cb05a44f35a7c9e27e46d74112cc841ca674ef43` (branch `sglang-miles`, shipped in the image) |
| Megatron-Bridge | [`Michaelsqj/Megatron-Bridge@e91c492328700f9cd59b8d1a82d40a5e54fbd835`](https://github.com/Michaelsqj/Megatron-Bridge/commit/e91c492328700f9cd59b8d1a82d40a5e54fbd835): `radixark/Megatron-Bridge@7f0fb345` plus the one-line sparse-MLA backward fix in `megatron_bridge_sparse_mla_bwd.patch` (see Known issues) |
| Harbor | [`Michaelsqj/harbor@5af13d825dd8b4b8ae330133ce25ecb3019653ba`](https://github.com/Michaelsqj/harbor/commit/5af13d825dd8b4b8ae330133ce25ecb3019653ba) (`HARBOR_COMMIT`) |
| Model | `zai-org/GLM-5.2@b4734de4facf877f85769a911abafc5283eab3d9` (BF16), `zai-org/GLM-5.2-FP8@ba978f7d347eaf65d22f1a86833408afdb953541` (rollout) |

The launcher warns when Miles, SGLang or Megatron-Bridge is not at these commits;
`STRICT_PINS=1` turns the warning into an error. It refuses to start without the
Megatron-Bridge fix, without the Miles startup-timeout flag, or with a Harbor
that lacks the two trial features it relies on (see Notes).

## Step 1: Start the container on every node

```bash
docker pull radixark/miles@sha256:616ac8b9fd1e51fabcafefa47faddb67fdac3148f9d53a631dcc9606c88300cd
docker run -d --name miles --gpus all --network host --ipc host --shm-size 64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /shared/models:/root/models \
  -v /shared/datasets:/datasets \
  -v /shared/seta:/workspace/seta \
  -v /local/nvme/miles_offload:/root/miles_train_offload \
  radixark/miles@sha256:616ac8b9fd1e51fabcafefa47faddb67fdac3148f9d53a631dcc9606c88300cd \
  sleep infinity
docker exec -it miles bash
```

Replace the host paths with yours: `/shared/...` on storage every node mounts,
`/local/nvme/...` on each node's own fast disk. Then, inside the container on
**every node**, check out the pinned Miles commit (the image ships upstream Miles)
and the Megatron-Bridge commit with the fix:

```bash
git -C /root/miles fetch https://github.com/Michaelsqj/miles.git 5479631b81d0747607cf424439299b766fdfe79d
git -C /root/miles checkout --detach 5479631b81d0747607cf424439299b766fdfe79d
cd /workspace/seta
bash scripts/miles/examples/glm5_2_lora_grpo/setup_megatron_bridge.sh   # -> /root/Megatron-Bridge-glm5_2
```

To use a Megatron-Bridge checkout of your own instead (at `radixark/Megatron-Bridge@7f0fb345`),
apply the fix and point `MEGATRON_BRIDGE_DIR` at it:

```bash
git -C /path/to/Megatron-Bridge apply /workspace/seta/scripts/miles/examples/glm5_2_lora_grpo/megatron_bridge_sparse_mla_bwd.patch
```

The launcher puts `${MEGATRON_BRIDGE_DIR}/src` first on `PYTHONPATH`, for itself
and for every Ray worker, so it shadows the Megatron-Bridge installed in the image.

## Step 2: Start Ray

Start Ray inside the container, head node first. Cap glibc's malloc arenas and
interleave host memory across NUMA nodes (see Notes for why; `apt-get install -y numactl`
if the image lacks it):

```bash
# head node
MALLOC_ARENA_MAX=2 numactl --interleave=all ray start --head --node-ip-address <HEAD_IP> \
  --port 6379 --dashboard-host 0.0.0.0 --dashboard-port 8265 --num-gpus 8 --disable-usage-stats
# every other node
MALLOC_ARENA_MAX=2 numactl --interleave=all ray start --address <HEAD_IP>:6379 \
  --node-ip-address <NODE_IP> --num-gpus 8 --disable-usage-stats
# check: 32 GPUs (64 with NUM_NODES=8)
ray status
```

The launcher runs on the head node and submits the job to this cluster.

## Step 3: Prepare the model

Once, from any container that sees `MODEL_ROOT`:

```bash
cd /workspace/seta
MODEL_ROOT=/root/models bash scripts/miles/examples/glm5_2_lora_grpo/prepare_model.sh
```

Result: `/root/models/GLM-5.2` (BF16, about 1.5 TB; the training model, loaded
directly through Megatron-Bridge, no conversion) and `/root/models/GLM-5.2-FP8`
(about 0.76 TB; what the rollout engines serve). Re-running resumes a partial download.

## Step 4: Sandbox backend

Pick one with `SANDBOX_BACKEND` (default `daytona`). Rewards depend on the sandbox
(resources, network access, image build); the recipe was run on GKE with 1 CPU,
2048 MiB memory and 8192 MiB ephemeral storage per sandbox, which this script
sets by default for `SANDBOX_BACKEND=gke`. The other backends use each task's
own resources.

#### Daytona

Set `DAYTONA_API_KEY` (from the Daytona dashboard) and, for a non-default
endpoint, `DAYTONA_API_URL`.

#### Modal

Run `modal token new` in the container on the head node, or set `MODAL_TOKEN_ID`
and `MODAL_TOKEN_SECRET`.

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

`SANDBOX_BACKEND=docker` runs sandboxes on the head node's Docker daemon; use it
only for small smoke tests.

The launcher installs Harbor at `HARBOR_COMMIT` into its own venv under
`~/.cache/seta/harbor/<commit>` (`HARBOR_HOME`) on first use. To install it ahead
of time on the head node:

```bash
HARBOR_COMMIT=5af13d825dd8b4b8ae330133ce25ecb3019653ba bash scripts/miles/common/harbor_install.sh
```

## Step 5: Dataset

The recipe needs `PROMPT_DATA` (a JSONL, one row per task) and `TASKS_DIR` (the
Harbor task directories the rows name); the format is in
[`../../data/README.md`](../../data/README.md). The files in `scripts/miles/data/`
are Terminal-Bench examples for smoke tests only; do not train on them. Build a
training set from your own tasks, for example
[SETA-Env](https://huggingface.co/datasets/camel-ai/SETA-Env):

```bash
cd /workspace/seta
python -m seta_env.dataset.download seta-env-final        # -> dataset/seta-env-final/<task>/
python scripts/miles/common/build_prompt_dataset.py build \
  --tasks-dir dataset/seta-env-final --agent-name terminus-2 --output /datasets/train.jsonl
# PROMPT_DATA=/datasets/train.jsonl  TASKS_DIR=/workspace/seta/dataset/seta-env-final
```

GRPO learns only from groups whose samples get different rewards, so prefer tasks
the base model solves sometimes but not always; tasks it always fails or always
solves cost a full rollout and add no gradient. Tasks that need a GPU cannot run
in CPU sandboxes; leave them out (`build_prompt_dataset.py filter --exclude`).

## Step 6: Fill in your settings

```bash
cd /workspace/seta/scripts/miles/examples/glm5_2_lora_grpo
cp env.example .env     # .env is ignored by git
```

Every script in this folder reads `.env` (or the file named by `ENV_FILE`).

| variable | required | meaning |
|---|---|---|
| `HEAD_IP` or `CLUSTER_CONFIG` | yes | Ray head address, or a copy of [`../../cluster.example.yaml`](../../cluster.example.yaml) |
| `PROMPT_DATA`, `TASKS_DIR` | yes | training prompts and their task directories (Step 5) |
| `SANDBOX_BACKEND` + its credentials | yes | Step 4 |
| `MODEL_ROOT` | yes (default `/root/models`) | where `prepare_model.sh` put the checkpoints; `MODEL_DIR` / `FP8_MODEL_DIR` override each one |
| `MEGATRON_BRIDGE_DIR` | yes (default `/root/Megatron-Bridge-glm5_2`) | Megatron-Bridge checkout with the fix |
| `OFFLOAD_DIR` | no (`/root/miles_train_offload`) | node-local disk for the offloaded training state |
| `NUM_NODES` | no (`4`) | `4` or `8`; must match the Ray cluster |
| `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME` | no | interconnect interface when it is not the default route |
| `MILES_DIR`, `MEGATRON_PATH`, `SGLANG_DIR` | no | checkouts in the image (`/root/miles`, `/root/Megatron-LM`, `/sgl-workspace/sglang`) |
| `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY` | no | W&B logging is on when `WANDB_API_KEY` is set; the key is passed to Miles as `--wandb-key` |
| `HF_TOKEN` | no | only if your Hugging Face access needs it |
| `RUNS_ROOT`, `RUN_NAME` | no (`./runs`, `glm5_2-lora-grpo-terminus2-<UTC time>`) | output folder, on shared storage |
| `HARBOR_COMMIT`, `HARBOR_HOME`, `AGENT_SERVER_PORT` | no | Harbor version, install location, agent-server port (`11000`) |
| `STRICT_PINS` | no (`0`) | `1`: fail instead of warn on a version mismatch |

The training settings are listed under Configuration reference.

## Step 7: Launch

On the head node, inside the container:

```bash
cd /workspace/seta
DRY_RUN=1 bash scripts/miles/examples/glm5_2_lora_grpo/run_harbor_terminus2.sh   # prints the resolved train.py command
tmux new -s glm5_2 'bash scripts/miles/examples/glm5_2_lora_grpo/run_harbor_terminus2.sh'
```

The script checks the checkpoints, the Miles and Megatron-Bridge checkouts and the
prompt data, installs Harbor, starts the Harbor agent server in the background
(stopped when the script exits), and submits the training job to Ray.

## Monitor

- `${RUNS_ROOT}/${RUN_NAME}/logs/train.log`: the training job's output.
- `${RUNS_ROOT}/${RUN_NAME}/logs/agent_server.log` and `trials/`: the Harbor agent
  server and one directory per trial (agent trajectory, verifier output, reward).
- Ray dashboard: `http://<HEAD_IP>:8265`. Harbor dashboard: `http://<HEAD_IP>:11000/dashboard/`.
- Miles dashboard over the per-step dumps:
  `python -m miles.dashboard.serve --dump-details ${RUNS_ROOT}/${RUN_NAME}/dump_details`.
- W&B, when `WANDB_API_KEY` is set (group = `RUN_NAME`).

## Outputs and resume

Under `${RUNS_ROOT}/${RUN_NAME}/`:

- `checkpoints/iter_<step>/adapter/`: written every `SAVE_INTERVAL` steps. It holds
  the adapter in Hugging Face PEFT format (`adapter_model.bin`,
  `adapter_config.json`), usable with PEFT or as an SGLang LoRA on
  `GLM-5.2-FP8`, plus per-rank shards and optimizer state for resuming. The base
  weights never change and are not saved.
- `checkpoints/sglang_fp8_rollout.yaml`: the generated rollout-engine config.
- `dump_details/`, `logs/`, `trials/`, `wandb/`, and `run_manifest.md` plus
  `launcher/` (versions, data checksum, a copy of the launch files).

To continue from a saved adapter, start a new run with
`EXTRA_ARGS="--lora-adapter-path <run>/checkpoints/iter_<step>/adapter"`; Miles
restores the adapter together with its optimizer and learning-rate state. This
recipe's own runs did not use it.

## Configuration reference

Environment variables (or `.env` lines); the defaults are the configuration this recipe was run with.

| variable | default | meaning |
|---|---|---|
| `NUM_NODES` | `4` | nodes of 8 GPUs: `4` (expert parallel 32, 2 engines) or `8` (expert parallel 64, 4 engines) |
| `NUM_ROLLOUT` | `200` | rollout / optimizer steps |
| `ROLLOUT_BATCH_SIZE` | `4` | prompt groups trained on per step |
| `N_SAMPLES_PER_PROMPT` | `8` | trials per prompt (GRPO group size) |
| `OVER_SAMPLING_BATCH_SIZE` | `6` | prompt groups started per step; the first `ROLLOUT_BATCH_SIZE` complete groups without an aborted sample are kept |
| `GLOBAL_BATCH_SIZE` | `ROLLOUT_BATCH_SIZE x N_SAMPLES_PER_PROMPT` | samples per optimizer step |
| `ROLLOUT_TEMPERATURE` | `0.8` | sampling temperature |
| `LR` | `3e-5` | constant learning rate (Adam, betas 0.9 / 0.98, weight decay 0.1) |
| `LORA_RANK`, `LORA_ALPHA` | `16`, `32` | adapter shape |
| `SAVE_INTERVAL` | `10` | checkpoint every N steps |
| `MAX_SEQ_LEN` | `49152` | longest training sample |
| `MAX_TOKENS_PER_GPU` | `MAX_SEQ_LEN` | token budget per GPU for dynamic batching |
| `MAX_RESPONSE_LEN` | `16384` | longest model response per turn |
| `SGLANG_CONTEXT_LENGTH` | `1048576` | rollout engine context window |
| `SGLANG_MEM_FRACTION` | `0.85` | engine static memory fraction |
| `SGLANG_LORA_FASTPATH` | `1` | SGLang's experimental LoRA forward path; `0` uses the default path |
| `SESSION_SERVERS`, `SESSION_SERVER_PORT` | `64`, `30000` | Miles session servers on ports `[30000, 30064)` |
| `SESSION_SERVER_STARTUP_TIMEOUT_SECS` | `600` | readiness budget for the session servers |
| `AGENT_MAX_CONCURRENT` | `OVER_SAMPLING_BATCH_SIZE x N_SAMPLES_PER_PROMPT` | trials the agent server runs at once |
| `HARBOR_AGENT_MAX_ITERATIONS` | `75` | agent turns per trial |
| `HARBOR_MAX_SEQ_LEN` | `49152` | agent context bound; Terminus-2 summarizes when fewer than 8000 tokens are left |
| `HARBOR_TERMINUS_PARSER` | `xml` | Terminus-2 action format |
| `HARBOR_INTERLEAVED_THINKING` | `true` | keep earlier turns' reasoning in the context |
| `HARBOR_TERMINUS_ENABLE_SUMMARIZE` | `true` | summarize and continue when the context fills |
| `HARBOR_RESPONSE_LENGTH_POLICY` | `regenerate` | what Terminus-2 does when a turn hits `MAX_RESPONSE_LEN` (Harbor's default) |
| `HARBOR_AGENT_TIMEOUT_SEC` | `5400` | agent budget per trial; the trial is still verified when it runs out |
| `HARBOR_AGENT_CALL_TIMEOUT_SEC` | `10800` | client deadline per trial; the sample is dropped when it fires |
| `SANDBOX_CPUS`, `SANDBOX_MEMORY_MB`, `SANDBOX_STORAGE_MB` | `1`, `2048`, `8192` | sandbox size, applied when `SANDBOX_BACKEND=gke` (`task` keeps each task's own value) |
| `GKE_COMPOSE_PREBUILT_MANIFEST_PATH` | unset | `SANDBOX_BACKEND=gke` only: prebuilt Compose image manifest, passed through to the GKE environment (see GKE above) |
| `EXTRA_ARGS` | empty | extra Miles arguments, appended as-is |

Fixed in `train.py`: GRPO with clip 0.2 / 0.28 and no KL or entropy term; rollout
routing replay; TileLang sparse attention for training; engine flags
(`nsa` attention, FP8 KV cache, 64 running requests, `triton` MoE and LoRA
backends, `glm47` tool parser, `glm45` reasoning parser); disk offload of the
training state; CPU optimizer offload.

## Notes

- **Timeouts.** `HARBOR_AGENT_TIMEOUT_SEC` gives every trial the same agent budget,
  whatever the task declares; when it runs out Harbor still runs the verifier, so
  the sample trains with a real reward. `HARBOR_AGENT_CALL_TIMEOUT_SEC` is the
  client's deadline for the whole trial (queue, sandbox start, agent,
  verification); when it fires the sample is lost, so keep it well above the agent budget.
- **Harbor features the launcher checks for.** A trial whose agent runs out of
  output length must still be verified (`OutputLengthExceededError` in
  `trial/single_step.py`); otherwise its sample is aborted and the whole group is
  discarded. The uniform agent budget needs `agent_uniform_timeout_sec` in
  `trial/trial.py`. Both are in `HARBOR_COMMIT`.
- **Context budget.** Summarization starts when fewer than 8000 tokens of
  `HARBOR_MAX_SEQ_LEN` are left; that point must stay below
  `SGLANG_CONTEXT_LENGTH - MAX_RESPONSE_LEN`, or the engine rejects long requests
  before summarization can fire. `train.py` checks this.
- **Concurrency.** All candidate trials run at once (6 groups x 8 = 48). Each
  engine serves up to 64 concurrent requests, so every trial decodes at full
  speed; admitting more trials than the engines hold slows all of them and pushes
  long ones past their deadlines.
- **Pause mode.** Weight updates pause generation with `--pause-generation-mode abort`:
  it is the only mode that drains in-flight requests before the engines flush
  their cache, and with session server v2 it keeps rollout routing replay
  incremental. The requests it aborts belong to surplus oversampled groups.
- **SGLang LoRA settings.** `SGLANG_LORA_FASTPATH=1` sets `SGLANG_EXPERIMENTAL_LORA_OPTI=1`
  and `SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC=1` on the engines (the second is required
  with CUDA graphs once the first is on). The engines use the `triton` LoRA backend;
  SGLang's default LoRA backend crashes this model's rollout.
- **Head-node memory.** The session servers are subprocesses of a Ray actor and
  inherit Ray's environment. With glibc's default arena count, 64 of them keep
  freed heap until the node runs out of memory, hence `MALLOC_ARENA_MAX=2` at
  `ray start`. Default local NUMA allocation can also fill one NUMA node while
  another stays free, hence `numactl --interleave=all`.
- **Offload disk.** `OFFLOAD_DIR` must be fast node-local disk. A tmpfs (as `/tmp`
  is on many systems) keeps the offloaded state in RAM; the launcher warns if the
  head node's `OFFLOAD_DIR` is tmpfs.

## Known issues

- **NaN LoRA gradients from the sparse-MLA backward kernel.** On
  `radixark/Megatron-Bridge@7f0fb345`, TileLang's aggressive shared-memory merge
  aliases buffers that are still live in the GLM-5 sparse-MLA backward kernel, so
  finite inputs produce NaN dQ/dKV. It shows up as `grad_norm=nan` and a skipped
  update on every step with a non-zero advantage. Use the Megatron-Bridge commit
  above or apply `megatron_bridge_sparse_mla_bwd.patch`, and keep its `src` first on
  `PYTHONPATH` (the launcher does this).
- **Session-server startup timeout.** Many session servers starting together can
  miss Miles' built-in readiness deadline, and the run fails before the first
  rollout. The launcher passes `--session-server-startup-timeout-secs 600`; if it
  still fails, raise `SESSION_SERVER_STARTUP_TIMEOUT_SECS` or lower `SESSION_SERVERS`.
- **Samples-collection deadline.** Miles collects each finished session's samples
  with a fixed 120 s HTTP deadline and no retry. A slow collection marks the sample
  aborted, and its whole group is discarded. The spare oversampled groups absorb
  occasional losses; keeping head-node memory pressure low (see Notes) keeps them rare.
