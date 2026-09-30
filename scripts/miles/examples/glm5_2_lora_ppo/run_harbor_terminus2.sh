#!/usr/bin/env bash
# Train GLM-5.2 (744B-A40B) with LoRA PPO (actor + critic) on terminal tasks.
# Rollouts run through the Harbor agent server with the Terminus-2 agent.
#
#   cp env.example .env   # fill in, then:
#   bash scripts/miles/examples/glm5_2_lora_ppo/run_harbor_terminus2.sh
#
# Runs inside the Miles container on the Ray head node, with Ray already started
# on every node. DRY_RUN=1 prints the resolved training command and exits.
# SMOKE=1 shrinks the run to 3 short steps to check the stack end to end.
# Every setting below is an environment variable (or a line in .env) whose
# default is the configuration this recipe was run with; see README.md.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOG_TAG=glm5_2_lora_ppo
source "${SCRIPT_DIR}/../../common/launcher.sh"
load_env_file "${SCRIPT_DIR}"

# Tested with
DOCKER_IMAGE=radixark/miles@sha256:616ac8b9fd1e51fabcafefa47faddb67fdac3148f9d53a631dcc9606c88300cd
MILES_COMMIT=1c1ff6b4383923e67263bdad0359cb1792fd63fb            # Michaelsqj/miles: LoRA critic under the bridge, session-server startup timeout, incremental R3 under abort, drain-aware flush_cache
SGLANG_COMMIT=cb05a44f35a7c9e27e46d74112cc841ca674ef43           # sgl-project/sglang sglang-miles, as shipped in DOCKER_IMAGE
MEGATRON_BRIDGE_COMMIT=e91c492328700f9cd59b8d1a82d40a5e54fbd835  # Michaelsqj/Megatron-Bridge: GLM-5 sparse-MLA backward fix
HARBOR_COMMIT=${HARBOR_COMMIT:-5af13d825dd8b4b8ae330133ce25ecb3019653ba}  # Michaelsqj/harbor

# Inputs. Paths must resolve to the same content on every node.
: "${PROMPT_DATA:?set PROMPT_DATA to a prompt JSONL (see scripts/miles/data/README.md)}"
: "${TASKS_DIR:?set TASKS_DIR to the Harbor task directories referenced by PROMPT_DATA}"
MODEL_ROOT=${MODEL_ROOT:-/root/models}
MODEL_DIR=${MODEL_DIR:-${MODEL_ROOT}/GLM-5.2}               # BF16: the actor's and the critic's base
FP8_MODEL_DIR=${FP8_MODEL_DIR:-${MODEL_ROOT}/GLM-5.2-FP8}   # FP8: what the rollout engines serve
MILES_DIR=${MILES_DIR:-/root/miles}
SGLANG_DIR=${SGLANG_DIR:-/sgl-workspace/sglang}
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}
MEGATRON_BRIDGE_DIR=${MEGATRON_BRIDGE_DIR:-/root/Megatron-Bridge-glm5_2}
# Where each node parks the actor's and the critic's training state while the
# rollout engines hold the GPUs: fast node-local disk, never tmpfs (that would
# keep it in RAM) or a network filesystem. Each model needs about 450-600 GB
# per node, so budget about 1.2 TB free.
OFFLOAD_DIR=${OFFLOAD_DIR:-/root/miles_train_offload}
OFFLOAD_MIN_FREE_GB=1200

# SMOKE=1: 3 steps (the first trains only the critic), one accepted group of 8
# trials, 10 agent turns per trial. Checks the stack end to end in well under
# the time of a full run; settings you set explicitly still win.
RUN_PREFIX=glm5_2-lora-ppo-terminus2
if [[ "${SMOKE:-0}" == 1 ]]; then
  RUN_PREFIX=${RUN_PREFIX}-smoke
  NUM_ROLLOUT=${NUM_ROLLOUT:-3}
  OVER_SAMPLING_BATCH_SIZE=${OVER_SAMPLING_BATCH_SIZE:-2}
  ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-1}
  N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
  HARBOR_AGENT_MAX_ITERATIONS=${HARBOR_AGENT_MAX_ITERATIONS:-10}
  AGENT_MAX_CONCURRENT=${AGENT_MAX_CONCURRENT:-8}
fi

# Topology. The actor, the critic and the rollout engines share every GPU
# (colocated): 32 GPUs, expert parallel 32 for both training models, two
# 16-GPU FP8 rollout engines.
NUM_NODES=${NUM_NODES:-4}

# Batch shape. Each rollout starts OVER_SAMPLING_BATCH_SIZE prompt groups of
# N_SAMPLES_PER_PROMPT trials and keeps the first ROLLOUT_BATCH_SIZE complete
# groups with no aborted sample; the spare groups absorb trials lost to
# infrastructure. PPO's baseline is the critic, not the group, so groups are 2
# trials and a step covers 16 distinct prompts.
NUM_ROLLOUT=${NUM_ROLLOUT:-200}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-2}
OVER_SAMPLING_BATCH_SIZE=${OVER_SAMPLING_BATCH_SIZE:-24}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.8}

# Optimization. The adapters (actor and critic) cover the attention and MLA
# projections only; the critic also trains a fresh scalar value head. The
# first NUM_CRITIC_ONLY_STEPS rollouts train the critic alone.
LR=${LR:-3e-5}
CRITIC_LR=${CRITIC_LR:-3e-5}
NUM_CRITIC_ONLY_STEPS=${NUM_CRITIC_ONLY_STEPS:-1}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
SAVE_INTERVAL=${SAVE_INTERVAL:-10}

# Lengths. MAX_SEQ_LEN bounds one training sample and MAX_RESPONSE_LEN one model
# turn. The engine window is large on purpose: the agent's own bound
# (HARBOR_MAX_SEQ_LEN) plus summarization keep real contexts well inside it.
MAX_SEQ_LEN=${MAX_SEQ_LEN:-49152}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-${MAX_SEQ_LEN}}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-16384}
SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH:-1048576}
# Each parked training model keeps about 4 GiB per GPU. With two of them, 0.85
# leaves too little for the engines' sparse-MLA prefill; 0.80 holds.
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.80}
# 1: SGLang's experimental LoRA forward path on the rollout engines; 0: default path.
SGLANG_LORA_FASTPATH=${SGLANG_LORA_FASTPATH:-0}

# Miles session servers, one process per port in [SESSION_SERVER_PORT,
# SESSION_SERVER_PORT + SESSION_SERVERS); the agent's model calls go through
# them, so training sees exactly the tokens the engines produced. Each process
# costs head-node memory, which PPO needs for its second training model, so
# this recipe runs 16 for its 48 concurrent trials. They all start at once and
# can exceed Miles' built-in readiness budget; hence the longer startup timeout.
SESSION_SERVERS=${SESSION_SERVERS:-16}
SESSION_SERVER_PORT=${SESSION_SERVER_PORT:-30000}
SESSION_SERVER_STARTUP_TIMEOUT_SECS=${SESSION_SERVER_STARTUP_TIMEOUT_SECS:-600}
EXTRA_ARGS=${EXTRA_ARGS:-}

# Harbor agent: Terminus-2 with the XML action parser, interleaved thinking
# (earlier turns keep their reasoning) and summarization when the context fills.
export HARBOR_AGENT_NAME=${HARBOR_AGENT_NAME:-terminus-2}
[[ "${HARBOR_AGENT_NAME}" == terminus-2 ]] || {
  echo "this script trains HARBOR_AGENT_NAME=terminus-2 only (got ${HARBOR_AGENT_NAME})" >&2
  exit 2
}
export HARBOR_AGENT_MAX_ITERATIONS=${HARBOR_AGENT_MAX_ITERATIONS:-75}
export HARBOR_MAX_SEQ_LEN=${HARBOR_MAX_SEQ_LEN:-49152}
export HARBOR_TERMINUS_PARSER=${HARBOR_TERMINUS_PARSER:-xml}
export HARBOR_INTERLEAVED_THINKING=${HARBOR_INTERLEAVED_THINKING:-true}
export HARBOR_TERMINUS_ENABLE_SUMMARIZE=${HARBOR_TERMINUS_ENABLE_SUMMARIZE:-true}
# What Terminus-2 does when one turn hits MAX_RESPONSE_LEN (Harbor's default).
export HARBOR_RESPONSE_LENGTH_POLICY=${HARBOR_RESPONSE_LENGTH_POLICY:-regenerate}
# One agent-phase budget for every trial, whatever the task declares. When it
# fires, Harbor still verifies the trial, so the sample keeps a real reward.
export HARBOR_AGENT_TIMEOUT_SEC=${HARBOR_AGENT_TIMEOUT_SEC:-5400}
# Client-side deadline for a whole trial (queue, sandbox start, agent,
# verification). When it fires the sample is dropped, so keep it well above
# HARBOR_AGENT_TIMEOUT_SEC.
export HARBOR_AGENT_CALL_TIMEOUT_SEC=${HARBOR_AGENT_CALL_TIMEOUT_SEC:-10800}
# Run every candidate trial at once (24 groups x 2 = 48). Each engine runs up
# to 64 requests, so every admitted trial decodes at full speed; admitting more
# trials than the engines hold slows all of them and pushes long trials past
# their deadlines.
export AGENT_MAX_CONCURRENT=${AGENT_MAX_CONCURRENT:-$((OVER_SAMPLING_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
# Sandbox size for SANDBOX_BACKEND=gke; the other backends use each task's own
# resources. Many concurrent cold starts need a longer pod readiness deadline
# than Harbor's default.
if [[ "${SANDBOX_BACKEND:-daytona}" == gke ]]; then
  export SANDBOX_CPUS=${SANDBOX_CPUS:-1} SANDBOX_MEMORY_MB=${SANDBOX_MEMORY_MB:-2048}
  export SANDBOX_STORAGE_MB=${SANDBOX_STORAGE_MB:-8192}
  export GKE_POD_READY_TIMEOUT_SEC=${GKE_POD_READY_TIMEOUT_SEC:-1200}
fi

init_run "${RUN_PREFIX}"

MILES_EXTRA_ARGS=(--dump-details "${RUN_ROOT}/dump_details" --use-miles-dashboard)
wandb_args MILES_EXTRA_ARGS
[[ -z "${EXTRA_ARGS}" ]] || MILES_EXTRA_ARGS+=("${EXTRA_ARGS}")
if [[ "${SGLANG_LORA_FASTPATH}" == 1 ]]; then FASTPATH_FLAG=--sglang-lora-fastpath; else FASTPATH_FLAG=--no-sglang-lora-fastpath; fi

TRAIN_ARGS=(
  --run-id "${RUN_NAME}"
  --num-nodes "${NUM_NODES}"
  --num-gpus-per-node 8
  --prompt-data "${PROMPT_DATA}"
  --hf-checkpoint "${MODEL_DIR}"
  --fp8-rollout-checkpoint "${FP8_MODEL_DIR}"
  --output-dir "${RUNS_ROOT}"
  --megatron-path "${MEGATRON_PATH}"
  --megatron-bridge-path "${MEGATRON_BRIDGE_DIR}"
  --offload-train-disk-dir "${OFFLOAD_DIR}"
  --max-seq-len "${MAX_SEQ_LEN}"
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --rollout-max-response-len "${MAX_RESPONSE_LEN}"
  --sglang-context-length "${SGLANG_CONTEXT_LENGTH}"
  --num-rollout "${NUM_ROLLOUT}"
  --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE}"
  --save-interval "${SAVE_INTERVAL}"
  --lr "${LR}"
  --critic-lr "${CRITIC_LR}"
  --num-critic-only-steps "${NUM_CRITIC_ONLY_STEPS}"
  --lora-rank "${LORA_RANK}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout 0.0
  --dsa-attention-backend tilelang
  --fp8-rollout-gpus-per-engine 16
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}"
  "${FASTPATH_FLAG}"
  --session-server-port-start "${SESSION_SERVER_PORT}"
  --session-server-port-end "$((SESSION_SERVER_PORT + SESSION_SERVERS))"
  --session-server-startup-timeout-secs "${SESSION_SERVER_STARTUP_TIMEOUT_SECS}"
  --extra-args "${MILES_EXTRA_ARGS[*]}"
)
# Megatron-Bridge first on PYTHONPATH: its sparse-MLA backward fix must shadow
# the copy installed in the image.
TRAIN_CMD=(env PYTHONPATH="${MEGATRON_BRIDGE_DIR}/src:${MILES_DIR}:${MILES_SCRIPTS_DIR}"
           python "${SCRIPT_DIR}/train.py" train "${TRAIN_ARGS[@]}")
dry_run_exit "${TRAIN_CMD[@]}"

# Preflight: models, checkouts and the fixes this recipe cannot train without.
for path in "${MODEL_DIR}/model.safetensors.index.json" \
            "${FP8_MODEL_DIR}/model.safetensors.index.json" \
            "${MILES_DIR}" "${MEGATRON_PATH}"; do
  [[ -e "${path}" ]] || { echo "missing ${path} (see README.md, Steps 1-3)" >&2; exit 2; }
done
# A checkout of your own with megatron_bridge_sparse_mla_bwd.patch applied passes too.
grep -qs 'TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: False' \
  "${MEGATRON_BRIDGE_DIR}/src/megatron/bridge/models/glm5/tilelang/tilelang_sparse_mla_bwd.py" || {
  echo "${MEGATRON_BRIDGE_DIR} lacks the GLM-5 sparse-MLA backward fix, so LoRA gradients would be NaN." >&2
  echo "Run ${SCRIPT_DIR}/setup_megatron_bridge.sh or apply ${SCRIPT_DIR}/megatron_bridge_sparse_mla_bwd.patch." >&2
  exit 2
}
# Each Miles feature below would otherwise fail (or silently misbehave) only
# after the Ray job has started.
miles_require() {  # <path under MILES_DIR> <grep -E pattern> <what breaks without it>
  grep -qsE -- "$2" "${MILES_DIR}/$1" || {
    echo "Miles at ${MILES_DIR} lacks '$2' in $1: $3." >&2
    echo "Check out Michaelsqj/miles@${MILES_COMMIT} (README.md, Step 1)." >&2
    exit 2
  }
}
if grep -qs 'Critic models are not supported with' "${MILES_DIR}/miles/utils/arguments.py"; then
  echo "Miles at ${MILES_DIR} rejects a critic under --megatron-to-hf-mode bridge;" \
       "check out Michaelsqj/miles@${MILES_COMMIT} (README.md, Step 1)." >&2
  exit 2
fi
miles_require miles/backends/megatron_utils/checkpoint.py '_hide_critic_value_head_from_hf_load' \
  "the critic's value head would be matched against the model's LM head when the base weights load"
miles_require miles/backends/megatron_utils/model.py 'role in \("actor", "critic"\) and args\.megatron_to_hf_mode == "bridge"' \
  'the critic would be built without LoRA, as a full-parameter 744B model'
miles_require miles/ray/train/actor_factory.py 'role_tag' \
  "the actor's and the critic's disk offload would share one directory per rank"
miles_require miles/utils/arguments.py '--session-server-startup-timeout-secs' \
  'unknown flag --session-server-startup-timeout-secs'
miles_require miles/rollout/session/server.py 'pause_generation_mode != "retract"' \
  'rollout routing replay would resend its full payload on every turn under --pause-generation-mode abort'
miles_require miles/backends/sglang_utils/sglang_engine.py 'drain before flush' \
  "a weight update's flush_cache would fail after 60 s while long requests drain"
# Newer Miles imports SGLang modules the image's SGLang does not have; the job
# would die at submission.
if grep -rqs 'sglang\.srt\.entrypoints\.anthropic' "${MILES_DIR}/miles" &&
   [[ ! -e "${SGLANG_DIR}/python/sglang/srt/entrypoints/anthropic" ]]; then
  echo "Miles at ${MILES_DIR} needs a newer SGLang than ${SGLANG_DIR}; check out Michaelsqj/miles@${MILES_COMMIT}." >&2
  exit 2
fi
mkdir -p "${OFFLOAD_DIR}"
[[ "$(df --output=fstype "${OFFLOAD_DIR}" 2>/dev/null | tail -n 1)" != tmpfs ]] ||
  log "WARNING: OFFLOAD_DIR=${OFFLOAD_DIR} is tmpfs; the offloaded training state would stay in RAM"
offload_free_gb=$(( $(df -B1 --output=avail "${OFFLOAD_DIR}" | tail -n 1) / 1000000000 ))
((offload_free_gb >= OFFLOAD_MIN_FREE_GB)) ||
  log "WARNING: OFFLOAD_DIR=${OFFLOAD_DIR} has ${offload_free_gb} GB free on this node; the actor and critic need about ${OFFLOAD_MIN_FREE_GB} GB per node"
check_git_pin miles "${MILES_DIR}" "${MILES_COMMIT}"
check_git_pin sglang "${SGLANG_DIR}" "${SGLANG_COMMIT}"
check_git_pin megatron-bridge "${MEGATRON_BRIDGE_DIR}" "${MEGATRON_BRIDGE_COMMIT}"
harbor_check_data

configure_ray
harbor_install
# A trial whose agent runs out of output length must still be verified;
# otherwise its sample is aborted and check_no_aborted drops the whole group.
harbor_require_source src/harbor/trial/single_step.py 'OutputLengthExceededError' \
  'length-exhausted trials would skip verification and abort their group'
harbor_require_source src/harbor/trial/trial.py 'agent_uniform_timeout_sec' \
  'HARBOR_AGENT_TIMEOUT_SEC would be silently ignored'
harbor_configure
harbor_start_server
write_manifest "$0" "${SCRIPT_DIR}/train.py"

"${TRAIN_CMD[@]}" 2>&1 | tee "${RUN_ROOT}/logs/train.log"
