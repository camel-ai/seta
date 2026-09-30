#!/usr/bin/env bash
# Train GLM-5.2 (744B-A40B) with LoRA GRPO on terminal tasks. Rollouts run through
# the Harbor agent server with the Terminus-2 agent.
#
#   cp env.example .env   # fill in, then:
#   bash scripts/miles/examples/glm5_2_lora_grpo/run_harbor_terminus2.sh
#
# Runs inside the Miles container on the Ray head node, with Ray already started
# on every node. DRY_RUN=1 prints the resolved training command and exits.
# Every setting below is an environment variable (or a line in .env) whose
# default is the configuration this recipe was run with; see README.md.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOG_TAG=glm5_2_lora_grpo
source "${SCRIPT_DIR}/../../common/launcher.sh"
load_env_file "${SCRIPT_DIR}"

# Tested with
DOCKER_IMAGE=radixark/miles@sha256:616ac8b9fd1e51fabcafefa47faddb67fdac3148f9d53a631dcc9606c88300cd
MILES_COMMIT=5479631b81d0747607cf424439299b766fdfe79d            # Michaelsqj/miles: session-server startup timeout, incremental R3 under abort, drain-aware flush_cache
SGLANG_COMMIT=cb05a44f35a7c9e27e46d74112cc841ca674ef43           # sgl-project/sglang sglang-miles, as shipped in DOCKER_IMAGE
MEGATRON_BRIDGE_COMMIT=e91c492328700f9cd59b8d1a82d40a5e54fbd835  # Michaelsqj/Megatron-Bridge: GLM-5 sparse-MLA backward fix
HARBOR_COMMIT=${HARBOR_COMMIT:-5af13d825dd8b4b8ae330133ce25ecb3019653ba}  # Michaelsqj/harbor

# Inputs. Paths must resolve to the same content on every node.
: "${PROMPT_DATA:?set PROMPT_DATA to a prompt JSONL (see scripts/miles/data/README.md)}"
: "${TASKS_DIR:?set TASKS_DIR to the Harbor task directories referenced by PROMPT_DATA}"
MODEL_ROOT=${MODEL_ROOT:-/root/models}
MODEL_DIR=${MODEL_DIR:-${MODEL_ROOT}/GLM-5.2}               # BF16: the training model
FP8_MODEL_DIR=${FP8_MODEL_DIR:-${MODEL_ROOT}/GLM-5.2-FP8}   # FP8: what the rollout engines serve
MILES_DIR=${MILES_DIR:-/root/miles}
SGLANG_DIR=${SGLANG_DIR:-/sgl-workspace/sglang}
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}
MEGATRON_BRIDGE_DIR=${MEGATRON_BRIDGE_DIR:-/root/Megatron-Bridge-glm5_2}
# Where each node parks the training state while the rollout engines hold the
# GPUs: fast node-local disk with about 500 GB free, never tmpfs (that would
# keep it in RAM) or a network filesystem.
OFFLOAD_DIR=${OFFLOAD_DIR:-/root/miles_train_offload}

# Topology. Training and rollout share every GPU (colocated).
#   NUM_NODES=4: 32 GPUs, expert parallel 32, two 16-GPU FP8 rollout engines
#   NUM_NODES=8: 64 GPUs, expert parallel 64, four 16-GPU FP8 rollout engines
NUM_NODES=${NUM_NODES:-4}

# Batch shape. Each rollout starts OVER_SAMPLING_BATCH_SIZE prompt groups of
# N_SAMPLES_PER_PROMPT trials and keeps the first ROLLOUT_BATCH_SIZE complete
# groups with no aborted sample; the spare groups absorb trials lost to
# infrastructure, so one slow sandbox does not stall the step.
NUM_ROLLOUT=${NUM_ROLLOUT:-200}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-4}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
OVER_SAMPLING_BATCH_SIZE=${OVER_SAMPLING_BATCH_SIZE:-6}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.8}

# Optimization. The adapter covers the attention and MLA projections only.
LR=${LR:-3e-5}
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
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.85}
# 1: SGLang's experimental LoRA forward path on the rollout engines; 0: default path.
SGLANG_LORA_FASTPATH=${SGLANG_LORA_FASTPATH:-1}

# Miles session servers, one process per port in [SESSION_SERVER_PORT,
# SESSION_SERVER_PORT + SESSION_SERVERS); the agent's model calls go through
# them, so training sees exactly the tokens the engines produced. They all start
# at once and each imports transformers, which can exceed Miles' built-in
# readiness budget; hence the longer startup timeout.
SESSION_SERVERS=${SESSION_SERVERS:-64}
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
# Run every candidate trial at once (6 groups x 8 = 48). Each engine runs up to
# 64 requests, so every admitted trial decodes at full speed; admitting more
# trials than the engines hold slows all of them and pushes long trials past
# their deadlines.
export AGENT_MAX_CONCURRENT=${AGENT_MAX_CONCURRENT:-$((OVER_SAMPLING_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
# Sandbox size for SANDBOX_BACKEND=gke; the other backends use each task's own
# resources.
if [[ "${SANDBOX_BACKEND:-daytona}" == gke ]]; then
  export SANDBOX_CPUS=${SANDBOX_CPUS:-1} SANDBOX_MEMORY_MB=${SANDBOX_MEMORY_MB:-2048}
  export SANDBOX_STORAGE_MB=${SANDBOX_STORAGE_MB:-8192}
fi

init_run glm5_2-lora-grpo-terminus2

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
  [[ -e "${path}" ]] || { echo "missing ${path} (see README.md, Prerequisites)" >&2; exit 2; }
done
# A checkout of your own with megatron_bridge_sparse_mla_bwd.patch applied passes too.
grep -qs 'TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: False' \
  "${MEGATRON_BRIDGE_DIR}/src/megatron/bridge/models/glm5/tilelang/tilelang_sparse_mla_bwd.py" || {
  echo "${MEGATRON_BRIDGE_DIR} lacks the GLM-5 sparse-MLA backward fix, so LoRA gradients would be NaN." >&2
  echo "Run ${SCRIPT_DIR}/setup_megatron_bridge.sh or apply ${SCRIPT_DIR}/megatron_bridge_sparse_mla_bwd.patch." >&2
  exit 2
}
# Upstream Miles lacks this flag and would reject it only after the Ray job starts.
grep -qs -- '--session-server-startup-timeout-secs' "${MILES_DIR}/miles/utils/arguments.py" || {
  echo "Miles at ${MILES_DIR} lacks --session-server-startup-timeout-secs; check out Michaelsqj/miles@${MILES_COMMIT}" >&2
  exit 2
}
mkdir -p "${OFFLOAD_DIR}"
[[ "$(df --output=fstype "${OFFLOAD_DIR}" 2>/dev/null | tail -n 1)" != tmpfs ]] ||
  log "WARNING: OFFLOAD_DIR=${OFFLOAD_DIR} is tmpfs; the offloaded training state would stay in RAM"
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
