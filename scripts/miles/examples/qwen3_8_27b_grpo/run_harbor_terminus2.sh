#!/usr/bin/env bash
# Train Qwen3.8-27B with GRPO (full fine-tuning) on terminal tasks, with a
# 131072-token context. Rollouts run fully asynchronously through the Harbor
# agent server with the Terminus-2 agent.
#
#   PROMPT_DATA=... TASKS_DIR=... HEAD_IP=... bash scripts/miles/examples/qwen3_8_27b_grpo/run_harbor_terminus2.sh
#
# Runs inside the Miles container on the Ray head node, with Ray already started
# on every node (with MALLOC_ARENA_MAX=2, see README.md). Settings come from .env
# in this folder (see env.example) or the environment. DRY_RUN=1 prints the
# resolved training command and exits. Every knob is an environment variable;
# the defaults are the configuration this script was run with. See README.md.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOG_TAG=qwen3-8-27b-grpo
# shellcheck source=../../common/launcher.sh
source "${SCRIPT_DIR}/../../common/launcher.sh"
load_env_file "${SCRIPT_DIR}"   # ${SCRIPT_DIR}/.env (copy of env.example) or $ENV_FILE

# Tested with
DOCKER_IMAGE=radixark/miles@sha256:616ac8b9fd1e51fabcafefa47faddb67fdac3148f9d53a631dcc9606c88300cd
# Michaelsqj/miles: upstream plus allocator-cache release after each training
# phase and --session-server-startup-timeout-secs.
MILES_COMMIT=1c1ff6b4383923e67263bdad0359cb1792fd63fb
SGLANG_COMMIT=cb05a44f35a7c9e27e46d74112cc841ca674ef43   # sglang-miles branch, in the image
# Michaelsqj/harbor: includes the Terminus-2 format-tolerance fixes and the
# sandbox exec-handshake retry.
HARBOR_COMMIT=${HARBOR_COMMIT:-ab9eaefa3c8c24777e5dd5ff2f0c615694573393}

# Inputs.
: "${PROMPT_DATA:?set PROMPT_DATA to a prompt JSONL (format: README.md in scripts/miles/data)}"
: "${TASKS_DIR:?set TASKS_DIR to the Harbor task directories referenced by PROMPT_DATA}"
# All paths must resolve to the same content on every node.
MODEL_ROOT=${MODEL_ROOT:-/root/models}
MODEL_DIR=${MODEL_DIR:-${MODEL_ROOT}/Qwen3.8-27B}
# torch_dist checkpoint the actor starts from, passed as --ref-load (so it is
# also the KL reference). To continue from an earlier run, point it at that
# run's checkpoints/ directory, not at its HF export; see README.md.
INIT_LOAD=${INIT_LOAD:-${MODEL_ROOT}/Qwen3.8-27B_torch_dist}
MILES_DIR=${MILES_DIR:-/root/miles}
SGLANG_DIR=${SGLANG_DIR:-/sgl-workspace/sglang}
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}

# Topology: NUM_NODES - ROLLOUT_NODES training nodes, ROLLOUT_NODES nodes of
# single-GPU SGLang engines. The actor runs at TP4 x PP1 x CP4: context
# parallelism splits each 131072-token sample over 4 GPUs (32768 tokens each),
# which is what fits; PP1 is affordable because GRPO has no critic.
NUM_NODES=${NUM_NODES:-4}
ROLLOUT_NODES=${ROLLOUT_NODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-1}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-4}

# Batch shape. One optimizer step per ROLLOUT_BATCH_SIZE prompts x
# N_SAMPLES_PER_PROMPT trials (= GLOBAL_BATCH_SIZE); the group of
# N_SAMPLES_PER_PROMPT trials is GRPO's baseline. There is no over-sampling:
# the fully-async producer keeps AGENT_MAX_CONCURRENT trials in flight and
# refills a filtered group.
NUM_ROLLOUT=${NUM_ROLLOUT:-3000}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
# Trials in flight: Miles' in-flight cap and the agent server's concurrency.
export AGENT_MAX_CONCURRENT=${AGENT_MAX_CONCURRENT:-128}
# A trial generated under weight version v still trains after the update to
# v+2, so a long step does not throw away the slowest trajectories.
MAX_WEIGHT_STALENESS=${MAX_WEIGHT_STALENESS:-2}
# none | check_no_infra_failure | check_no_infra_failure_and_nonzero_std
# check_no_infra_failure refills groups with a trial that never reached the
# verifier (AgentError, Flushed, Unknown) instead of training on it as reward 0.
# ..._and_nonzero_std also refills groups whose trials all scored the same
# (zero advantage); on a dataset where most groups are uniform, that multiplies
# the rollout work per step.
DYNAMIC_SAMPLING_FILTER=${DYNAMIC_SAMPLING_FILTER:-check_no_infra_failure}

# Optimization.
LR=${LR:-1e-6}
SAVE_INTERVAL=${SAVE_INTERVAL:-30}

# Lengths. MAX_SEQ_LEN bounds one training sample (prompt + all turns),
# MAX_RESPONSE_LEN one model turn. MAX_TOKENS_PER_GPU x CONTEXT_PARALLEL_SIZE
# must hold one full-length sample, so every sample fits one micro-batch.
MAX_SEQ_LEN=${MAX_SEQ_LEN:-131072}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-32768}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-$((MAX_SEQ_LEN / CONTEXT_PARALLEL_SIZE))}
EXTRA_ARGS=${EXTRA_ARGS:-}

# Harbor agent: Terminus-2 with the XML parser, interleaved thinking and
# summarization. The turn cap is effectively off; the agent-phase time budget
# and the context bound (with summarization) end a trajectory instead.
export HARBOR_AGENT_NAME=${HARBOR_AGENT_NAME:-terminus-2}
[[ "${HARBOR_AGENT_NAME}" == terminus-2 ]] || {
  echo "this script trains HARBOR_AGENT_NAME=terminus-2 only (got ${HARBOR_AGENT_NAME})" >&2
  exit 2
}
export HARBOR_AGENT_MAX_ITERATIONS=${HARBOR_AGENT_MAX_ITERATIONS:-10000}
export HARBOR_MAX_SEQ_LEN=${HARBOR_MAX_SEQ_LEN:-${MAX_SEQ_LEN}}
export HARBOR_TERMINUS_PARSER=${HARBOR_TERMINUS_PARSER:-xml}
export HARBOR_INTERLEAVED_THINKING=${HARBOR_INTERLEAVED_THINKING:-true}
export HARBOR_TERMINUS_ENABLE_SUMMARIZE=${HARBOR_TERMINUS_ENABLE_SUMMARIZE:-true}
# Uniform agent-phase budget for every trial (8 h), whatever the task declares.
# It is a cap, not a duration: most trials finish far earlier. When it fires
# Harbor still verifies the trial, so the sample keeps a real reward.
export HARBOR_AGENT_TIMEOUT_SEC=${HARBOR_AGENT_TIMEOUT_SEC:-28800}
# Client-side deadline per trial (queue, sandbox start, agent, verification).
# When it fires the sample is dropped, so keep it well above the agent budget.
export HARBOR_AGENT_CALL_TIMEOUT_SEC=${HARBOR_AGENT_CALL_TIMEOUT_SEC:-36000}
# Sandbox size for SANDBOX_BACKEND=gke (other backends use each task's own
# resources): 1 CPU, 2 GiB memory, 8 GiB ephemeral storage. Many concurrent
# cold starts need a longer pod readiness deadline than the default.
if [[ "${SANDBOX_BACKEND:-daytona}" == gke ]]; then
  export SANDBOX_CPUS=${SANDBOX_CPUS:-1} SANDBOX_MEMORY_MB=${SANDBOX_MEMORY_MB:-2048}
  export SANDBOX_STORAGE_MB=${SANDBOX_STORAGE_MB:-8192}
  export GKE_POD_READY_TIMEOUT_SEC=${GKE_POD_READY_TIMEOUT_SEC:-1200}
fi

# The Ray job does not run in this directory: make the user-supplied paths absolute.
for var in PROMPT_DATA TASKS_DIR MODEL_DIR INIT_LOAD RUNS_ROOT; do
  [[ -z "${!var:-}" ]] || printf -v "${var}" '%s' "$(realpath -m -- "${!var}")"
done
init_run qwen3-8-27b-terminus2-grpo

MILES_EXTRA_ARGS=()
wandb_args MILES_EXTRA_ARGS
[[ -z "${EXTRA_ARGS}" ]] || MILES_EXTRA_ARGS+=("${EXTRA_ARGS}")

TRAIN_ARGS=(
  --run-id "${RUN_NAME}"
  --num-nodes "${NUM_NODES}"
  --rollout-num-nodes "${ROLLOUT_NODES}"
  --num-gpus-per-node "${GPUS_PER_NODE}"
  --prompt-data "${PROMPT_DATA}"
  --hf-checkpoint "${MODEL_DIR}"
  --ref-load "${INIT_LOAD}"
  --output-dir "${RUNS_ROOT}"
  --megatron-path "${MEGATRON_PATH}"
  --pipeline-model-parallel-size "${PIPELINE_PARALLEL_SIZE}"
  --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --max-seq-len "${MAX_SEQ_LEN}"
  --rollout-max-response-len "${MAX_RESPONSE_LEN}"
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --async-max-concurrent-samples "${AGENT_MAX_CONCURRENT}"
  --max-weight-staleness "${MAX_WEIGHT_STALENESS}"
  --dynamic-sampling-filter "${DYNAMIC_SAMPLING_FILTER}"
  --save-interval "${SAVE_INTERVAL}"
  --lr "${LR}"
  --extra-args "${MILES_EXTRA_ARGS[*]}"
)
TRAIN_CMD=(env PYTHONPATH="${MILES_DIR}:${MILES_SCRIPTS_DIR}" python "${SCRIPT_DIR}/train.py" train "${TRAIN_ARGS[@]}")
dry_run_exit "${TRAIN_CMD[@]}"

# Preflight: model, checkouts, the Miles features this script cannot train without.
for path in "${MODEL_DIR}/model.safetensors.index.json" "${INIT_LOAD}/latest_checkpointed_iteration.txt" \
            "${MILES_DIR}" "${MEGATRON_PATH}"; do
  [[ -e "${path}" ]] || { echo "missing ${path} (see README.md, Step 3)" >&2; exit 2; }
done
miles_require() {  # <path under MILES_DIR> <grep -E pattern | ""> <what breaks>
  if [[ -z "$2" ]]; then [[ -e "${MILES_DIR}/$1" ]]; else grep -qsE -- "$2" "${MILES_DIR}/$1"; fi || {
    echo "Miles at ${MILES_DIR} lacks ${2:-$1}: $3 (tested with Michaelsqj/miles@${MILES_COMMIT})" >&2
    exit 2
  }
}
miles_require scripts/models/qwen3.5-27B.py "" 'no model args for the qwen3.5-27B Megatron type'
miles_require miles/utils/arguments.py '--session-server-startup-timeout-secs' 'unknown flag'
miles_require miles/rollout/fully_async_rollout.py "" 'no --fully-async driver'
miles_require miles/rollout/submission_scheduler.py 'class SampleBackfillSubmission' \
  'in-flight trials would not track AGENT_MAX_CONCURRENT'
check_git_pin miles "${MILES_DIR}" "${MILES_COMMIT}"
check_git_pin sglang "${SGLANG_DIR}" "${SGLANG_COMMIT}"
harbor_check_data
# The Miles session servers are children of Ray workers and inherit the
# raylet's environment. With glibc's default arena count their freed heap is
# never returned to the OS, and at this context length the head node can run
# out of host memory mid-run. Start Ray with MALLOC_ARENA_MAX=2 on every node.
raylet_pid=$(pgrep -o -f 'raylet/raylet' 2>/dev/null || true)
if [[ -n "${raylet_pid}" && -r "/proc/${raylet_pid}/environ" ]] &&
   ! tr '\0' '\n' < "/proc/${raylet_pid}/environ" | grep -q '^MALLOC_ARENA_MAX='; then
  log "WARNING: the local raylet was started without MALLOC_ARENA_MAX; restart Ray with MALLOC_ARENA_MAX=2 (README.md)"
fi

configure_ray
harbor_install
# A trial whose agent exhausts its output length must still be verified: without
# this the sample is aborted and the filter discards its whole group.
harbor_require_source src/harbor/trial/single_step.py 'OutputLengthExceededError' \
  'length-exhausted trials would skip verification and abort their group'
harbor_require_source src/harbor/trial/trial.py 'agent_uniform_timeout_sec' \
  'HARBOR_AGENT_TIMEOUT_SEC would be silently ignored'
harbor_configure
harbor_start_server
write_manifest "$0" "${SCRIPT_DIR}/train.py"

"${TRAIN_CMD[@]}" 2>&1 | tee "${RUN_ROOT}/logs/train.log"
