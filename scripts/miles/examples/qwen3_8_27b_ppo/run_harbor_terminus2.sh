#!/usr/bin/env bash
# Train Qwen3.8-27B with PPO (actor + critic, full fine-tuning) on terminal tasks.
# Rollouts run fully asynchronously through the Harbor agent server with the
# Terminus-2 agent.
#
#   PROMPT_DATA=... TASKS_DIR=... HEAD_IP=... bash scripts/miles/examples/qwen3_8_27b_ppo/run_harbor_terminus2.sh
#
# Runs inside the Miles container on the Ray head node, with Ray already started
# on every node. Settings come from .env in this folder (see env.example) or the
# environment. DRY_RUN=1 prints the resolved training command and exits.
# Every knob is an environment variable; the defaults are the configuration
# this script was run with. See README.md.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOG_TAG=qwen3-8-27b-ppo
# shellcheck source=../../common/launcher.sh
source "${SCRIPT_DIR}/../../common/launcher.sh"
load_env_file "${SCRIPT_DIR}"   # ${SCRIPT_DIR}/.env (copy of env.example) or $ENV_FILE

# Tested with
DOCKER_IMAGE=radixark/miles@sha256:616ac8b9fd1e51fabcafefa47faddb67fdac3148f9d53a631dcc9606c88300cd
# Michaelsqj/miles: upstream plus the pipeline-parallel last-stage guard for
# --use-rollout-logprobs, allocator-cache release after each training phase,
# and --session-server-startup-timeout-secs.
MILES_COMMIT=1c1ff6b4383923e67263bdad0359cb1792fd63fb
SGLANG_COMMIT=cb05a44f35a7c9e27e46d74112cc841ca674ef43   # sglang-miles branch, in the image
HARBOR_COMMIT=${HARBOR_COMMIT:-5af13d825dd8b4b8ae330133ce25ecb3019653ba}   # Michaelsqj/harbor

# Inputs.
: "${PROMPT_DATA:?set PROMPT_DATA to a prompt JSONL (format: README.md in scripts/miles/data)}"
: "${TASKS_DIR:?set TASKS_DIR to the Harbor task directories referenced by PROMPT_DATA}"
# All paths must resolve to the same content on every node.
MODEL_ROOT=${MODEL_ROOT:-/root/models}
MODEL_DIR=${MODEL_DIR:-${MODEL_ROOT}/Qwen3.8-27B}
# torch_dist checkpoint the actor and the critic start from (prepare_model.sh).
INIT_LOAD=${INIT_LOAD:-${MODEL_ROOT}/Qwen3.8-27B_torch_dist}
MILES_DIR=${MILES_DIR:-/root/miles}
SGLANG_DIR=${SGLANG_DIR:-/sgl-workspace/sglang}
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}

# Topology: NUM_NODES - ROLLOUT_NODES training nodes (actor and critic share
# them), ROLLOUT_NODES nodes of single-GPU SGLang engines. The actor runs at
# TP4 x PP2 x CP1, so 2 training nodes give DP2. PP2 halves the resident
# weights and gradients per GPU so the co-resident critic fits.
NUM_NODES=${NUM_NODES:-4}
ROLLOUT_NODES=${ROLLOUT_NODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-2}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}

# Batch shape. One optimizer step per ROLLOUT_BATCH_SIZE prompts x
# N_SAMPLES_PER_PROMPT trials (= GLOBAL_BATCH_SIZE). There is no over-sampling:
# the fully-async producer keeps AGENT_MAX_CONCURRENT trials in flight and
# refills a filtered group.
NUM_ROLLOUT=${NUM_ROLLOUT:-3000}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-64}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-2}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
# Trials in flight: Miles' in-flight cap and the agent server's concurrency.
export AGENT_MAX_CONCURRENT=${AGENT_MAX_CONCURRENT:-128}
# A trial generated under weight version v still trains after the update to
# v+2, so a long step does not throw away the slowest trajectories.
MAX_WEIGHT_STALENESS=${MAX_WEIGHT_STALENESS:-2}
# none | check_no_infra_failure | check_no_infra_failure_and_nonzero_std
# check_no_infra_failure refills groups with a trial that never reached the
# verifier (AgentError, Flushed, Unknown) instead of training on it as reward 0.
DYNAMIC_SAMPLING_FILTER=${DYNAMIC_SAMPLING_FILTER:-check_no_infra_failure}

# Optimization.
LR=${LR:-1e-6}
CRITIC_LR=${CRITIC_LR:-1e-5}
SAVE_INTERVAL=${SAVE_INTERVAL:-30}

# Lengths. MAX_SEQ_LEN bounds one training sample (prompt + all turns),
# MAX_RESPONSE_LEN one model turn. MAX_TOKENS_PER_GPU x CONTEXT_PARALLEL_SIZE
# must hold one full-length sample, so every sample fits one micro-batch.
MAX_SEQ_LEN=${MAX_SEQ_LEN:-49152}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-16384}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-$((MAX_SEQ_LEN / CONTEXT_PARALLEL_SIZE))}
EXTRA_ARGS=${EXTRA_ARGS:-}

# Harbor agent: Terminus-2 with the XML parser, interleaved thinking and
# summarization, bounded to the training context.
export HARBOR_AGENT_NAME=${HARBOR_AGENT_NAME:-terminus-2}
[[ "${HARBOR_AGENT_NAME}" == terminus-2 ]] || {
  echo "this script trains HARBOR_AGENT_NAME=terminus-2 only (got ${HARBOR_AGENT_NAME})" >&2
  exit 2
}
export HARBOR_AGENT_MAX_ITERATIONS=${HARBOR_AGENT_MAX_ITERATIONS:-75}
export HARBOR_MAX_SEQ_LEN=${HARBOR_MAX_SEQ_LEN:-${MAX_SEQ_LEN}}
export HARBOR_TERMINUS_PARSER=${HARBOR_TERMINUS_PARSER:-xml}
export HARBOR_INTERLEAVED_THINKING=${HARBOR_INTERLEAVED_THINKING:-true}
export HARBOR_TERMINUS_ENABLE_SUMMARIZE=${HARBOR_TERMINUS_ENABLE_SUMMARIZE:-true}
# Uniform agent-phase budget for every trial, whatever the task declares. When
# it fires Harbor still verifies the trial, so the sample keeps a real reward.
export HARBOR_AGENT_TIMEOUT_SEC=${HARBOR_AGENT_TIMEOUT_SEC:-5400}
# Client-side deadline per trial (queue, sandbox start, agent, verification).
# When it fires the sample is dropped, so keep it well above the agent budget.
export HARBOR_AGENT_CALL_TIMEOUT_SEC=${HARBOR_AGENT_CALL_TIMEOUT_SEC:-28800}
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
init_run qwen3-8-27b-terminus2-ppo

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
  --critic-lr "${CRITIC_LR}"
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
miles_require miles/backends/training_utils/loss.py 'if not get_parallel_state\(\)\.is_pp_last_stage:' \
  'PP > 1 with --use-rollout-logprobs crashes with values=None'
miles_require miles/backends/megatron_utils/actor.py "Release this phase.s cached allocator blocks" \
  'the idle critic keeps its allocator cache and the actor phase runs out of memory'
check_git_pin miles "${MILES_DIR}" "${MILES_COMMIT}"
check_git_pin sglang "${SGLANG_DIR}" "${SGLANG_COMMIT}"
harbor_check_data

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
