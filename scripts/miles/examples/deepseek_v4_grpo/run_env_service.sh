#!/usr/bin/env bash
# Train DeepSeek-V4-Flash-FP8 (all parameters) with GRPO on terminal tasks. Rollouts run through
# the seta env_service with seta's CAMEL agent in Daytona sandboxes. Miles train_async.py: two
# nodes serve SGLang, the others train, and a continuous rollout worker keeps sampling while the
# trainer steps. Routing replay (R3) is on.
#
#   cp env.example .env   # fill in, then:
#   bash scripts/miles/examples/deepseek_v4_grpo/run_env_service.sh
#
# Runs inside the Miles container on the Ray head node, with Ray started on every node; the
# env_service runs on the head node next to it. DRY_RUN=1 prints the resolved training command
# and exits. Every setting below is an environment variable whose default is the configuration
# this recipe was run with.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOG_TAG=deepseek-v4
source "${SCRIPT_DIR}/../../common/launcher.sh"
source "${SCRIPT_DIR}/../../common/env_service.sh"
load_env_file "${SCRIPT_DIR}"

# Inputs. Paths must resolve to the same content on every node.
: "${PROMPT_DATA:?set PROMPT_DATA to a prompt JSONL (see scripts/miles/data/README.md)}"
: "${TASKS_DIR:?set TASKS_DIR to the task directories referenced by PROMPT_DATA}"
MODEL_ROOT=${MODEL_ROOT:-/root/models}  # DeepSeek-V4-Flash-FP8 and DeepSeek-V4-Flash-FP8_torch_dist (train.py prepare)
MILES_DIR=${MILES_DIR:-/root/miles}
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}

# Topology. ROLLOUT_NUM_NODES nodes run one TP8 SGLang engine each; the other nodes train
# (TP8 / EP8, one pipeline stage per node: 6 stages on 8 nodes).
NUM_NODES=${NUM_NODES:-8}
ROLLOUT_NUM_NODES=${ROLLOUT_NUM_NODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
HARDWARE=${HARDWARE:-auto}  # FP8 scale format: Hopper vs Blackwell (auto-detected)

# Batch shape. Each optimizer step trains on ROLLOUT_BATCH_SIZE prompt groups of
# N_SAMPLES_PER_PROMPT trajectories (8 x 16 = 128). The rollout worker keeps up to
# ROLLOUT_CONCURRENCY groups in flight independently of that, and drops groups generated more
# than MAX_WEIGHT_STALENESS weight updates ago.
NUM_ROLLOUT=${NUM_ROLLOUT:-3000}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}
TEMPERATURE=${TEMPERATURE:-0.8}
export ROLLOUT_CONCURRENCY=${ROLLOUT_CONCURRENCY:-12}
MAX_WEIGHT_STALENESS=${MAX_WEIGHT_STALENESS:-4}

# Optimization: constant LR, Adam state offloaded to the CPU.
LR=${LR:-1e-6}
# Weights-only checkpoints every SAVE_INTERVAL steps. SKIP_SAVING=1 writes none.
SAVE_INTERVAL=${SAVE_INTERVAL:-50}
SKIP_SAVING=${SKIP_SAVING:-0}
# Tokens per model call. The agent's own limits (turns, context, per-call tokens, sampling)
# are in env_service.yaml.
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-8192}
# 1: write per-rollout samples and tensors under RUN_ROOT/dump_details.
DUMP_DETAILS=${DUMP_DETAILS:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}  # extra Miles flags, appended last (they override the ones above)
export WANDB_PROJECT=${WANDB_PROJECT:-deepseek-v4-grpo}

init_run deepseek-v4-grpo-env-service

# env_service: seta_env's service on the head node (common/env_service.sh). It resolves a task
# as DATASET_ROOT/<dataset>/<instance_id> (both derived from TASKS_DIR), runs the CAMEL agent
# against the Miles session URL in a Daytona sandbox and returns the verified reward. The
# agent and sandbox settings are in env_service.yaml; the service settings follow.
export ENV_SERVICE_CONFIG=${ENV_SERVICE_CONFIG:-${SCRIPT_DIR}/env_service.yaml}
export MAX_SLOTS=${MAX_SLOTS:-160}  # concurrent sandboxes; sized with the sandbox caps in env_service.yaml
# One trajectory end to end; must exceed the sum of the task_timeouts in env_service.yaml (4260 s).
export STEP_TIMEOUT_SECONDS=${STEP_TIMEOUT_SECONDS:-5000}
export STEP_USE_SUBPROCESS=${STEP_USE_SUBPROCESS:-1}  # each trajectory in a killable subprocess
export BUILD_CONCURRENCY=${BUILD_CONCURRENCY:-24}
export INFLIGHT_LEAD=${INFLIGHT_LEAD:-64}
export HARBOR_DAYTONA_MAX_CREATES=${HARBOR_DAYTONA_MAX_CREATES:-24}  # concurrent Daytona create calls
export DAYTONA_DECLARATIVE=${DAYTONA_DECLARATIVE:-1}
export DAYTONA_SNAPSHOT_EVICT_AGE_HOURS=${DAYTONA_SNAPSHOT_EVICT_AGE_HOURS:-3}
# The agent's client timeout for one model call. DeepSeek-V4 reasoning turns of up to 8192
# tokens take minutes under load; with the 180 s default most trajectories fail.
export MODEL_TIMEOUT=${MODEL_TIMEOUT:-900}

WANDB_ARGS=()
wandb_args WANDB_ARGS
WANDB_FLAGS=""
if (( ${#WANDB_ARGS[@]} )); then WANDB_FLAGS=$(printf '%q ' "${WANDB_ARGS[@]}"); fi
if [[ "${SKIP_SAVING}" == 1 ]]; then SAVE_FLAG=--skip-saving; else SAVE_FLAG=--no-skip-saving; fi
if [[ "${DUMP_DETAILS}" == 1 ]]; then DUMP_FLAG=--dump-details; else DUMP_FLAG=--no-dump-details; fi

TRAIN_ARGS=(
  --rollout-backend env-service
  --num-nodes "${NUM_NODES}"
  --rollout-num-nodes "${ROLLOUT_NUM_NODES}"
  --num-gpus-per-node "${GPUS_PER_NODE}"
  --hardware "${HARDWARE}"
  --model-root "${MODEL_ROOT}"
  --megatron-path "${MEGATRON_PATH}"
  --prompt-data "${PROMPT_DATA}"
  --run-root "${RUN_ROOT}"
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --temperature "${TEMPERATURE}"
  --max-response-len "${MAX_RESPONSE_LEN}"
  --max-weight-staleness "${MAX_WEIGHT_STALENESS}"
  --lr "${LR}"
  --save-interval "${SAVE_INTERVAL}"
  "${SAVE_FLAG}"
  "${DUMP_FLAG}"
  --wandb-args "${WANDB_FLAGS}"
  --extra-args "${EXTRA_ARGS}"
)
TRAIN_CMD=(env PYTHONPATH="${MILES_DIR}:${MILES_SCRIPTS_DIR}" python "${SCRIPT_DIR}/train.py" train "${TRAIN_ARGS[@]}")
dry_run_exit "${TRAIN_CMD[@]}"

for path in "${MODEL_ROOT}/DeepSeek-V4-Flash-FP8/config.json" \
            "${MODEL_ROOT}/DeepSeek-V4-Flash-FP8_torch_dist/latest_checkpointed_iteration.txt" \
            "${MILES_DIR}" "${MEGATRON_PATH}"; do
  [[ -e "${path}" ]] || { echo "missing ${path} (see README.md, Step 3)" >&2; exit 2; }
done
# The session server must accept only tool results as new turns from the CAMEL agent; Miles
# versions without this flag reject the command line.
grep -qs -- '--tito-allowed-append-roles' "${MILES_DIR}/miles/utils/arguments.py" || {
  echo "Miles at ${MILES_DIR} has no --tito-allowed-append-roles; see README.md, Versions" >&2
  exit 2
}

configure_ray
env_service_configure  # DATASET_ROOT, CAMEL_* (forwarded to the Ray rollout workers by train.py)
env_service_check_data
env_service_start
write_manifest "$0" "${SCRIPT_DIR}/train.py" "${ENV_SERVICE_CONFIG}"

"${TRAIN_CMD[@]}" 2>&1 | tee "${RUN_ROOT}/logs/train.log"
