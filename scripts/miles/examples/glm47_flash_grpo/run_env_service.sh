#!/usr/bin/env bash
# Train GLM-4.7-Flash (all parameters) with GRPO on terminal tasks. Rollouts run through the
# seta env_service with seta's CAMEL agent in Daytona sandboxes, talking to the model through
# the Miles session server. Miles train_async.py: seven nodes serve SGLang, one node trains,
# and a continuous rollout worker keeps sampling while the trainer steps.
#
#   cp env.example .env   # fill in, then:
#   bash scripts/miles/examples/glm47_flash_grpo/run_env_service.sh
#
# Runs inside the Miles container on the Ray head node, with Ray started on every node; the
# env_service runs on the head node next to it. DRY_RUN=1 prints the resolved training command
# and exits. Every setting below is an environment variable whose default is the configuration
# this recipe was run with.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOG_TAG=glm47-flash
# shellcheck source=../../common/launcher.sh
source "${SCRIPT_DIR}/../../common/launcher.sh"
# shellcheck source=../../common/env_service.sh
source "${SCRIPT_DIR}/../../common/env_service.sh"
load_env_file "${SCRIPT_DIR}"

# Tested with
# Docker image: the radixark/miles image of MILES_COMMIT (SGLang v0.5.13 base with the
# sglang-miles branch, transformers with native GLM-4.7-Flash support); the reference run did
# not record its digest. See README.md, Versions.
# Upstream radixark/miles main (also in Michaelsqj/miles); no fork changes are needed.
MILES_COMMIT=46847f29378f2e3ff4f29898bddcee35a1f67b8d

# Inputs. Paths must resolve to the same content on every node.
: "${PROMPT_DATA:?set PROMPT_DATA to a prompt JSONL or parquet (README.md, Step 5)}"
: "${TASKS_DIR:?set TASKS_DIR to the task directories referenced by PROMPT_DATA}"
MODEL_ROOT=${MODEL_ROOT:-/root/models}  # GLM-4.7-Flash and GLM-4.7-Flash_torch_dist (train.py prepare)
MILES_DIR=${MILES_DIR:-/root/miles}
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}

# Topology. Rollout (sandbox and tool time) bounds throughput, so ROLLOUT_NUM_NODES nodes serve
# (two TP4 SGLang engines per node) and the rest train: one node is enough for GLM-4.7-Flash at
# TP4 x EP8 and keeps its rate close to what rollout delivers.
NUM_NODES=${NUM_NODES:-8}
ROLLOUT_NUM_NODES=${ROLLOUT_NUM_NODES:-7}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# Batch shape. Each optimizer step trains on ROLLOUT_BATCH_SIZE prompt groups of
# N_SAMPLES_PER_PROMPT trajectories (16 x 16 = 256). Independently of that the rollout worker
# keeps ROLLOUT_CONCURRENCY groups in flight: 30 x 16 = 480 trajectories for MAX_SLOTS = 400
# sandboxes, so every sandbox slot stays busy and a step takes the groups that finish first.
# Groups generated more than MAX_WEIGHT_STALENESS weight updates ago are re-queued.
NUM_ROLLOUT=${NUM_ROLLOUT:-3000}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}
ROLLOUT_CONCURRENCY=${ROLLOUT_CONCURRENCY:-30}
MAX_WEIGHT_STALENESS=${MAX_WEIGHT_STALENESS:-4}

# Group filter: drop groups whose trajectories all got the same reward (no GRPO signal) and
# groups with any trajectory whose sandbox or env_service failed (their reward says nothing
# about the policy). The worker replaces a dropped group with a new prompt.
GROUP_FILTER_MIN_REWARD_STD=${GROUP_FILTER_MIN_REWARD_STD:-1e-8}
GROUP_FILTER_MAX_ENV_FAILURES=${GROUP_FILTER_MAX_ENV_FAILURES:-1}

# Optimization: constant LR, Adam state offloaded to the CPU. Weights-only checkpoints every
# SAVE_INTERVAL steps, all kept.
LR=${LR:-1e-6}
SAVE_INTERVAL=${SAVE_INTERVAL:-50}
# 1: write per-step rollout and train tensors under RUN_ROOT/dump_details (large).
DUMP_DETAILS=${DUMP_DETAILS:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}  # extra Miles flags, appended last (they override the ones above)
export WANDB_PROJECT=${WANDB_PROJECT:-glm47-flash-grpo}

# env_service (common/env_service.sh). The agent, its limits and the sandbox size are in
# env_service.yaml.
export ENV_SERVICE_CONFIG=${ENV_SERVICE_CONFIG:-${SCRIPT_DIR}/env_service.yaml}
export ENV_SERVICE_PORT=${ENV_SERVICE_PORT:-8002}
export MAX_SLOTS=${MAX_SLOTS:-400}  # concurrent trajectories = live Daytona sandboxes
# One trajectory end to end; must exceed the sum of the task_timeouts in env_service.yaml
# (4260 s). Run in a killable subprocess so this cap also frees a slot held by a hung call.
export STEP_TIMEOUT_SECONDS=${STEP_TIMEOUT_SECONDS:-5000}
export STEP_USE_SUBPROCESS=${STEP_USE_SUBPROCESS:-1}
# Image builds: up to BUILD_CONCURRENCY at a time, at most INFLIGHT_LEAD trajectories built
# ahead of a free slot, and at most HARBOR_DAYTONA_MAX_CREATES concurrent Daytona create calls,
# which keeps the burst of a 400-slot pool under Daytona's create rate limit.
export BUILD_CONCURRENCY=${BUILD_CONCURRENCY:-24}
export INFLIGHT_LEAD=${INFLIGHT_LEAD:-64}
export HARBOR_DAYTONA_MAX_CREATES=${HARBOR_DAYTONA_MAX_CREATES:-24}
# Build task images through Daytona's declarative image cache rather than named snapshots,
# which count against a per-organization snapshot quota. The eviction age applies only to
# named snapshots (DAYTONA_DECLARATIVE=0).
export DAYTONA_DECLARATIVE=${DAYTONA_DECLARATIVE:-1}
export DAYTONA_SNAPSHOT_EVICT_AGE_HOURS=${DAYTONA_SNAPSHOT_EVICT_AGE_HOURS:-3}
# The agent's client timeout for one model call. GLM-4.7-Flash turns of up to 8192 tokens
# under full load exceed the 180 s default; 900 s is still well inside the agent's 2400 s budget.
export MODEL_TIMEOUT=${MODEL_TIMEOUT:-900}

init_run glm47-flash-grpo
# HEAD_IP (or CLUSTER_CONFIG) is needed for the env_service URL the rollout workers use; a dry
# run works without it.
if [[ -n "${HEAD_IP:-}${CLUSTER_CONFIG:-}" || "${DRY_RUN:-0}" != 1 ]]; then configure_ray; fi
env_service_configure

WANDB_ARGS=()
wandb_args WANDB_ARGS
WANDB_FLAGS=""
if (( ${#WANDB_ARGS[@]} )); then WANDB_FLAGS=$(printf '%q ' "${WANDB_ARGS[@]}"); fi
if [[ "${DUMP_DETAILS}" == 1 ]]; then DUMP_FLAG=--dump-details; else DUMP_FLAG=--no-dump-details; fi

TRAIN_ARGS=(
  --num-nodes "${NUM_NODES}"
  --rollout-num-nodes "${ROLLOUT_NUM_NODES}"
  --num-gpus-per-node "${GPUS_PER_NODE}"
  --model-root "${MODEL_ROOT}"
  --megatron-path "${MEGATRON_PATH}"
  --prompt-data "${PROMPT_DATA}"
  --run-root "${RUN_ROOT}"
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-concurrency "${ROLLOUT_CONCURRENCY}"
  --max-weight-staleness "${MAX_WEIGHT_STALENESS}"
  --group-filter-min-reward-std "${GROUP_FILTER_MIN_REWARD_STD}"
  --group-filter-max-env-failures "${GROUP_FILTER_MAX_ENV_FAILURES}"
  --lr "${LR}"
  --save-interval "${SAVE_INTERVAL}"
  "${DUMP_FLAG}"
  --env-service-url "${CAMEL_ENV_SERVICE_URL}"
  --dataset-name "${CAMEL_DATASET_NAME}"
  --trial-name "${CAMEL_TRIAL_NAME}"
  --wandb-args "${WANDB_FLAGS}"
  --extra-args "${EXTRA_ARGS}"
)
TRAIN_CMD=(env PYTHONPATH="${MILES_DIR}:${MILES_SCRIPTS_DIR}" python "${SCRIPT_DIR}/train.py" train "${TRAIN_ARGS[@]}")
dry_run_exit "${TRAIN_CMD[@]}"

# Preflight: model, checkouts, the Miles features this recipe cannot run without, data.
for path in "${MODEL_ROOT}/GLM-4.7-Flash/config.json" \
            "${MODEL_ROOT}/GLM-4.7-Flash_torch_dist/latest_checkpointed_iteration.txt" \
            "${MILES_DIR}" "${MEGATRON_PATH}"; do
  [[ -e "${path}" ]] || { echo "missing ${path} (see README.md, Steps 1 and 3)" >&2; exit 2; }
done
miles_require() {  # <path under MILES_DIR> <grep -E pattern> <what breaks>
  grep -qsE -- "$2" "${MILES_DIR}/$1" || {
    echo "Miles at ${MILES_DIR} lacks '$2' in $1: $3 (tested with radixark/miles@${MILES_COMMIT})" >&2
    exit 2
  }
}
miles_require scripts/models/glm4.7-flash.sh 'MODEL_ARGS' 'no Megatron model arguments for glm4.7-flash'
miles_require miles/utils/arguments.py '--tito-allowed-append-roles' \
  'the session server cannot be told that the agent appends user and tool turns'
miles_require miles/rollout/sglang_rollout.py 'def generate_and_rm_group' \
  'common.env_service_rollout cannot run its rollout worker'
miles_require miles/utils/types.py 'def reset_for_retry' '--max-weight-staleness cannot re-queue stale groups'
check_git_pin miles "${MILES_DIR}" "${MILES_COMMIT}"
env_service_check_data

env_service_start
write_manifest "$0" "${SCRIPT_DIR}/train.py" "${ENV_SERVICE_CONFIG}"

"${TRAIN_CMD[@]}" 2>&1 | tee "${RUN_ROOT}/logs/train.log"
