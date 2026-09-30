#!/usr/bin/env bash
# Train DeepSeek-V4-Flash-FP8 (all parameters) with GRPO on terminal tasks. Rollouts run through
# the Harbor agent server with Harbor's Terminus-2 agent. Sync Miles train.py: actor and SGLang
# engines share every GPU, routing replay (R3) is on, and the surplus of each over-sampled batch
# is cancelled in Harbor through /flush.
#
#   cp env.example .env   # fill in, then:
#   bash scripts/miles/examples/deepseek_v4_grpo/run_harbor_terminus2.sh
#
# Runs inside the Miles container on the Ray head node, with Ray started on every node.
# DRY_RUN=1 prints the resolved training command and exits. Every setting below is an
# environment variable whose default is the configuration this recipe was run with.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOG_TAG=deepseek-v4
source "${SCRIPT_DIR}/../../common/launcher.sh"
load_env_file "${SCRIPT_DIR}"

# Tested with
# shellcheck disable=SC2034  # informational: this script runs inside this image
DOCKER_IMAGE=radixark/miles@sha256:ca0bb593dd6f4011b444f64d478b72c213e4c70421f4d7f94e593a709562429e
MILES_COMMIT=5d3c77e66281871245ce43e32b3c0c8c96f147e8  # Michaelsqj/miles: upstream Miles + R3 tolerates the extra routed-experts row of a response that ends on a stop token
HARBOR_COMMIT=${HARBOR_COMMIT:-1e02f96f00a18b5e9158a4baffbaa2bb5a0b2146}  # Michaelsqj/harbor: agent server (/run, /flush) with per-trial subprocess workers

# Inputs. Paths must resolve to the same content on every node.
: "${PROMPT_DATA:?set PROMPT_DATA to a prompt JSONL (see scripts/miles/data/README.md)}"
: "${TASKS_DIR:?set TASKS_DIR to the Harbor task directories referenced by PROMPT_DATA}"
MODEL_ROOT=${MODEL_ROOT:-/root/models}  # DeepSeek-V4-Flash-FP8 and DeepSeek-V4-Flash-FP8_torch_dist (train.py prepare)
MILES_DIR=${MILES_DIR:-/root/miles}
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}

# Topology. Colocated: every GPU trains and serves. The actor is TP8 / EP8 with one pipeline
# stage per node; SGLang runs 4-GPU engines (TP4 / EP4), two per node. 8 nodes is the tested
# layout; 5, 6 and 7 nodes also have a Megatron layout.
NUM_NODES=${NUM_NODES:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
HARDWARE=${HARDWARE:-auto}  # FP8 scale format: Hopper vs Blackwell (auto-detected)

# Batch shape. Each rollout starts OVER_SAMPLING_BATCH_SIZE prompt groups of N_SAMPLES_PER_PROMPT
# trials and keeps the first ROLLOUT_BATCH_SIZE complete groups without an aborted sample; the
# rest are aborted. One optimizer step per rollout (global batch = 16 x 8 = 128 trajectories).
NUM_ROLLOUT=${NUM_ROLLOUT:-3000}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
OVER_SAMPLING_BATCH_SIZE=${OVER_SAMPLING_BATCH_SIZE:-24}
TEMPERATURE=${TEMPERATURE:-1.0}

# Optimization: constant LR, Adam state offloaded to the CPU.
LR=${LR:-1e-6}
# Weights-only checkpoints every SAVE_INTERVAL rollouts. SKIP_SAVING=1 writes none.
SAVE_INTERVAL=${SAVE_INTERVAL:-50}
SKIP_SAVING=${SKIP_SAVING:-0}

# Lengths. MAX_RESPONSE_LEN bounds one model call. MAX_SEQ_LEN bounds one training sample
# (prompt plus every turn); longer trajectories are truncated for training.
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-16384}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-65536}
# 1: write per-rollout samples and tensors under RUN_ROOT/dump_details.
DUMP_DETAILS=${DUMP_DETAILS:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}  # extra Miles flags, appended last (they override the ones above)
export WANDB_PROJECT=${WANDB_PROJECT:-deepseek-v4-grpo}

# Harbor agent: Terminus-2 with its defaults. This Harbor commit forwards unknown agent options
# into the model call, so the parser and summarization switches must stay unset.
export HARBOR_AGENT_NAME=terminus-2
unset HARBOR_TERMINUS_PARSER HARBOR_TERMINUS_ENABLE_SUMMARIZE HARBOR_CAMEL_MAX_COMPACTIONS
export HARBOR_AGENT_MAX_ITERATIONS=${HARBOR_AGENT_MAX_ITERATIONS:-50}  # turns per trial
# The agent's own context bound, effectively unbounded: a trial ends on the turn cap or the
# task's agent timeout. Training samples are cut at MAX_SEQ_LEN separately.
export HARBOR_MAX_SEQ_LEN=${HARBOR_MAX_SEQ_LEN:-1048576}
export HARBOR_AGENT_TIMEOUT_MULTIPLIER=${HARBOR_AGENT_TIMEOUT_MULTIPLIER:-12}  # x each task's agent timeout
# Deadline for one trial as seen by the rollout worker: queue + sandbox start + agent + verifier.
export HARBOR_AGENT_CALL_TIMEOUT_SEC=${HARBOR_AGENT_CALL_TIMEOUT_SEC:-10800}
# 24 groups x 8 = 192 trials start per rollout; the ones above this limit wait in the server queue.
export AGENT_MAX_CONCURRENT=${AGENT_MAX_CONCURRENT:-128}

init_run deepseek-v4-grpo-terminus2

WANDB_ARGS=()
wandb_args WANDB_ARGS
WANDB_FLAGS=""
if (( ${#WANDB_ARGS[@]} )); then WANDB_FLAGS=$(printf '%q ' "${WANDB_ARGS[@]}"); fi
if [[ "${SKIP_SAVING}" == 1 ]]; then SAVE_FLAG=--skip-saving; else SAVE_FLAG=--no-skip-saving; fi
if [[ "${DUMP_DETAILS}" == 1 ]]; then DUMP_FLAG=--dump-details; else DUMP_FLAG=--no-dump-details; fi

TRAIN_ARGS=(
  --rollout-backend harbor
  --num-nodes "${NUM_NODES}"
  --num-gpus-per-node "${GPUS_PER_NODE}"
  --hardware "${HARDWARE}"
  --model-root "${MODEL_ROOT}"
  --megatron-path "${MEGATRON_PATH}"
  --prompt-data "${PROMPT_DATA}"
  --run-root "${RUN_ROOT}"
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}"
  --temperature "${TEMPERATURE}"
  --max-response-len "${MAX_RESPONSE_LEN}"
  --max-seq-len "${MAX_SEQ_LEN}"
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
check_git_pin miles "${MILES_DIR}" "${MILES_COMMIT}"
harbor_check_data

configure_ray
harbor_install
# Harbor features this recipe cannot train without.
harbor_require_source src/harbor/agent_server/app.py '@app\.post\("/flush"\)' \
  'the surplus trials of an over-sampled batch could not be cancelled'
harbor_require_source src/harbor/agent_server/compat.py 'pop\("max_iterations"' \
  'HARBOR_AGENT_MAX_ITERATIONS would not cap the Terminus-2 turns'
harbor_require_source src/harbor/agent_server/compat.py 'max_input_tokens.*max_seq_len' \
  'HARBOR_MAX_SEQ_LEN would not bound the Terminus-2 context'
# common/harbor_server.sh passes these flags only when the variable is set.
for pair in HARBOR_AGENT_TIMEOUT_SEC:agent-timeout-sec HARBOR_EXTRA_ARTIFACTS:artifact HARBOR_EXTRA_COLLECT:collect; do
  var=${pair%%:*} flag=${pair#*:}
  [[ -z "${!var:-}" ]] || harbor_require_source src/harbor/agent_server/cli.py "\"--${flag}\"" "${var} is set"
done
harbor_configure
harbor_start_server
write_manifest "$0" "${SCRIPT_DIR}/train.py"

"${TRAIN_CMD[@]}" 2>&1 | tee "${RUN_ROOT}/logs/train.log"
