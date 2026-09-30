#!/usr/bin/env bash
# Train Inkling-Small (276B-A12B MoE, all parameters) with GRPO on terminal tasks. Rollouts run
# through the Harbor agent server with the CAMEL agent, which uses Inkling's native tool calls.
#
#   cp env.example .env   # then fill it in
#   bash scripts/miles/examples/inkling_grpo/run_harbor_camel.sh
#
# Runs inside the Miles container on the Ray head node, with Ray already started on every
# node (README.md walks through the setup). DRY_RUN=1 prints the resolved training command and
# exits. Every setting below can be set in .env or the environment; the defaults are the
# configuration this recipe was run with, except LR (see below).
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOG_TAG=inkling
source "${SCRIPT_DIR}/../../common/launcher.sh"
load_env_file "${SCRIPT_DIR}"   # settings from ./.env (copy of env.example) or $ENV_FILE

# Tested with
DOCKER_IMAGE=radixark/miles@sha256:946f29396ac313d03b50ac0c0c8930b7eaaa403ae9e7d6115595d57e59cdad12
MILES_COMMIT=38bef2a605191c137323976c211ebdac467b3295            # Michaelsqj/miles: session server parses Inkling completions, longer session-server startup
SGLANG_COMMIT=cb05a44f35a7c9e27e46d74112cc841ca674ef43           # sgl-project/sglang sglang-miles
HARBOR_COMMIT=${HARBOR_COMMIT:-acac1c20e0350f70c60fc6a0755a99d9302b90dd}  # Michaelsqj/harbor: CAMEL feedback switches reach the agent
# CAMEL is pinned by that Harbor commit's uv.lock: Michaelsqj/camel@1c42729b29b6f9f216bab2d08a5e7a52bbb6cf8e

# Inputs. Paths must resolve to the same content on every node.
: "${PROMPT_DATA:?set PROMPT_DATA to a prompt JSONL (see scripts/miles/data/README.md)}"
: "${TASKS_DIR:?set TASKS_DIR to the Harbor task directories referenced by PROMPT_DATA}"
MODEL_ROOT=${MODEL_ROOT:-/root/models}
MODEL_DIR=${MODEL_DIR:-${MODEL_ROOT}/Inkling-Small}                     # Hugging Face checkpoint
TORCH_DIST_DIR=${TORCH_DIST_DIR:-${MODEL_ROOT}/Inkling-Small_torch_dist}  # its Megatron conversion (prepare_model.py)
# Start from an earlier run's weights instead: LOAD_DIR=<old run>/checkpoints. Checkpoints hold
# weights only, so the optimizer state and the rollout count start fresh.
LOAD_DIR=${LOAD_DIR:-${TORCH_DIST_DIR}}
MILES_DIR=${MILES_DIR:-/root/miles}
SGLANG_DIR=${SGLANG_DIR:-/sgl-workspace/sglang}
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}

# Topology. Training and rollout share every GPU (colocated): the actor is TP8/EP8 with one
# pipeline stage per node, and each node serves one TP8 SGLang engine. 8 nodes is the tested
# layout; 6 and 7 nodes (6 or 7 pipeline stages) also resolve but leave less memory headroom.
NUM_NODES=${NUM_NODES:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# Batch shape. Each rollout starts OVER_SAMPLING_BATCH_SIZE prompt groups of
# N_SAMPLES_PER_PROMPT trials and keeps the first ROLLOUT_BATCH_SIZE complete groups with no
# aborted sample; the spare groups absorb slow or failed sandboxes. One optimizer step per rollout.
NUM_ROLLOUT=${NUM_ROLLOUT:-100}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
OVER_SAMPLING_BATCH_SIZE=${OVER_SAMPLING_BATCH_SIZE:-20}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}

# Optimization. Constant LR. For full fine-tuning of this model 3e-5 was the highest LR that
# trained stably; 5e-5 and 1e-4 collapsed.
LR=${LR:-1e-5}
# Weights-only checkpoints (about 550 GB each: 276B parameters in BF16) every SAVE_INTERVAL
# rollouts and after the last one.
SAVE_INTERVAL=${SAVE_INTERVAL:-99}

# Lengths. MAX_SEQ_LEN bounds one training sample (prompt plus every turn); longer samples
# are truncated for training. MAX_RESPONSE_LEN bounds one model turn.
MAX_SEQ_LEN=${MAX_SEQ_LEN:-32768}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-16384}

# Miles session servers, one process per port in [SESSION_SERVER_PORT,
# SESSION_SERVER_PORT + SESSION_SERVERS). The agent sends its model calls there.
SESSION_SERVERS=${SESSION_SERVERS:-64}
SESSION_SERVER_PORT=${SESSION_SERVER_PORT:-30000}
# 1: write per-rollout samples and token ids under RUN_ROOT/dump_details.
DUMP_DETAILS=${DUMP_DETAILS:-1}
EXTRA_ARGS=${EXTRA_ARGS:-}

# Harbor agent: CAMEL. A malformed or length-cut turn ends the trajectory as the policy wrote
# it; no corrective feedback, no retry (common/harbor_agent.py sets this for camel).
export HARBOR_AGENT_NAME=camel
export HARBOR_AGENT_MAX_ITERATIONS=${HARBOR_AGENT_MAX_ITERATIONS:-50}
# The agent's own context bound. Training samples are cut at MAX_SEQ_LEN separately.
export HARBOR_MAX_SEQ_LEN=${HARBOR_MAX_SEQ_LEN:-1048576}
export AGENT_MODEL_NAME=${AGENT_MODEL_NAME:-inkling-small}
# Terminus-2 switches do not apply, and this Harbor commit has no CAMEL compaction switch.
unset HARBOR_TERMINUS_PARSER HARBOR_INTERLEAVED_THINKING HARBOR_TERMINUS_ENABLE_SUMMARIZE HARBOR_CAMEL_MAX_COMPACTIONS
# Run every candidate trial at once (20 groups x 8 = 160).
export AGENT_MAX_CONCURRENT=${AGENT_MAX_CONCURRENT:-$((OVER_SAMPLING_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
# Sandbox size for SANDBOX_BACKEND=gke; the other backends use each task's own resources.
if [[ "${SANDBOX_BACKEND:-daytona}" == gke ]]; then
  export SANDBOX_CPUS=${SANDBOX_CPUS:-1} SANDBOX_MEMORY_MB=${SANDBOX_MEMORY_MB:-2048}
  export SANDBOX_STORAGE_MB=${SANDBOX_STORAGE_MB:-6144}
fi

init_run inkling-grpo-camel
# Shared across runs; a cold cache makes the first actor step compile for a long time.
TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${RUNS_ROOT}/torchinductor_cache}

MILES_EXTRA_ARGS=()
wandb_args MILES_EXTRA_ARGS
[[ -z "${EXTRA_ARGS}" ]] || MILES_EXTRA_ARGS+=("${EXTRA_ARGS}")
if [[ "${DUMP_DETAILS}" == 1 ]]; then DUMP_FLAG=--dump-details; else DUMP_FLAG=--no-dump-details; fi

TRAIN_ARGS=(
  --agent camel
  --num-nodes "${NUM_NODES}"
  --num-gpus-per-node "${GPUS_PER_NODE}"
  --prompt-data "${PROMPT_DATA}"
  --hf-checkpoint "${MODEL_DIR}"
  --torch-dist "${TORCH_DIST_DIR}"
  --load "${LOAD_DIR}"
  --megatron-path "${MEGATRON_PATH}"
  --output-dir "${RUN_ROOT}"
  --torchinductor-cache-dir "${TORCHINDUCTOR_CACHE_DIR}"
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --lr "${LR}"
  --save-interval "${SAVE_INTERVAL}"
  --max-seq-len "${MAX_SEQ_LEN}"
  --max-response-len "${MAX_RESPONSE_LEN}"
  --session-server-port-start "${SESSION_SERVER_PORT}"
  --session-server-port-end "$((SESSION_SERVER_PORT + SESSION_SERVERS))"
  "${DUMP_FLAG}"
  --extra-args "${MILES_EXTRA_ARGS[*]}"
)
TRAIN_CMD=(env PYTHONPATH="${MILES_DIR}:${MILES_SCRIPTS_DIR}" python "${SCRIPT_DIR}/train.py" train "${TRAIN_ARGS[@]}")
dry_run_exit "${TRAIN_CMD[@]}"

# Preflight: model, checkouts and the Miles features this recipe cannot train without.
for path in "${MODEL_DIR}/config.json" "${TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" \
            "${LOAD_DIR}/latest_checkpointed_iteration.txt" \
            "${MILES_DIR}" "${MEGATRON_PATH}"; do
  [[ -e "${path}" ]] || { echo "missing ${path} (see README.md, Step 3)" >&2; exit 2; }
done
grep -qs 'InklingResponseParser' "${MILES_DIR}/miles/utils/chat_template_utils/tito_tokenizer.py" || {
  echo "Miles at ${MILES_DIR} cannot parse Inkling completions in the session server, which CAMEL" >&2
  echo "needs with SGLang's parsers off; check out Michaelsqj/miles@${MILES_COMMIT}" >&2
  exit 2
}
if grep -qs 'wait_for_server_ready(ip, port, process, timeout=30)' "${MILES_DIR}/miles/ray/rollout/router_manager.py"; then
  echo "Miles at ${MILES_DIR} gives the session servers 30 s to start; ${SESSION_SERVERS} processes importing" >&2
  echo "the tokenizer stack at once need longer; check out Michaelsqj/miles@${MILES_COMMIT}" >&2
  exit 2
fi
check_git_pin miles "${MILES_DIR}" "${MILES_COMMIT}"
# Checks the head node only; every node must run the same SGLang (README.md, Step 1).
check_git_pin sglang "${SGLANG_DIR}" "${SGLANG_COMMIT}"
harbor_check_data

configure_ray
harbor_install
# response_feedback=false must reach the CAMEL agent itself; nested in the model kwargs it is
# ignored and length-cut turns get corrective feedback.
harbor_require_source src/harbor/agent_server/compat.py 'sampling\.pop\("response_feedback"' \
  'CAMEL response_feedback/max_response_feedback would be ignored'
# CAMEL's library logging at INFO writes every request payload and can fill the run disk.
harbor_require_source src/harbor/agents/camel/camel.py '^disable_logging\(\)' \
  'CAMEL library logging is on in the agent workers'
harbor_configure
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
harbor_start_server
write_manifest "$0" "${SCRIPT_DIR}/train.py"

"${TRAIN_CMD[@]}" 2>&1 | tee "${RUN_ROOT}/logs/train.log"
