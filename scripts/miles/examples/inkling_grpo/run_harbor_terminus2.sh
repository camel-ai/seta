#!/usr/bin/env bash
# Train Inkling-Small (276B-A12B MoE, all parameters) with GRPO on terminal tasks. Rollouts run
# through the Harbor agent server with the Terminus-2 agent (JSON commands in plain text).
#
#   cp env.example .env   # then fill it in
#   bash scripts/miles/examples/inkling_grpo/run_harbor_terminus2.sh
#
# Runs inside the Miles container on the Ray head node, with Ray already started on every
# node and the SGLang build below installed on every node (README.md walks through the setup).
# DRY_RUN=1 prints the resolved training command and exits. Every setting below can be set in
# .env or the environment; the defaults are the configuration this recipe was run with, except
# LR (see below).
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOG_TAG=inkling
source "${SCRIPT_DIR}/../../common/launcher.sh"
load_env_file "${SCRIPT_DIR}"   # settings from ./.env (copy of env.example) or $ENV_FILE

# Tested with
DOCKER_IMAGE=radixark/miles@sha256:946f29396ac313d03b50ac0c0c8930b7eaaa403ae9e7d6115595d57e59cdad12
MILES_COMMIT=80a25cb568982b8e445498ab7d3a0fc3d5d3670e            # Michaelsqj/miles: keeps Inkling's ordered thinking/text blocks through token-in/token-out; session-server startup scales with the pool
SGLANG_COMMIT=baccf651fe984a825c35b2554db3572b31e9e25a           # michaelsqj/sglang (on sglang-miles): returns Inkling's ordered thinking/text blocks
HARBOR_COMMIT=${HARBOR_COMMIT:-1e02f96f00a18b5e9158a4baffbaa2bb5a0b2146}  # Michaelsqj/harbor: agent server with per-trial subprocess workers

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
# trained stably; at 5e-5 this Terminus-2 setup degraded (see README.md, Notes).
LR=${LR:-1e-5}
# Weights-only checkpoints (about 550 GB each: 276B parameters in BF16) every SAVE_INTERVAL
# rollouts and after the last one.
SAVE_INTERVAL=${SAVE_INTERVAL:-50}

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

# Harbor agent: Terminus-2 with its default JSON format and interleaved thinking (earlier
# turns keep their reasoning). This Harbor commit has no parser or summarization switch, so
# those variables must stay unset.
export HARBOR_AGENT_NAME=terminus-2
export HARBOR_AGENT_MAX_ITERATIONS=${HARBOR_AGENT_MAX_ITERATIONS:-50}
# The agent's own context bound. Training samples are cut at MAX_SEQ_LEN separately.
export HARBOR_MAX_SEQ_LEN=${HARBOR_MAX_SEQ_LEN:-1048576}
export HARBOR_INTERLEAVED_THINKING=${HARBOR_INTERLEAVED_THINKING:-true}
export AGENT_MODEL_NAME=${AGENT_MODEL_NAME:-inkling-small}
unset HARBOR_TERMINUS_PARSER HARBOR_TERMINUS_ENABLE_SUMMARIZE HARBOR_CAMEL_MAX_COMPACTIONS
# Run every candidate trial at once (20 groups x 8 = 160).
export AGENT_MAX_CONCURRENT=${AGENT_MAX_CONCURRENT:-$((OVER_SAMPLING_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
# Sandbox size for SANDBOX_BACKEND=gke; the other backends use each task's own resources.
if [[ "${SANDBOX_BACKEND:-daytona}" == gke ]]; then
  export SANDBOX_CPUS=${SANDBOX_CPUS:-1} SANDBOX_MEMORY_MB=${SANDBOX_MEMORY_MB:-2048}
  export SANDBOX_STORAGE_MB=${SANDBOX_STORAGE_MB:-6144}
fi

init_run inkling-grpo-terminus2
# Shared across runs; a cold cache makes the first actor step compile for a long time.
TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${RUNS_ROOT}/torchinductor_cache}

MILES_EXTRA_ARGS=()
wandb_args MILES_EXTRA_ARGS
[[ -z "${EXTRA_ARGS}" ]] || MILES_EXTRA_ARGS+=("${EXTRA_ARGS}")
if [[ "${DUMP_DETAILS}" == 1 ]]; then DUMP_FLAG=--dump-details; else DUMP_FLAG=--no-dump-details; fi

TRAIN_ARGS=(
  --agent terminus-2
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

# Preflight: model, checkouts and the Miles/SGLang features this recipe cannot train without.
for path in "${MODEL_DIR}/config.json" "${TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" \
            "${LOAD_DIR}/latest_checkpointed_iteration.txt" \
            "${MILES_DIR}" "${MEGATRON_PATH}"; do
  [[ -e "${path}" ]] || { echo "missing ${path} (see README.md, Step 3)" >&2; exit 2; }
done
# Terminus-2 replays the chat history as plain messages. Without the ordered-block fix the
# replayed thinking/text blocks no longer re-tokenize to the tokens the engine generated.
grep -qs '_preserve_ordered_content_blocks' "${MILES_DIR}/miles/rollout/session/linear_trajectory.py" || {
  echo "Miles at ${MILES_DIR} does not keep Inkling's ordered thinking/text blocks through the session" >&2
  echo "server; check out Michaelsqj/miles@${MILES_COMMIT}" >&2
  exit 2
}
grep -qs 'content_blocks' "${SGLANG_DIR}/python/sglang/srt/entrypoints/openai/protocol.py" || {
  echo "SGLang at ${SGLANG_DIR} does not return Inkling's ordered thinking/text blocks; install" >&2
  echo "michaelsqj/sglang@${SGLANG_COMMIT} on every node (README.md, Step 1)" >&2
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
harbor_require_source src/harbor/agent_server/compat.py 'interleaved_thinking' \
  'HARBOR_INTERLEAVED_THINKING would be ignored'
harbor_configure
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
harbor_start_server
write_manifest "$0" "${SCRIPT_DIR}/train.py"

"${TRAIN_CMD[@]}" 2>&1 | tee "${RUN_ROOT}/logs/train.log"
