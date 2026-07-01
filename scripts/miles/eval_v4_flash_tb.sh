#!/usr/bin/env bash
# DeepSeek-V4-Flash EVAL against terminal-bench datasets.
#
# Parameterized via env vars:
#   DATASET           — dataset name (terminal-bench-2.0 | terminal-bench-core-0.1.1_migrated)
#   LOAD_DIR          — empty (base model) | <path>/checkpoints (trained model)
#   EVAL_N_SAMPLES    — trajectories per task (default 8)
#   WANDB_GROUP       — wandb group name (default derived from DATASET + LOAD_DIR)
#
# Usage examples:
#   # base model on tb-2.0:
#   DATASET=terminal-bench-2.0 bash eval_v4_flash_tb.sh
#
#   # iter_99 model on tb-2.0:
#   DATASET=terminal-bench-2.0 \
#     LOAD_DIR=/data/training_runs/camel-combined-988tasks-20260514-231723/checkpoints \
#     bash eval_v4_flash_tb.sh

set -euo pipefail

# ─── Required env ──────────────────────────────────────────────────────────────
: "${DATASET:?DATASET env var required: terminal-bench-2.0 or terminal-bench-core-0.1.1_migrated}"
LOAD_DIR="${LOAD_DIR:-}"

# ─── Hard-coded cluster config ────────────────────────────────────────────────
HEAD_IP="10.220.51.32"
HEAD_HOST="h200-10-220-51-32"
GCS_PORT=6379
RAY_DASHBOARD_PORT=8265
ENV_SERVICE_PORT=8002
MAX_SLOTS="${MAX_SLOTS:-196}"

NUM_NODES="${NUM_NODES:-8}"
NUM_GPUS_PER_NODE=8

# Dataset + parquet (we built these from instruction.md per task)
DATASET_NAME="$DATASET"
DATASET_DIR="${DATASET_DIR:-/data/terminal_agent/dataset/${DATASET_NAME}}"
# PARQUET_PATH overridable so we can run a SUBSET of tasks (backfill) while keeping
# DATASET_DIR pointed at the full per-task build files.
PARQUET_PATH="${PARQUET_PATH:-/data/terminal_agent/dataset/${DATASET_NAME}.parquet}"

MODEL_NAME="DeepSeek-V4-Flash-FP8"
MODEL_DIR="/data/models"
HF_CHECKPOINT="${MODEL_DIR}/${MODEL_NAME}"
TORCH_DIST_DIR="${MODEL_DIR}/${MODEL_NAME}_torch_dist"

REPO_ROOT="/data/terminal_agent"
LAUNCHER="${REPO_ROOT}/scripts/miles/run_deepseek_v4.py"
GENERATE_FN_DIR="${REPO_ROOT}/scripts/miles"
MEGATRON_PATH="/root/Megatron-LM"
SETA_ENV_EXTRA_PYTHONPATH="${REPO_ROOT}:${GENERATE_FN_DIR}"
EXPORTED_PYTHONPATH="${SETA_ENV_EXTRA_PYTHONPATH}:${MEGATRON_PATH}"

# wandb identity
WANDB_ORG="${WANDB_ORG:-zhichenzeng_zzz-org}"
WANDB_ENTITY="${WANDB_ENTITY:-eigent_radixark_training}"
WANDB_PROJECT="${WANDB_PROJECT:-v4flash-terminal-grpo}"

# Auto-name based on dataset and base/trained mode
if [ -z "${WANDB_GROUP:-}" ]; then
  if [ -n "$LOAD_DIR" ]; then
    WANDB_GROUP="eval-${DATASET_NAME}-trained"
  else
    WANDB_GROUP="eval-${DATASET_NAME}-base"
  fi
fi
RUN_STAMP="$(date -u +%Y%m%d-%H%M%S)"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${WANDB_GROUP}-${RUN_STAMP}}"

RUN_ROOT="${RUN_ROOT:-/data/training_runs/${WANDB_RUN_NAME}}"
ENV_SERVICE_LOG_DIR="${RUN_ROOT}/env_service"
DUMP_DETAILS_DIR="${DUMP_DETAILS_DIR:-${RUN_ROOT}/dump_details}"
WANDB_DIR="${WANDB_DIR:-${RUN_ROOT}/wandb}"
HARBOR_ROOT="${HARBOR_ROOT:-${RUN_ROOT}}"

EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-8}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-2048}"

# ─── Stage 1: pre-flight ──────────────────────────────────────────────────────
echo "═══ stage 1: pre-flight checks ═══"
echo "  DATASET=$DATASET_NAME"
echo "  PARQUET=$PARQUET_PATH"
echo "  LOAD_DIR=${LOAD_DIR:-(none, base model)}"
echo "  N_TRAJ_PER_TASK=$EVAL_N_SAMPLES"

actual_host="$(hostname)"
[ "$actual_host" = "$HEAD_HOST" ] || { echo "[fatal] run on $HEAD_HOST" >&2; exit 1; }

for required in "$PARQUET_PATH" "$DATASET_DIR" "$HF_CHECKPOINT" "$LAUNCHER"; do
  [ -e "$required" ] || { echo "[fatal] missing: $required" >&2; exit 1; }
done
[ -f "${TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" ] || { echo "[fatal] missing torch_dist marker"; exit 1; }

# Resume-eval hook: if LOAD_DIR is set, verify it has a valid checkpoint.
if [ -n "$LOAD_DIR" ]; then
  [ -f "${LOAD_DIR}/latest_checkpointed_iteration.txt" ] || {
    echo "[fatal] LOAD_DIR=$LOAD_DIR has no latest_checkpointed_iteration.txt" >&2; exit 1;
  }
  echo "[eval-trained] loading iter $(cat ${LOAD_DIR}/latest_checkpointed_iteration.txt) from ${LOAD_DIR}"
fi

mkdir -p "$RUN_ROOT" "$ENV_SERVICE_LOG_DIR" "$DUMP_DETAILS_DIR" "$WANDB_DIR"
mkdir -p "${RUN_ROOT}/trials" "${RUN_ROOT}/_builds"
echo ""

# ─── Stage 2: env_service ─────────────────────────────────────────────────────
echo "═══ stage 2: env_service (MAX_SLOTS=$MAX_SLOTS) ═══"
TMUX_SESSION=train1 bash "${REPO_ROOT}/scripts/miles/core/env_service.sh" stop 2>&1 | tail -2 || true
MAX_SLOTS=$MAX_SLOTS TMUX_SESSION=train1 LOG_DIR="$ENV_SERVICE_LOG_DIR" HARBOR_ROOT="$HARBOR_ROOT" \
  HARBOR_DAYTONA_MAX_CREATES="${HARBOR_DAYTONA_MAX_CREATES:-16}" \
  CONFIG="${CONFIG:-${REPO_ROOT}/scripts/miles/core/configs/seta_env_config.yaml}" \
  DAYTONA_SNAPSHOT_EVICT_AGE_HOURS="${DAYTONA_SNAPSHOT_EVICT_AGE_HOURS:-0}" \
  bash "${REPO_ROOT}/scripts/miles/core/env_service.sh" start

# ─── Stage 3: health ──────────────────────────────────────────────────────────
echo "═══ stage 3: env_service health ═══"
TMUX_SESSION=train1 bash "${REPO_ROOT}/scripts/miles/core/env_service.sh" wait
curl -fsS "http://127.0.0.1:${ENV_SERVICE_PORT}/health" || { echo "[fatal] env_service /health failed"; exit 1; }
echo ""

# ─── Stage 4: ray sanity ──────────────────────────────────────────────────────
echo "═══ stage 4: ray cluster ═══"
ray status >/tmp/ray_status.out 2>&1 || { echo "[fatal] ray status failed"; cat /tmp/ray_status.out >&2; exit 1; }
cat /tmp/ray_status.out

reported_gpus=$(grep -oE "[0-9]+\.[0-9]+/[0-9]+\.[0-9]+ GPU" /tmp/ray_status.out | awk -F/ '{print $2}' | awk '{print $1}' | head -1)
expected_gpus="$((NUM_NODES * NUM_GPUS_PER_NODE))"
# Eval needs at least expected_gpus; extra cluster nodes stay idle (Ray schedules
# the actor on NUM_NODES of them). Relaxed from == to >= so an 8-node eval can run
# on a 9-node cluster.
[ "${reported_gpus%.*}" -ge "$expected_gpus" ] || { echo "[fatal] need >= $expected_gpus GPUs, got $reported_gpus"; exit 1; }

# Stop any RUNNING ray submissions
python3 -c "
from ray.job_submission import JobSubmissionClient
c = JobSubmissionClient('http://127.0.0.1:${RAY_DASHBOARD_PORT}')
for j in c.list_jobs():
    if str(j.status) == 'RUNNING' and j.submission_id:
        print(j.submission_id)
" 2>/dev/null | while read -r sid; do
  [ -z "$sid" ] && continue
  ray job stop "$sid" 2>&1 | tail -1 || true
done
echo ""

# ─── Stage 5: launch eval ─────────────────────────────────────────────────────
echo "═══ stage 5: launch eval ═══"

# Resume hook: --load points at iter checkpoint if LOAD_DIR set
RESUME_ARGS=""
if [ -n "$LOAD_DIR" ]; then
  RESUME_ARGS="--load ${LOAD_DIR} --no-load-optim"
fi

EXTRA_ARGS="\
--debug-rollout-only \
--num-rollout 0 \
--eval-interval 1 \
--rollout-batch-size 4 \
${RESUME_ARGS} \
--dump-details ${DUMP_DETAILS_DIR} \
--use-wandb \
--wandb-project ${WANDB_PROJECT} \
--wandb-entity ${WANDB_ENTITY} \
--wandb-group ${WANDB_GROUP} \
--wandb-exp-name ${WANDB_RUN_NAME} \
--wandb-dir ${WANDB_DIR} \
--disable-wandb-random-suffix"

export CAMEL_TRIAL_NAME="${WANDB_RUN_NAME}"
export CAMEL_ENV_SERVICE_URL="http://${HEAD_IP}:${ENV_SERVICE_PORT}"
export CAMEL_DATASET_NAME="${DATASET_NAME}"
export MILES_SCRIPT_EXTERNAL_RAY=1
export MASTER_ADDR="${HEAD_IP}"
export RAY_ADDRESS="http://${HEAD_IP}:${RAY_DASHBOARD_PORT}"
export PYTHONPATH="${EXPORTED_PYTHONPATH}"
export WANDB_ENTITY="${WANDB_ENTITY}"
export WANDB_DIR="${WANDB_DIR}"
export WANDB_TAGS="${WANDB_TAGS:-org=${WANDB_ORG},task=camel_terminal_agent,dataset=${DATASET_NAME},mode=eval}"
[ -n "${WANDB_API_KEY:-}" ] || echo "[warn] WANDB_API_KEY not set"

echo "[eval] wandb: project=${WANDB_PROJECT}  group=${WANDB_GROUP}  run=${WANDB_RUN_NAME}"
echo "[eval] launching (blocks until eval exits — wrap in tmux to detach)"
echo ""

cd "$REPO_ROOT"
python "$LAUNCHER" train \
  --task camel_terminal_agent \
  --num-nodes "$NUM_NODES" \
  --num-gpus-per-node "$NUM_GPUS_PER_NODE" \
  --model-name "$MODEL_NAME" \
  --hf-checkpoint "$HF_CHECKPOINT" \
  --model-dir "$MODEL_DIR" \
  --seta-env-dataset-dir "$DATASET_DIR" \
  --seta-env-parquet-path "$PARQUET_PATH" \
  --seta-env-eval-n-samples "$EVAL_N_SAMPLES" \
  --seta-env-max-response-len "$EVAL_MAX_RESPONSE_LEN" \
  --seta-env-extra-pythonpath "$SETA_ENV_EXTRA_PYTHONPATH" \
  --skip-saving \
  --extra-args "$EXTRA_ARGS"
