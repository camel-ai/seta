#!/bin/bash
# ============================================================================
# HELD-OUT EVAL — DeepSeek V4 docker (rolled-back image).
# Evaluates the BASE model on the held-out seta-env set
#   /data/terminal_agent/dataset/seta-env-eval-heldout.parquet  (2073 tasks)
# via run_deepseek_v4.py (SYNC train.py eval-only branch: --debug-rollout-only
# --num-rollout 0 --eval-interval 1). The async 1stepoff launcher CANNOT do this
# (train_async.py has no eval-only branch), so eval goes through the canonical
# sync launcher, which emits --use-miles-router (docker-compatible) and the
# COLOCATE path (rollout_num_nodes=0) -> all 8 nodes / 64 GPUs serve SGLang.
#   8 samples/task, max_iteration 200 (eval yaml), MAX_SLOTS 200.
# Reuses the seta-env env_service wiring from the 1stepoff launcher.
# ============================================================================
set -uo pipefail

MODEL_NAME="DeepSeek-V4-Flash-FP8"
MODEL_DIR="/data/models"
HF_CHECKPOINT="${MODEL_DIR}/${MODEL_NAME}"

REPO_ROOT="/data/terminal_agent"
LAUNCHER="${REPO_ROOT}/scripts/miles/run_deepseek_v4.py"   # canonical SYNC launcher (eval-only branch)
GENERATE_FN_DIR="${REPO_ROOT}/scripts/miles"
MEGATRON_PATH="/root/Megatron-LM"

# Held-out eval set (built by build_eval_parquet: no-easy minus pass8 trainset).
PARQUET_PATH="${PARQUET_PATH:-/data/terminal_agent/dataset/seta-env-eval-heldout.parquet}"
EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-8}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-8192}"

NUM_NODES="${NUM_NODES:-8}"          # all 8 nodes; colocate -> 64 GPUs serve
NUM_GPUS_PER_NODE=8

HEAD_IP="${HEAD_IP:-10.220.51.32}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

WANDB_ORG="${WANDB_ORG:-zhichenzeng_zzz-org}"
WANDB_ENTITY="${WANDB_ENTITY:-eigent_radixark_training}"
WANDB_PROJECT="${WANDB_PROJECT:-v4flash-terminal-eval}"
WANDB_GROUP="${WANDB_GROUP:-eval-seta-heldout-base}"
RUN_STAMP="$(date -u +%Y%m%d-%H%M%S)"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${WANDB_GROUP}-${RUN_STAMP}}"
RUN_ROOT="${RUN_ROOT:-/data/training_runs/${WANDB_RUN_NAME}}"
WANDB_DIR="${WANDB_DIR:-${RUN_ROOT}/wandb}"
DUMP_DETAILS_DIR="${RUN_ROOT}/dump_details"
ENV_SERVICE_LOG_DIR="${RUN_ROOT}/env_service"
HARBOR_ROOT="${HARBOR_ROOT:-${RUN_ROOT}}"   # harbor writes trials/<name>/ + _builds/ here

SETA_ENV_EXTRA_PYTHONPATH="${REPO_ROOT}:${GENERATE_FN_DIR}"

export WANDB_ENTITY="${WANDB_ENTITY}"
export WANDB_DIR="${WANDB_DIR}"
export WANDB_TAGS="${WANDB_TAGS:-org=${WANDB_ORG},task=camel_terminal_agent,dataset=seta-env-heldout,mode=eval}"
eval "$(bash -ic 'v=WANDB_API_KEY; val="${!v-}"; [ -n "$val" ] && printf "export %s=%q\n" "$v" "$val"' 2>/dev/null)"

# camel rollout env (generate_with_camel requires these; forwarded to ray workers)
export CAMEL_DATASET_NAME="${CAMEL_DATASET_NAME:-seta-env-final-camel-combined}"
export CAMEL_TRIAL_NAME="${CAMEL_TRIAL_NAME:-$WANDB_RUN_NAME}"
export CAMEL_ENV_SERVICE_URL="${CAMEL_ENV_SERVICE_URL:-http://${HEAD_IP}:8002}"

# ─── External Ray (connect to the running cluster; do not tear it down) ───────
export MILES_SCRIPT_EXTERNAL_RAY="${MILES_SCRIPT_EXTERNAL_RAY:-1}"
export MASTER_ADDR="${MASTER_ADDR:-${HEAD_IP}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond0}"
export RAY_ADDRESS="${RAY_ADDRESS:-http://${HEAD_IP}:${RAY_DASHBOARD_PORT}}"

cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT" "$ENV_SERVICE_LOG_DIR" "$DUMP_DETAILS_DIR" "$WANDB_DIR" "$RUN_ROOT/trials" "$RUN_ROOT/_builds"

# ─── Restart seta_env env_service with the EVAL config (max_iteration 200) ────
ENV_SERVICE="${REPO_ROOT}/scripts/miles/core/env_service.sh"
bash "$ENV_SERVICE" stop || true
CONFIG="${REPO_ROOT}/scripts/miles/core/configs/seta_env_config_eval_v4docker.yaml" \
LOG_DIR="$ENV_SERVICE_LOG_DIR" HARBOR_ROOT="$HARBOR_ROOT" \
MAX_SLOTS=200 STEP_TIMEOUT_SECONDS=5000 STEP_USE_SUBPROCESS=1 \
BUILD_CONCURRENCY=24 INFLIGHT_LEAD=64 HARBOR_DAYTONA_MAX_CREATES=24 \
DAYTONA_DECLARATIVE=1 DAYTONA_SNAPSHOT_EVICT_AGE_HOURS=3 \
  bash "$ENV_SERVICE" start
bash "$ENV_SERVICE" wait

# Eval-only flags (sync train.py eval-only branch). No --load => BASE model.
EXTRA_ARGS="\
--debug-rollout-only \
--num-rollout 0 \
--eval-interval 1 \
--rollout-batch-size 16 \
--dump-details ${DUMP_DETAILS_DIR} \
--use-wandb \
--wandb-project ${WANDB_PROJECT} \
--wandb-entity ${WANDB_ENTITY} \
--wandb-group ${WANDB_GROUP} \
--wandb-exp-name ${WANDB_RUN_NAME} \
--wandb-dir ${WANDB_DIR} \
--disable-wandb-random-suffix"

LAUNCH_LOG="${RUN_ROOT}/launch.log"
RAY_JOB_LOG="${RUN_ROOT}/ray_job.log"
RAY_ADDRESS_HTTP="${RAY_ADDRESS:-http://${HEAD_IP}:${RAY_DASHBOARD_PORT}}"

# NOTE: no --rollout-num-nodes => run_deepseek_v4.py default 0 => --colocate => 64 GPUs.
#       no --load             => eval the BASE model.
export PYTHONPATH="${SETA_ENV_EXTRA_PYTHONPATH}:${MEGATRON_PATH}:${PYTHONPATH:-}"
python -u "$LAUNCHER" train \
  --task camel_terminal_agent \
  --num-nodes "$NUM_NODES" \
  --num-gpus-per-node "$NUM_GPUS_PER_NODE" \
  --model-name "$MODEL_NAME" \
  --hf-checkpoint "$HF_CHECKPOINT" \
  --model-dir "$MODEL_DIR" \
  --seta-env-parquet-path "$PARQUET_PATH" \
  --seta-env-eval-n-samples "$EVAL_N_SAMPLES" \
  --seta-env-max-response-len "$EVAL_MAX_RESPONSE_LEN" \
  --seta-env-extra-pythonpath "$SETA_ENV_EXTRA_PYTHONPATH" \
  --skip-saving \
  --extra-args "$EXTRA_ARGS" &> "${LAUNCH_LOG}" &
LAUNCH_PID=$!

echo "[info] run dir:      ${RUN_ROOT}"
echo "[info] eval parquet: ${PARQUET_PATH}  (n-samples=${EVAL_N_SAMPLES})"
echo "[info] launcher log: ${LAUNCH_LOG}"
echo "[info] waiting for Ray job id ..."
RAY_JOB_ID=""
for _ in $(seq 1 120); do
  RAY_JOB_ID="$(grep -oE 'raysubmit_[A-Za-z0-9]+' "${LAUNCH_LOG}" 2>/dev/null | head -1)"
  [ -n "${RAY_JOB_ID}" ] && break
  kill -0 "${LAUNCH_PID}" 2>/dev/null || break
  sleep 5
done

if [ -n "${RAY_JOB_ID}" ]; then
  echo "[info] Ray job id: ${RAY_JOB_ID}"
  DRIVER_LOG=""
  for _ in $(seq 1 60); do
    DRIVER_LOG="$(readlink -f "/tmp/ray/session_latest/logs/job-driver-${RAY_JOB_ID}.log" 2>/dev/null)"
    [ -n "${DRIVER_LOG}" ] && [ -e "${DRIVER_LOG}" ] && break
    sleep 2
  done
  if [ -n "${DRIVER_LOG}" ] && [ -e "${DRIVER_LOG}" ]; then
    ln -sfn "${DRIVER_LOG}" "${RAY_JOB_LOG}"
    echo "[info] complete eval log -> ${RAY_JOB_LOG} -> ${DRIVER_LOG}"
  else
    echo "[warn] driver log not found yet; falling back to buffered follow"
    ray job logs "${RAY_JOB_ID}" --follow --address="${RAY_ADDRESS_HTTP}" &> "${RAY_JOB_LOG}" &
  fi
else
  echo "[warn] could not find a Ray job id in ${LAUNCH_LOG}; check it manually"
fi

wait "${LAUNCH_PID}"
