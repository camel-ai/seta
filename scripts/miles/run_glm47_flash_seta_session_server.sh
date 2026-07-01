#!/bin/bash
# ============================================================================
# GLM-4.7-Flash SETA trainer via the miles SESSION SERVER (new docker).
# SGLang parses GLM tool-calls/reasoning engine-side (--sglang-tool-call-parser glm47,
# --sglang-reasoning-parser glm45) and the miles session server captures TITO
# (--use-session-server --tito-model glm47) — replacing our custom sglang_glm47_flash
# backend (which had a parallel-tool empty-observation TITO bug). Agent runs in our
# seta_env env_service via seta_agent_function.run + agentic_tool_call.generate.
# 8 nodes = 4 serve + 4 train (disaggregated), fully-async. SAME dataset + RL config.
# Uses LAUNCHER=run_glm47_flash_seta_session_server.py and
# CONFIG=seta_env_config_session_server_glm47.yaml (max_iteration 60, parallel tools off).
# ============================================================================

MODEL_NAME="GLM-4.7-Flash"
MODEL_DIR="/data/models"
HF_CHECKPOINT="${MODEL_DIR}/${MODEL_NAME}"
TORCH_DIST_DIR="${MODEL_DIR}/${MODEL_NAME}_torch_dist"

REPO_ROOT="/data/terminal_agent"
LAUNCHER="${REPO_ROOT}/scripts/miles/run_glm47_flash_seta_session_server.py"
GENERATE_FN_DIR="${REPO_ROOT}/scripts/miles"
MEGATRON_PATH="/root/Megatron-LM"

DATA_DIR="/root/data/datasets/"
NUM_NODES="${NUM_NODES:-8}"
NUM_GPUS_PER_NODE=8

HEAD_IP="${HEAD_IP:-10.220.51.32}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

WANDB_ORG="${WANDB_ORG:-zhichenzeng_zzz-org}"
WANDB_ENTITY="${WANDB_ENTITY:-eigent_radixark_training}"
WANDB_PROJECT="${WANDB_PROJECT:-glm47flash-terminal}"
WANDB_GROUP="${WANDB_GROUP:-glm47-terminal-session}"
RUN_STAMP="$(date -u +%Y%m%d-%H%M%S)"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${WANDB_GROUP}-${RUN_STAMP}}"
RUN_ROOT="${RUN_ROOT:-/data/training_runs/${WANDB_RUN_NAME}}"
WANDB_DIR="${WANDB_DIR:-${RUN_ROOT}/wandb}"
ENV_SERVICE_LOG_DIR="${RUN_ROOT}/env_service"
HARBOR_ROOT="${HARBOR_ROOT:-${RUN_ROOT}}"

export WANDB_ENTITY="${WANDB_ENTITY}"
export WANDB_DIR="${WANDB_DIR}"
export WANDB_TAGS="${WANDB_TAGS:-org=${WANDB_ORG},task=camel_terminal_agent,arch=session_server}"
# wandb auth: source key from ~/.bashrc into the (non-interactive) launcher env
eval "$(bash -ic 'v=WANDB_API_KEY; val="${!v-}"; [ -n "$val" ] && printf "export %s=%q\n" "$v" "$val"' 2>/dev/null)"
# camel rollout env (seta_agent_function requires these; launcher forwards them to ray workers)
export CAMEL_DATASET_NAME="${CAMEL_DATASET_NAME:-tbench-tasks-migrated}"
export CAMEL_TRIAL_NAME="${CAMEL_TRIAL_NAME:-$WANDB_RUN_NAME}"
export CAMEL_ENV_SERVICE_URL="${CAMEL_ENV_SERVICE_URL:-http://${HEAD_IP}:8002}"
# [session-server] continuous-worker groups-in-flight (each group = 16 samples).
export ROLLOUT_CONCURRENCY="${ROLLOUT_CONCURRENCY:-30}"
# [session-server] TITOChatModel (OpenAI-compat) client timeout. Defaults to 180s, which
# is too short for GLM generations up to max_tokens=8192. Raise to 900s (well under
# agent_astep=2400s so the agent still does many turns).
export MODEL_TIMEOUT="${MODEL_TIMEOUT:-900}"

# ─── External Ray (use the existing bootstrapped cluster) ────────────────────
export MILES_SCRIPT_EXTERNAL_RAY="${MILES_SCRIPT_EXTERNAL_RAY:-1}"
export MASTER_ADDR="${MASTER_ADDR:-${HEAD_IP}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond0}"
export RAY_ADDRESS="${RAY_ADDRESS:-http://${HEAD_IP}:${RAY_DASHBOARD_PORT}}"

cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT" "$ENV_SERVICE_LOG_DIR" "$RUN_ROOT/checkpoints" "$RUN_ROOT/dump_details" "$WANDB_DIR" "$RUN_ROOT/trials" "$RUN_ROOT/_builds"

# ─── Restart the seta_env env_service (required for the agent rollout) ────────
# Daytona creds are sourced from ~/.bashrc inside env_service.sh.
ENV_SERVICE="${REPO_ROOT}/scripts/miles/core/env_service.sh"
bash "$ENV_SERVICE" stop || true
CONFIG="${REPO_ROOT}/scripts/miles/core/configs/seta_env_config_session_server_glm47.yaml" \
LOG_DIR="$ENV_SERVICE_LOG_DIR" HARBOR_ROOT="$HARBOR_ROOT" \
MAX_SLOTS=400 STEP_TIMEOUT_SECONDS=5000 STEP_USE_SUBPROCESS=1 \
BUILD_CONCURRENCY=24 INFLIGHT_LEAD=64 HARBOR_DAYTONA_MAX_CREATES=24 \
DAYTONA_DECLARATIVE=1 DAYTONA_SNAPSHOT_EVICT_AGE_HOURS=3 \
  bash "$ENV_SERVICE" start
bash "$ENV_SERVICE" wait

LAUNCH_LOG="${RUN_ROOT}/launch.log"
RAY_JOB_LOG="${RUN_ROOT}/ray_job.log"
RAY_ADDRESS_HTTP="${RAY_ADDRESS:-http://${HEAD_IP}:${RAY_DASHBOARD_PORT}}"

python -u "$LAUNCHER" train \
  --mode debug_minimal \
  --num-nodes "$NUM_NODES" \
  --model-name "$MODEL_NAME" \
  --model-dir "$MODEL_DIR" \
  --data-dir "$DATA_DIR" \
  --save-dir "$RUN_ROOT" &> "${LAUNCH_LOG}" &
LAUNCH_PID=$!

echo "[info] run dir: ${RUN_ROOT}"
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
    echo "[info] complete training log -> ${RAY_JOB_LOG} -> ${DRIVER_LOG}"
  else
    echo "[warn] driver log not found yet; falling back to buffered follow"
    ray job logs "${RAY_JOB_ID}" --follow --address="${RAY_ADDRESS_HTTP}" &> "${RAY_JOB_LOG}" &
  fi
else
  echo "[warn] could not find a Ray job id in ${LAUNCH_LOG}; check it manually"
fi

wait "${LAUNCH_PID}"
