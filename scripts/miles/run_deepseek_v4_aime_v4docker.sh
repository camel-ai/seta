#!/bin/bash

MODEL_NAME="DeepSeek-V4-Flash-FP8"
# /data is the GPFS mount visible on every container. Use this canonical
# path everywhere (avoid /root/data — not every container in the pool is
# guaranteed to have the /root/data → /data convenience symlink, and
# SGLang resolves the model path on every node independently).
MODEL_DIR="/data/models"
HF_CHECKPOINT="${MODEL_DIR}/${MODEL_NAME}"
TORCH_DIST_DIR="${MODEL_DIR}/${MODEL_NAME}_torch_dist"

REPO_ROOT="/data/terminal_agent"
LAUNCHER="${REPO_ROOT}/scripts/miles/run_deepseek_v4_aime_v4docker.py"
GENERATE_FN_DIR="${REPO_ROOT}/scripts/miles"
MEGATRON_PATH="/root/Megatron-LM"

DATA_DIR="/root/data/datasets/"
NUM_NODES=8
NUM_GPUS_PER_NODE=8

# Ray HEAD coordinates. This script runs ON the head and connects to the
# EXTERNALLY-bootstrapped Ray cluster (head + 7 workers joined by your node
# manager). HEAD_IP must be the real head address, not 127.0.0.1 — workers use
# it as MASTER_ADDR for torch distributed.
HEAD_IP="${HEAD_IP:-10.220.51.32}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

WANDB_ORG="${WANDB_ORG:-zhichenzeng_zzz-org}"
WANDB_ENTITY="${WANDB_ENTITY:-eigent_radixark_training}"
WANDB_PROJECT="${WANDB_PROJECT:-v4flash-aime}"
WANDB_GROUP="${WANDB_GROUP:-aime-miles-main}"
RUN_STAMP="$(date -u +%Y%m%d-%H%M%S)"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${WANDB_GROUP}-${RUN_STAMP}}"
RUN_ROOT="${RUN_ROOT:-/data/training_runs/${WANDB_RUN_NAME}}"
WANDB_DIR="${WANDB_DIR:-${RUN_ROOT}/wandb}"

export WANDB_ENTITY="${WANDB_ENTITY}"
export WANDB_DIR="${WANDB_DIR}"
# Hint to wandb UI which org owns the entity (purely informational; entity is
# the authoritative routing key).
export WANDB_TAGS="${WANDB_TAGS:-org=${WANDB_ORG},task=dapo_aime,dataset=${DATASET_NAME:-}}"


# ─── External Ray (matches scripts/miles/train_v4_flash_milesrouter_r3.sh) ────
# Use the EXISTING, externally-bootstrapped Ray cluster instead of letting the
# launcher tear it down and start a fresh single-node head. execute_train then
# skips `ray stop --force` / `pkill -9 ray` / `ray start --head` and submits the
# job to the running cluster. All three vars are required:
#   MILES_SCRIPT_EXTERNAL_RAY=1  -> skip ray stop/start in execute_train
#   MASTER_ADDR=<head ip>        -> torch-distributed master reachable by workers
#   RAY_ADDRESS=<head dashboard> -> `ray job submit` targets the existing head
export MILES_SCRIPT_EXTERNAL_RAY="${MILES_SCRIPT_EXTERNAL_RAY:-1}"
export MASTER_ADDR="${MASTER_ADDR:-${HEAD_IP}}"

# Inter-node NCCL/Gloo must bootstrap over bond0 on this cluster (matches the
# other train_v4_flash_*/relaunch_* scripts). IB handles the actual transport,
# but without pinning the socket interface NCCL auto-detects a non-routable one
# and dies with "unhandled system error" during distributed init. execute_train
# forwards these to the Ray workers only if they are present in os.environ.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond0}"


# Now point `ray job submit` (inside execute_train) at the existing head.
export RAY_ADDRESS="${RAY_ADDRESS:-http://${HEAD_IP}:${RAY_DASHBOARD_PORT}}"

cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT" "$RUN_ROOT/checkpoints" "$RUN_ROOT/dump_details" "$WANDB_DIR"

# The launcher only *submits* a detached Ray job and then returns, so its own
# stdout (LAUNCH_LOG) stops at "Job submitted". The actual training output lives
# in the Ray job-driver log. We grab the submitted job id from LAUNCH_LOG and
# stream the real logs into RAY_JOB_LOG so the run dir holds everything.
LAUNCH_LOG="${RUN_ROOT}/launch.log"
RAY_JOB_LOG="${RUN_ROOT}/ray_job.log"
RAY_ADDRESS_HTTP="${RAY_ADDRESS:-http://${HEAD_IP}:${RAY_DASHBOARD_PORT}}"

python -u "$LAUNCHER" train \
  --mode debug_minimal \
  --num-nodes "$NUM_NODES" \
  --model-name "$MODEL_NAME" \
  --hf-checkpoint "$HF_CHECKPOINT" \
  --model-dir "$MODEL_DIR" \
  --data-dir "$DATA_DIR" \
  --save-dir "$RUN_ROOT" \
  --optimizer-offload \
  --fp8-training \
  --enable-mtp &> "${LAUNCH_LOG}" &
LAUNCH_PID=$!

# Wait (up to ~10 min) for the Ray job id to appear, then follow its logs.
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
  # The complete training log is the Ray job-driver file on disk. `ray job logs
  # --follow` block-buffers when redirected to a file, so instead symlink the
  # run-dir log straight at the live driver file (unbuffered, complete history).
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
