#!/usr/bin/env bash
# Download Qwen3.8-27B at the pinned revision and convert it to a Megatron
# torch_dist checkpoint.
#
# Run inside the Miles container (the same image the recipes use), on a node
# that sees MODEL_ROOT, with the node's GPUs free. Idempotent: an existing
# download is resumed/verified by `hf download`, and a finished conversion
# (latest_checkpointed_iteration.txt present) is skipped.
#
#   MODEL_ROOT=/path/to/models bash scripts/miles/examples/qwen3_8_27b_grpo/prepare_model.sh
#
# Produces ${MODEL_DIR} (~56 GB bf16) and ${TORCH_DIST_DIR} (~51 GB).
set -euo pipefail

MODEL_ROOT=${MODEL_ROOT:-/root/models}
MODEL_DIR=${MODEL_DIR:-${MODEL_ROOT}/Qwen3.8-27B}
TORCH_DIST_DIR=${TORCH_DIST_DIR:-${MODEL_ROOT}/Qwen3.8-27B_torch_dist}
MILES_DIR=${MILES_DIR:-/root/miles}
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}

# Qwen3.8-27B is a vision-language checkpoint (Qwen3_5ForConditionalGeneration);
# the recipes train only its text path. Apache-2.0, not gated.
MODEL_REPO=Qwen/Qwen3.8-27B
MODEL_REVISION=${MODEL_REVISION:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}
# The pinned Miles has no qwen3.8-27B entry; qwen3.5-27B is the same
# architecture (same spec function; 64 layers, hidden 5120, 24 heads,
# 4 query groups, ffn 17408, vocab 248320, kv-channels 256).
MODEL_TYPE=qwen3.5-27B

free_bytes() { df -B1 --output=avail "$1" | awk 'NR == 2 {print $1}'; }
mkdir -p "$(dirname -- "${MODEL_DIR}")" "$(dirname -- "${TORCH_DIST_DIR}")"

if [[ ! -s "${MODEL_DIR}/model.safetensors.index.json" ]]; then
  need=60000000000
  (( $(free_bytes "$(dirname -- "${MODEL_DIR}")") >= need )) || {
    echo "need ~${need} free bytes under $(dirname -- "${MODEL_DIR}") for the download" >&2
    exit 2
  }
fi
echo "[prepare_model] ${MODEL_REPO}@${MODEL_REVISION} -> ${MODEL_DIR}"
hf download "${MODEL_REPO}" --revision "${MODEL_REVISION}" --local-dir "${MODEL_DIR}"
test -s "${MODEL_DIR}/model.safetensors.index.json"

if [[ -s "${TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" ]]; then
  echo "[prepare_model] ${TORCH_DIST_DIR} already converted"
  exit 0
fi
[[ ! -e "${TORCH_DIST_DIR}" ]] || {
  echo "${TORCH_DIST_DIR} exists but has no release tracker (interrupted conversion?); remove it first" >&2
  exit 2
}
need=75000000000
(( $(free_bytes "$(dirname -- "${TORCH_DIST_DIR}")") >= need )) || {
  echo "need ~${need} free bytes under $(dirname -- "${TORCH_DIST_DIR}") for the conversion" >&2
  exit 2
}

cd "${MILES_DIR}"
MODEL_ARGS=$(PYTHONPATH="${MILES_DIR}" python -c \
  "from miles.utils.external_utils.model_args_utils import load_model_args; print(load_model_args('${MODEL_TYPE}'))")
[[ -n "${MODEL_ARGS}" ]] || { echo "empty model args for ${MODEL_TYPE}" >&2; exit 2; }
echo "[prepare_model] converting ${MODEL_DIR} -> ${TORCH_DIST_DIR} as ${MODEL_TYPE}"
# MODEL_ARGS is a pre-tokenised argument list.
# shellcheck disable=SC2086
PYTHONPATH="${MEGATRON_PATH}:${MILES_DIR}" python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS} \
  --hf-checkpoint "${MODEL_DIR}" \
  --save "${TORCH_DIST_DIR}"

test -s "${TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" || {
  echo "conversion did not write latest_checkpointed_iteration.txt" >&2
  exit 1
}
echo "[prepare_model] done: MODEL_DIR=${MODEL_DIR} TORCH_DIST_DIR=${TORCH_DIST_DIR}"
