#!/usr/bin/env bash
# Download the GLM-5.2 checkpoints the scripts in this folder use, at pinned revisions.
#
#   MODEL_DIR     (MODEL_ROOT/GLM-5.2)      BF16 weights: the Megatron-Bridge training model (LoRA base)
#   FP8_MODEL_DIR (MODEL_ROOT/GLM-5.2-FP8)  FP8 weights: what the SGLang rollout engines serve
#
# Run once, on a machine that sees MODEL_ROOT at the same path as every training
# node (a shared filesystem). Re-running resumes a partial download.
#
# Environment:
#   MODEL_ROOT     default /root/models
#   MODEL_DIR      default ${MODEL_ROOT}/GLM-5.2
#   FP8_MODEL_DIR  default ${MODEL_ROOT}/GLM-5.2-FP8
#   HF_TOKEN       optional; both repositories are public
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../../common/launcher.sh"
load_env_file "${SCRIPT_DIR}"   # MODEL_ROOT, HF_TOKEN, ... from .env, if present

MODEL_ROOT=${MODEL_ROOT:-/root/models}
BF16_REPO=zai-org/GLM-5.2
BF16_REVISION=b4734de4facf877f85769a911abafc5283eab3d9
BF16_BYTES=1506687604850
FP8_REPO=zai-org/GLM-5.2-FP8
FP8_REVISION=ba978f7d347eaf65d22f1a86833408afdb953541
FP8_BYTES=755663627013
BF16_DIR=${MODEL_DIR:-${MODEL_ROOT}/GLM-5.2}
FP8_DIR=${FP8_MODEL_DIR:-${MODEL_ROOT}/GLM-5.2-FP8}

if command -v hf >/dev/null 2>&1; then
  HF_DOWNLOAD=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_DOWNLOAD=(huggingface-cli download)
else
  echo "need the Hugging Face CLI (pip install -U huggingface_hub)" >&2
  exit 2
fi

mkdir -p "$(dirname -- "${BF16_DIR}")" "$(dirname -- "${FP8_DIR}")"
# About 2.3 TB in total. Check the space still needed before starting, so a
# full disk does not leave a half-written checkpoint behind.
present() { du -sb "$1" 2>/dev/null | awk '{print $1}' || true; }
bf16_present=$(present "${BF16_DIR}"); fp8_present=$(present "${FP8_DIR}")
bf16_left=$((BF16_BYTES > ${bf16_present:-0} ? BF16_BYTES - ${bf16_present:-0} : 0))
fp8_left=$((FP8_BYTES > ${fp8_present:-0} ? FP8_BYTES - ${fp8_present:-0} : 0))
free=$(df -B1 --output=avail "$(dirname -- "${BF16_DIR}")" | awk 'NR == 2 {print $1}')
((free >= bf16_left + fp8_left)) || {
  echo "$(dirname -- "${BF16_DIR}") needs $((bf16_left + fp8_left)) free bytes to finish both checkpoints; found ${free}" >&2
  exit 2
}

echo "[prepare_model] ${BF16_REPO}@${BF16_REVISION} -> ${BF16_DIR}"
"${HF_DOWNLOAD[@]}" "${BF16_REPO}" --revision "${BF16_REVISION}" --local-dir "${BF16_DIR}"
echo "[prepare_model] ${FP8_REPO}@${FP8_REVISION} -> ${FP8_DIR}"
"${HF_DOWNLOAD[@]}" "${FP8_REPO}" --revision "${FP8_REVISION}" --local-dir "${FP8_DIR}"

test -s "${BF16_DIR}/model.safetensors.index.json"
test -s "${FP8_DIR}/model.safetensors.index.json"
echo "[prepare_model] GLM-5.2 BF16 and FP8 checkpoints are complete: ${BF16_DIR}, ${FP8_DIR}"
