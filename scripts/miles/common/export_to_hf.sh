#!/usr/bin/env bash
# Export one trained torch_dist iteration to a Hugging Face checkpoint that SGLang
# (or transformers) can load.
#
#   MODEL_DIR=/root/models/<Model> bash scripts/miles/common/export_to_hf.sh <ITER_DIR> <OUT_DIR>
#
#   MODEL_DIR  the base HF checkpoint the run started from (config, tokenizer, untrained weights)
#   ITER_DIR   e.g. ${RUNS_ROOT}/${RUN_NAME}/checkpoints/iter_0000119 (the actor; a PPO run's
#              critic lives under checkpoints/critic and is not exported)
#   OUT_DIR    new directory; must not exist
#   EXPORT_PREFIXES  optional, space-separated: only copy untrained tensors with these name
#              prefixes (e.g. "model.visual. mtp." for Qwen3.5/3.8 vision-language checkpoints)
#
# Run inside the Miles container. Miles' converter writes only what Megatron trained;
# complete_hf_export.py then copies the base checkpoint's remaining tensors (e.g. a
# vision tower or an MTP head) and checks the export has exactly the base tensor
# names and shapes.
set -euo pipefail

[[ $# -eq 2 ]] || { echo "usage: $0 ITER_DIR OUT_DIR" >&2; exit 2; }
ITER_DIR=$1
OUT_DIR=$2
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${MODEL_DIR:?set MODEL_DIR to the base HF checkpoint the run started from}"
MILES_DIR=${MILES_DIR:-/root/miles}
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}
# Padded vocabulary of the base model (top-level or text_config.vocab_size).
VOCAB_SIZE=${VOCAB_SIZE:-$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c.get("vocab_size") or c["text_config"]["vocab_size"])' "${MODEL_DIR}/config.json")}

[[ -f "${ITER_DIR}/common.pt" && -f "${ITER_DIR}/.metadata" ]] || {
  echo "not a torch_dist checkpoint iteration: ${ITER_DIR}" >&2
  exit 2
}
[[ -f "${MODEL_DIR}/model.safetensors.index.json" ]] || {
  echo "missing base HF checkpoint: ${MODEL_DIR}" >&2
  exit 2
}
[[ ! -e "${OUT_DIR}" ]] || { echo "output exists: ${OUT_DIR}" >&2; exit 2; }
ITER_DIR=$(cd -- "${ITER_DIR}" && pwd)
mkdir -p "$(dirname -- "${OUT_DIR}")"
OUT_DIR=$(cd -- "$(dirname -- "${OUT_DIR}")" && pwd)/$(basename -- "${OUT_DIR}")
# Roughly the size of the base checkpoint, plus headroom.
need=$(( $(du -sb "${MODEL_DIR}" | awk '{print $1}') * 5 / 4 ))
avail=$(df -B1 --output=avail "$(dirname -- "${OUT_DIR}")" | awk 'NR == 2 {print $1}')
(( avail >= need )) || { echo "need ~${need} free bytes for the export, found ${avail}" >&2; exit 2; }

echo "[export] ${ITER_DIR} -> ${OUT_DIR} (base ${MODEL_DIR})"
cd "${MILES_DIR}"
# --origin-hf-dir selects the architecture's converter (the same one the
# run's weight sync used) and copies config and tokenizer.
PYTHONPATH="${MEGATRON_PATH}:${MILES_DIR}" python tools/convert_torch_dist_to_hf.py \
  --input-dir "${ITER_DIR}" \
  --output-dir "${OUT_DIR}" \
  --origin-hf-dir "${MODEL_DIR}" \
  --vocab-size "${VOCAB_SIZE}"

PREFIX_ARGS=()
for prefix in ${EXPORT_PREFIXES:-}; do PREFIX_ARGS+=(--prefix "${prefix}"); done
python "${SCRIPT_DIR}/complete_hf_export.py" --export-dir "${OUT_DIR}" --base-hf-dir "${MODEL_DIR}" \
  ${PREFIX_ARGS[@]+"${PREFIX_ARGS[@]}"}
printf '%s\n' "${ITER_DIR}" > "${OUT_DIR}/SOURCE_TORCH_DIST_ITER"
echo "[export] done: ${OUT_DIR}"
