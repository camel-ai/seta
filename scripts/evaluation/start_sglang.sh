#!/usr/bin/env bash
# Start SGLang server with Qwen3-8B
#
# Usage: bash scripts/start_sglang.sh [model_path_or_id]
#
# Defaults:
#   model:          Qwen/Qwen3-8B
#   tensor-parallel: 2 GPUs
#   context length: 32768
#   tool-call-parser: qwen25

set -euo pipefail

PYTHON="${PYTHON:-python}"
MODEL="${1:-Qwen/Qwen3-8B}"
HOST="${SGLANG_HOST:-0.0.0.0}"
PORT="${SGLANG_PORT:-30000}"
TP="${SGLANG_TP:-1}"
CONTEXT_LEN="${SGLANG_CONTEXT_LEN:-32768}"

echo "Starting SGLang server"
echo "  model:           $MODEL"
echo "  host:port:       $HOST:$PORT"
echo "  tensor-parallel: $TP"
echo "  context length:  $CONTEXT_LEN"
echo "  tool-call-parser: qwen25"
echo ""

exec $PYTHON -m sglang.launch_server \
    --model-path "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --tp "$TP" \
    --context-length "$CONTEXT_LEN" \
    --tool-call-parser qwen25
