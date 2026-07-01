#!/bin/bash
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PYTHONPATH="$SCRIPT_DIR/../.." \
    python -u \
    "$SCRIPT_DIR/eval.py" --config "$SCRIPT_DIR/configs/eval_default.yaml" \
    "$@" &> "$SCRIPT_DIR/log"
