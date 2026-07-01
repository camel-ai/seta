#!/bin/bash
SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR"
PYTHONPATH="$SCRIPT_DIR/../.." \
    python -m areal.launcher.local \
    "$SCRIPT_DIR/eval.py" --config "$SCRIPT_DIR/configs/config_eval.yaml" \
    "$@" &> "$SCRIPT_DIR/log"
