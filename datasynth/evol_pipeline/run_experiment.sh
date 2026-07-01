#!/bin/bash
# Run the 4-config experiment plan sequentially.
# Runs 1 & 2 are independent (on original tasks).
# Runs 3 & 4 chain from Runs 1 & 2 respectively.
#
# Usage:
#   bash run_experiment.sh           # run all 4
#   bash run_experiment.sh --dry-run # preview only

set -euo pipefail
cd "$(dirname "$0")"

EXTRA_ARGS="${*}"

echo "============================================================"
echo "Run 1/4: INCREASE_DIFFICULTY on 5 tasks"
echo "============================================================"
python run_evol_orchestrator.py configs/test_increase_difficulty.yaml $EXTRA_ARGS

echo ""
echo "============================================================"
echo "Run 2/4: CHANGE_CONTEXT on 5 tasks"
echo "============================================================"
python run_evol_orchestrator.py configs/test_change_context.yaml $EXTRA_ARGS

echo ""
echo "============================================================"
echo "Run 3/4: INCREASE_DIFFICULTY turn 2 (chain from Run 1)"
echo "============================================================"
python run_evol_orchestrator.py configs/test_increase_difficulty_turn2.yaml $EXTRA_ARGS

echo ""
echo "============================================================"
echo "Run 4/4: INCREASE_DIFFICULTY after CHANGE_CONTEXT (chain from Run 2)"
echo "============================================================"
python run_evol_orchestrator.py configs/test_increase_after_context.yaml $EXTRA_ARGS

echo ""
echo "============================================================"
echo "All 4 runs complete. Generating summaries..."
echo "============================================================"
python run_evol_orchestrator.py configs/test_increase_difficulty.yaml --generate-summary
python run_evol_orchestrator.py configs/test_change_context.yaml --generate-summary
python run_evol_orchestrator.py configs/test_increase_difficulty_turn2.yaml --generate-summary
python run_evol_orchestrator.py configs/test_increase_after_context.yaml --generate-summary

echo "Done."
