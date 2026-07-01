#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
EXTRA_ARGS="${*}"

echo "============================================================"
echo "Run 1/4: seta-env-harbor + INCREASE_DIFFICULTY"
echo "============================================================"
python run_evol_orchestrator.py configs/sample100_harbor_increase_difficulty.yaml $EXTRA_ARGS

echo ""
echo "============================================================"
echo "Run 2/4: seta-env-harbor + CHANGE_CONTEXT"
echo "============================================================"
python run_evol_orchestrator.py configs/sample100_harbor_change_context.yaml $EXTRA_ARGS

echo ""
echo "============================================================"
echo "Run 3/4: seta-env-v2 + INCREASE_DIFFICULTY"
echo "============================================================"
python run_evol_orchestrator.py configs/sample100_v2_increase_difficulty.yaml $EXTRA_ARGS

echo ""
echo "============================================================"
echo "Run 4/4: seta-env-v2 + CHANGE_CONTEXT"
echo "============================================================"
python run_evol_orchestrator.py configs/sample100_v2_change_context.yaml $EXTRA_ARGS

echo ""
echo "============================================================"
echo "All 4 runs complete. Generating summaries..."
echo "============================================================"
python run_evol_orchestrator.py configs/sample100_harbor_increase_difficulty.yaml --generate-summary
python run_evol_orchestrator.py configs/sample100_harbor_change_context.yaml --generate-summary
python run_evol_orchestrator.py configs/sample100_v2_increase_difficulty.yaml --generate-summary
python run_evol_orchestrator.py configs/sample100_v2_change_context.yaml --generate-summary
echo "Done."
