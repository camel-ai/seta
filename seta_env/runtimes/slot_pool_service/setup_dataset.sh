#!/usr/bin/env bash
# setup_dataset.sh — activate a dataset across all nodes via the scheduler.
#
# Usage:
#   bash setup_dataset.sh <dataset_name>
#   bash setup_dataset.sh <dataset_name> --scheduler http://127.0.0.1:8000
#
# The scheduler fans out to every node in nodes.yaml in parallel.
# Each node git-clones the dataset on first use and caches it locally.
# Subsequent calls with the same dataset name are instant (already cached).
#
# Available dataset names are defined in datasets.yaml.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <dataset_name> [--scheduler URL]"
    echo ""
    echo "Example:"
    echo "  bash setup_dataset.sh seta-env-harbor"
    echo "  bash setup_dataset.sh terminal-bench-2.0 --scheduler http://127.0.0.1:8000"
    exit 1
fi

DATASET_NAME="$1"; shift
SCHEDULER_URL="http://127.0.0.1:8000"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. All datasets are hosted on HuggingFace and require authentication."
    echo "       export HF_TOKEN=<your_token> and re-run."
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scheduler) SCHEDULER_URL="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== Setting up dataset: $DATASET_NAME ==="
echo "    Scheduler: $SCHEDULER_URL"
echo "    Downloading on all nodes in parallel (first use may take ~1 min)..."

# Run curl in background so we can show a progress spinner.
RESPONSE_FILE=$(mktemp)
curl -sf -X POST "$SCHEDULER_URL/setup_dataset" \
    -H "Content-Type: application/json" \
    -d "{\"dataset_name\": \"$DATASET_NAME\", \"hf_token\": \"$HF_TOKEN\"}" \
    -o "$RESPONSE_FILE" &
CURL_PID=$!

# Spinner while waiting.
echo -n "    "
while kill -0 $CURL_PID 2>/dev/null; do
    for ch in '|' '/' '-' '\'; do
        printf "\r    %s" "$ch"
        sleep 0.3
    done
done
printf "\r    \r"  # clear spinner line

if ! wait $CURL_PID; then
    rm -f "$RESPONSE_FILE"
    echo "ERROR: Could not reach scheduler at $SCHEDULER_URL"
    echo "       Is the scheduler running? (bash start.sh)"
    exit 1
fi

RESPONSE=$(cat "$RESPONSE_FILE")
rm -f "$RESPONSE_FILE"

echo "$RESPONSE" | python3 -c "
import json, sys
data = json.load(sys.stdin)
results = data.get('results', [])
for r in results:
    status = r['status']
    node   = r['node']
    body   = r['body']
    ok = status == 200 and (isinstance(body, dict) and body.get('success'))
    mark = 'OK' if ok else 'FAIL'
    already = isinstance(body, dict) and body.get('already_present')
    note = ' (already cached)' if already else ' (downloaded)'
    print(f'  [{mark}] {node}{note if ok else \"\"}')
    if not ok:
        detail = body if (body and body != '{}') else '(no detail — check node manager logs)'
        print(f'         status={status}')
        print(f'         error={detail}')

print()
if data.get('success'):
    print('All nodes ready.')
else:
    failed = data.get('failed_nodes', [])
    print(f'FAILED on {len(failed)} node(s): {failed}')
    print()
    print('To check logs on a failed node:')
    for r in results:
        if r['status'] != 200 or not (isinstance(r['body'], dict) and r['body'].get('success')):
            host = r['node'].split('//')[1].split(':')[0]
            print(f'  ssh root@{host} journalctl -u node-manager -n 50 --no-pager')
    sys.exit(1)
"
