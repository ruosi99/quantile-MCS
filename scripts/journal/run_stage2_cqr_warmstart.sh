#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/activate_env.sh"

cd "$REPO_ROOT"

python scripts/journal/calibrate_multihorizon_cqr.py \
    --stage1-output-dir "journal_results/shenzhen_multihorizon/warmstart_raw" \
    --output-dir "journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw" \
    --use-cuda True \
    --deltas 0.1,0.2,0.4
