#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/activate_env.sh"

cd "$REPO_ROOT"

python scripts/journal/evaluate_stage4_decision.py \
    --stage1-output-dir "journal_results/shenzhen_multihorizon/warmstart_raw" \
    --stage2-output-dir "journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw" \
    --output-dir "journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw" \
    --use-cuda True \
    --machine Lenovo
