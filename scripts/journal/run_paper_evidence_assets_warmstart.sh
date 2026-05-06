#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/activate_env.sh"

cd "$REPO_ROOT"

python scripts/journal/build_paper_evidence_assets.py \
    --stage1-output-dir "journal_results/shenzhen_multihorizon/warmstart_raw" \
    --stage2-output-dir "journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw" \
    --stage2-diagnostics-dir "journal_results/shenzhen_multihorizon/stage2_diagnostics/warmstart_raw" \
    --stage4-output-dir "journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw" \
    --output-dir "journal_results/shenzhen_multihorizon/paper_assets" \
    --deltas 0.1 \
    --primary-delta 0.1 \
    --bootstrap-samples 300 \
    --bootstrap-seed 20260506 \
    --machine Lenovo
