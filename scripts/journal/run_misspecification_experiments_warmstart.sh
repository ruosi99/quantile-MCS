#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/activate_env.sh"

cd "$REPO_ROOT"

python scripts/journal/build_misspecification_experiments.py \
    --stage1-output-dir "journal_results/shenzhen_multihorizon/warmstart_raw" \
    --stage4-output-dir "journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw" \
    --reinforcement-dir "journal_results/shenzhen_multihorizon/reinforcement" \
    --data-dir "data/datasets/ST_EVCDP_v2_canonical" \
    --output-dir "journal_results/shenzhen_multihorizon/misspecification" \
    --phases "${MISSPECIFICATION_PHASES:-1}" \
    --include-interpolated-ratios false \
    --bootstrap-samples 300 \
    --bootstrap-seed 20260507 \
    --machine Lenovo
