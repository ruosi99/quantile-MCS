#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/activate_env.sh"

cd "$REPO_ROOT"

python scripts/journal/build_reinforcement_experiments.py \
    --stage1-output-dir "journal_results/shenzhen_multihorizon/warmstart_raw" \
    --stage2-output-dir "journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw" \
    --stage4-output-dir "journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw" \
    --data-dir "data/datasets/ST_EVCDP_v2_canonical" \
    --output-dir "journal_results/shenzhen_multihorizon/reinforcement" \
    --deltas 0.1 \
    --primary-delta 0.1 \
    --cost-ratios "1:1,3:1,5:1,9:1,19:1" \
    --station-groups 4 \
    --bootstrap-samples 300 \
    --bootstrap-seed 20260506 \
    --machine Lenovo
