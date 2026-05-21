#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/activate_env.sh"

cd "$REPO_ROOT"

python scripts/journal/build_preference_mapping_experiments.py \
    --stage1-output-dir "journal_results/shenzhen_multihorizon/warmstart_raw" \
    --stage2-output-dir "journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw" \
    --stage4-output-dir "journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw" \
    --reinforcement-dir "journal_results/shenzhen_multihorizon/reinforcement" \
    --paper-assets-dir "journal_results/shenzhen_multihorizon/paper_assets" \
    --crc-v1-dir "journal_results/shenzhen_multihorizon/crc_risk_control" \
    --crc-followup-dir "journal_results/shenzhen_multihorizon/crc_risk_control_followup" \
    --output-dir "journal_results/shenzhen_multihorizon/preference_mapping" \
    --cost-ratios "1:1,3:1,5:1,9:1,19:1" \
    --alphas-violation "0.05,0.10,0.20" \
    --h1-alphas "0.03,0.04,0.05,0.06,0.08,0.10" \
    --bootstrap-samples 300 \
    --bootstrap-seed 20260601 \
    --enable-risk-screened true \
    --enable-aggregate-procurement false \
    --machine Lenovo
