#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/activate_env.sh"

cd "$REPO_ROOT"

python scripts/journal/build_crc_followup_experiments.py \
    --stage1-output-dir "journal_results/shenzhen_multihorizon/warmstart_raw" \
    --stage2-output-dir "journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw" \
    --stage4-output-dir "journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw" \
    --reinforcement-dir "journal_results/shenzhen_multihorizon/reinforcement" \
    --paper-assets-dir "journal_results/shenzhen_multihorizon/paper_assets" \
    --crc-v1-dir "journal_results/shenzhen_multihorizon/crc_risk_control" \
    --output-dir "journal_results/shenzhen_multihorizon/crc_risk_control_followup" \
    --alphas-violation "0.05,0.10,0.20" \
    --h1-alphas "0.03,0.04,0.05,0.06,0.08,0.10" \
    --cost-ratios "3:1,5:1,9:1,19:1" \
    --bootstrap-samples 300 \
    --bootstrap-seed 20260521 \
    --machine Lenovo
