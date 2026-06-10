#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/activate_env.sh"

cd "$REPO_ROOT"

CITY="${1:-LOA}"
RUN="charged_${CITY}_graph_patchtst_case_volume_seq48_p8s4_ws_g2_weightdecay0_lr1e4_epoch200"
STAGE1_DIR="journal_results/charged_case_study/${CITY}/${RUN}"
STAGE2_DIR="journal_results/charged_case_study/${CITY}/stage2_cqr/${RUN}"
STAGE4_DIR="journal_results/charged_case_study/${CITY}/stage4_decision/${RUN}"

python scripts/journal/calibrate_multihorizon_cqr.py \
  --stage1-output-dir "$STAGE1_DIR" \
  --output-dir "$STAGE2_DIR" \
  --use-cuda True \
  --deltas 0.1,0.2,0.4

python scripts/journal/evaluate_stage4_decision.py \
  --stage1-output-dir "$STAGE1_DIR" \
  --stage2-output-dir "$STAGE2_DIR" \
  --output-dir "$STAGE4_DIR" \
  --use-cuda True \
  --cost-ratios 1:1,3:1,5:1,9:1,19:1 \
  --machine charged_case_study
