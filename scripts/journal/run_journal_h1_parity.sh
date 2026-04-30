#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/activate_env.sh"

cd "$REPO_ROOT"

python scripts/journal/write_stage0_run_specs.py

RESULT_DIR="data/results/ST_EVCDP_v2_canonical/dura_pag_informer_quantile_on_pretrain_results"
CHECKPOINT_NAME="dura_pag_informer_quantile_on_pretrain_1_bs8_completed.pt"
LEGACY_RESULT_DIR="quantile_model/dura_pag_informer_quantile_on_pretrain_results"
CANDIDATE_DIR="journal_results/stage0_h1_parity"
REPORT_FILE="docs/journal_stage0/h1_parity_report.json"

mkdir -p "$RESULT_DIR"
if [[ ! -f "$RESULT_DIR/$CHECKPOINT_NAME" && -f "$LEGACY_RESULT_DIR/$CHECKPOINT_NAME" ]]; then
    cp "$LEGACY_RESULT_DIR/$CHECKPOINT_NAME" "$RESULT_DIR/$CHECKPOINT_NAME"
    echo "Copied checkpoint from $LEGACY_RESULT_DIR to $RESULT_DIR"
fi

python train.py \
    --data_dir "data/datasets/ST_EVCDP_v2_canonical/" \
    --model_name dura_pag_informer_quantile_on_pretrain \
    --load_method "models.PAGInformerQuantile(a_sparse=adj_sparse, seq=seq_len, quantiles=quantiles).to(device)" \
    --is_train False \
    --is_pre_train True \
    --use_cuda True \
    --batch_size 8 \
    --quantiles 0.05,0.1,0.2,0.5,0.8,0.9,0.95

echo "H=[1] journal parity candidate run finished."

mkdir -p "$CANDIDATE_DIR"
cp -a "$RESULT_DIR"/. "$CANDIDATE_DIR"/
echo "Copied candidate outputs to $CANDIDATE_DIR"

python scripts/journal/check_h1_parity.py \
    --reference-dir canonical_main_results \
    --candidate-dir "$CANDIDATE_DIR" \
    --reference-point-metrics-file canonical_point_q50.csv \
    --candidate-point-metrics-file dura_pag_informer_quantile_on_pretrain_1bs8_point_q50.csv \
    --report-file "$REPORT_FILE"

echo "Wrote H=[1] parity report to $REPORT_FILE"
