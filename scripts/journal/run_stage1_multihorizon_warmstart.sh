#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/activate_env.sh"

cd "$REPO_ROOT"

python scripts/journal/train_multihorizon_raw.py \
    --data-dir "data/datasets/ST_EVCDP_v2_canonical/" \
    --output-dir "journal_results/shenzhen_multihorizon/warmstart_raw" \
    --model-name journal_dura_pag_informer_quantile_multihorizon_warmstart \
    --warm-start-checkpoint "quantile_model/dura_pag_informer_quantile_on_pretrain_results/dura_pag_informer_quantile_on_pretrain_1_bs8_completed.pt" \
    --use-cuda True \
    --train True \
    --epochs 200 \
    --batch-size 4 \
    --seq-len 24 \
    --horizons 1,3,6,12,24 \
    --quantiles 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.833333333333,0.9,0.95
