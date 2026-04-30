#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/activate_env.sh"

cd "$REPO_ROOT"

python scripts/journal/write_stage0_run_specs.py

python train.py \
    --data_dir "data/datasets/ST_EVCDP_v2_canonical/" \
    --model_name dura_pag_informer_quantile_on_pretrain \
    --load_method "models.PAGInformerQuantile(a_sparse=adj_sparse, seq=seq_len, quantiles=quantiles).to(device)" \
    --is_train False \
    --is_pre_train True \
    --use_cuda True \
    --batch_size 8 \
    --quantiles 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.833333333333,0.9,0.95

echo "H=[1] journal parity candidate run finished."
echo "Copy or move the generated result directory into journal_results/stage0_h1_parity before parity comparison."
