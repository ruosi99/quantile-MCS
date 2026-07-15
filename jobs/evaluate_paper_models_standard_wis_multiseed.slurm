#!/bin/bash
#SBATCH --job-name=wisfix_ms
#SBATCH --partition=gpu
#SBATCH --account=kunf0115
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-34%5
#SBATCH --output=logs/wisfix_ms.%A_%a.out
#SBATCH --error=logs/wisfix_ms.%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/kunet.ae/100064422/project/forecasting/quantile-MCS}"
PYTHON="${PYTHON:-/home/kunet.ae/100064422/project/envs/forecasting-cu118/bin/python}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-journal_results/shenzhen_multihorizon/multiseed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-journal_results/shenzhen_multihorizon/multiseed_standard_wis}"

cd "$REPO_ROOT"
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if ! grep -q "mean() / 1.5" utils/model_training/conformal.py; then
  echo "The standard WIS correction is missing from utils/model_training/conformal.py."
  echo "Pull the latest forecasting-upgrade branch before submitting this job."
  exit 2
fi

SEEDS=(2023 2024 2025 2026 2027)
MODEL_KEYS=(
  "dlinear_quantile"
  "nlinear_quantile"
  "lstm_quantile"
  "tft_quantile"
  "patchtst_quantile_hidden256_seq48_p8s4_lr2e4_epoch500"
  "multi_scale_temporal_graph_quantile"
  "horizon_gated_repatchtst_p8s4_from_repatch_identity_wd0_lr5e5_epoch300"
)
ARCHS=(
  "dlinear_quantile"
  "nlinear_quantile"
  "lstm_quantile"
  "tft_quantile"
  "patchtst_quantile"
  "multi_scale_temporal_graph_quantile"
  "horizon_gated_repatchtst_quantile"
)
SEQ_LENS=(48 48 24 24 48 24 48)
HIDDEN_DIMS=(128 128 128 128 256 128 256)
GRAPH_LAYERS=(2 2 2 2 0 2 2)
LEARNING_RATES=(0.001 0.001 0.001 0.001 0.0002 0.001 0.00005)
WEIGHT_DECAYS=(1e-5 1e-5 1e-5 1e-5 1e-5 1e-5 0)
EPOCHS=(200 200 200 200 500 200 300)
ADJACENCY_MODES=(distance distance distance distance distance distance identity)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED_COUNT="${#SEEDS[@]}"
MODEL_INDEX=$((TASK_ID / SEED_COUNT))
SEED_INDEX=$((TASK_ID % SEED_COUNT))

if [ "$MODEL_INDEX" -lt 0 ] || [ "$MODEL_INDEX" -ge "${#MODEL_KEYS[@]}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=$TASK_ID; expected 0-34"
  exit 2
fi

MODEL_KEY="${MODEL_KEYS[$MODEL_INDEX]}"
ARCH="${ARCHS[$MODEL_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
MODEL_NAME="journal_dura_${MODEL_KEY}_seed${SEED}"
CHECKPOINT="$CHECKPOINT_ROOT/$MODEL_KEY/seed_$SEED/${MODEL_NAME}_checkpoint.pt"
OUTPUT_DIR="$OUTPUT_ROOT/$MODEL_KEY/seed_$SEED"

if [ ! -f "$CHECKPOINT" ]; then
  echo "Checkpoint not found: $CHECKPOINT"
  exit 2
fi

if [ -f "$OUTPUT_DIR/run_metadata.json" ] \
  && [ -f "$OUTPUT_DIR/point_metrics_by_horizon.csv" ] \
  && [ -f "$OUTPUT_DIR/raw_interval_metrics_by_horizon.csv" ]; then
  echo "Standard-WIS evaluation already complete: $OUTPUT_DIR"
  exit 0
fi

echo "MODEL_KEY=$MODEL_KEY"
echo "SEED=$SEED"
echo "CHECKPOINT=$CHECKPOINT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

"$PYTHON" scripts/journal/train_multihorizon_raw.py \
  --data-dir data/datasets/ST_EVCDP_v2_canonical/ \
  --output-dir "$OUTPUT_DIR" \
  --model-name "$MODEL_NAME" \
  --checkpoint-path "$CHECKPOINT" \
  --architecture "$ARCH" \
  --seed "$SEED" \
  --use-cuda True \
  --train False \
  --epochs "${EPOCHS[$MODEL_INDEX]}" \
  --batch-size 4 \
  --learning-rate "${LEARNING_RATES[$MODEL_INDEX]}" \
  --weight-decay "${WEIGHT_DECAYS[$MODEL_INDEX]}" \
  --seq-len "${SEQ_LENS[$MODEL_INDEX]}" \
  --short-seq 24 \
  --hidden-dim "${HIDDEN_DIMS[$MODEL_INDEX]}" \
  --temporal-layers 1 \
  --graph-layers "${GRAPH_LAYERS[$MODEL_INDEX]}" \
  --patch-len 8 \
  --patch-stride 4 \
  --moving-avg 7 \
  --transformer-layers 3 \
  --attention-heads 4 \
  --ff-dim 0 \
  --adjacency-mode "${ADJACENCY_MODES[$MODEL_INDEX]}" \
  --adjacency-random-seed "$SEED" \
  --dropout 0.1 \
  --horizons 1,3,6,12,24 \
  --quantiles 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.833333333333,0.9,0.95 \
  --save-arrays False
