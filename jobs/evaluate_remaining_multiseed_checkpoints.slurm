#!/bin/bash
#SBATCH --job-name=remain_eval
#SBATCH --partition=gpu
#SBATCH --account=kunf0115
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-9%5
#SBATCH --output=logs/remain_eval.%A_%a.out
#SBATCH --error=logs/remain_eval.%A_%a.err

set -euo pipefail

# Array mapping:
#   0-4  PatchTST checkpoints, seeds 2023-2027
#   5-9  HG-Re-PatchTST (Ours) checkpoints, seeds 2023-2027
#
# Results are intentionally written to a snapshot root so incomplete checkpoints
# can never be mistaken for final paper runs.

REPO_ROOT="${REPO_ROOT:-/home/kunet.ae/100064422/project/forecasting/quantile-MCS}"
PYTHON="${PYTHON:-/home/kunet.ae/100064422/project/envs/forecasting-cu118/bin/python}"
MULTISEED_ROOT="${MULTISEED_ROOT:-journal_results/shenzhen_multihorizon/multiseed}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-journal_results/shenzhen_multihorizon/multiseed_checkpoint_snapshot}"

cd "$REPO_ROOT"
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SEEDS=(2023 2024 2025 2026 2027)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -gt 9 ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=$TASK_ID; expected 0-9"
  exit 2
fi

SEED="${SEEDS[$((TASK_ID % 5))]}"

if [ "$TASK_ID" -lt 5 ]; then
  MODEL_KEY="patchtst_quantile_hidden256_seq48_p8s4_lr2e4_epoch500"
  ARCH="patchtst_quantile"
  EPOCHS=500
  LEARNING_RATE=0.0002
  WEIGHT_DECAY=0.00001
  GRAPH_LAYERS=0
  ADJACENCY_MODE="distance"
else
  MODEL_KEY="horizon_gated_repatchtst_p8s4_from_repatch_identity_wd0_lr5e5_epoch300"
  ARCH="horizon_gated_repatchtst_quantile"
  EPOCHS=300
  LEARNING_RATE=0.00005
  WEIGHT_DECAY=0
  GRAPH_LAYERS=2
  ADJACENCY_MODE="identity"
fi

MODEL="journal_dura_${MODEL_KEY}_seed${SEED}"
RUN_DIR="$MULTISEED_ROOT/$MODEL_KEY/seed_${SEED}"
CHECKPOINT="$RUN_DIR/${MODEL}_checkpoint.pt"
OUT="$SNAPSHOT_ROOT/$MODEL_KEY/seed_${SEED}"

if [ ! -f "$CHECKPOINT" ]; then
  echo "Checkpoint not found: $CHECKPOINT"
  echo "This seed cannot be evaluated until a checkpoint exists."
  exit 2
fi

if [ -f "$OUT/run_metadata.json" ] \
  && [ -f "$OUT/point_metrics_by_horizon.csv" ] \
  && [ -f "$OUT/raw_interval_metrics_by_horizon.csv" ]; then
  echo "Snapshot evaluation already complete: $OUT"
  exit 0
fi

echo "MODE=checkpoint_snapshot_evaluation"
echo "MODEL_KEY=$MODEL_KEY"
echo "SEED=$SEED"
echo "CHECKPOINT=$CHECKPOINT"
echo "OUT=$OUT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

"$PYTHON" scripts/journal/train_multihorizon_raw.py \
  --data-dir data/datasets/ST_EVCDP_v2_canonical/ \
  --output-dir "$OUT" \
  --model-name "$MODEL" \
  --checkpoint-path "$CHECKPOINT" \
  --architecture "$ARCH" \
  --seed "$SEED" \
  --use-cuda True \
  --train False \
  --epochs "$EPOCHS" \
  --batch-size 4 \
  --learning-rate "$LEARNING_RATE" \
  --weight-decay "$WEIGHT_DECAY" \
  --seq-len 48 \
  --short-seq 24 \
  --hidden-dim 256 \
  --temporal-layers 1 \
  --graph-layers "$GRAPH_LAYERS" \
  --patch-len 8 \
  --patch-stride 4 \
  --moving-avg 7 \
  --transformer-layers 3 \
  --attention-heads 4 \
  --ff-dim 0 \
  --adjacency-mode "$ADJACENCY_MODE" \
  --adjacency-random-seed "$SEED" \
  --dropout 0.1 \
  --horizons 1,3,6,12,24 \
  --quantiles 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.833333333333,0.9,0.95 \
  --save-arrays False
