#!/bin/bash
#SBATCH --job-name=hg_cqr_ms
#SBATCH --partition=gpu
#SBATCH --account=kunf0115
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --array=0-4%2
#SBATCH --output=logs/hg_cqr_ms.%A_%a.out
#SBATCH --error=logs/hg_cqr_ms.%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/kunet.ae/100064422/project/forecasting/quantile-MCS}"
PYTHON="${PYTHON:-/home/kunet.ae/100064422/project/envs/forecasting-cu118/bin/python}"
MODEL_KEY="horizon_gated_repatchtst_p8s4_from_repatch_identity_wd0_lr5e5_epoch300"
STAGE1_ROOT="${STAGE1_ROOT:-journal_results/shenzhen_multihorizon/multiseed_checkpoint_snapshot}"
STAGE2_ROOT="${STAGE2_ROOT:-journal_results/shenzhen_multihorizon/stage2_cqr_multiseed}"
DELTAS="${DELTAS:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-4}"

cd "$REPO_ROOT"
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SEEDS=(2023 2024 2025 2026 2027)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "${#SEEDS[@]}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=$TASK_ID"
  exit 2
fi

SEED="${SEEDS[$TASK_ID]}"
STAGE1_DIR="$STAGE1_ROOT/$MODEL_KEY/seed_$SEED"
STAGE2_DIR="$STAGE2_ROOT/$MODEL_KEY/seed_$SEED"

if [ ! -f "$STAGE1_DIR/run_metadata.json" ]; then
  echo "Stage 1 snapshot metadata not found: $STAGE1_DIR/run_metadata.json"
  exit 2
fi

if [ -f "$STAGE2_DIR/global_summary.csv" ]; then
  echo "CQR output already complete: $STAGE2_DIR"
  exit 0
fi

echo "MODEL_KEY=$MODEL_KEY"
echo "SEED=$SEED"
echo "STAGE1_DIR=$STAGE1_DIR"
echo "STAGE2_DIR=$STAGE2_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

"$PYTHON" scripts/journal/calibrate_multihorizon_cqr.py \
  --stage1-output-dir "$STAGE1_DIR" \
  --output-dir "$STAGE2_DIR" \
  --use-cuda True \
  --batch-size "$BATCH_SIZE" \
  --deltas "$DELTAS" \
  --log-every-batches 25

echo "Completed CQR for seed $SEED"
cat "$STAGE2_DIR/global_summary.csv"
