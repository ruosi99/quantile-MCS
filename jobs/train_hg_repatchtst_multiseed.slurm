#!/bin/bash
#SBATCH --job-name=hg_ms
#SBATCH --partition=gpu
#SBATCH --account=kunf0115
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4%2
#SBATCH --output=logs/hg_ms.%A_%a.out
#SBATCH --error=logs/hg_ms.%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/kunet.ae/100064422/project/forecasting/quantile-MCS}"
PYTHON="${PYTHON:-/home/kunet.ae/100064422/project/envs/forecasting-cu118/bin/python}"

cd "$REPO_ROOT"
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SEEDS=(2023 2024 2025 2026 2027)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$TASK_ID" -ge "${#SEEDS[@]}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=$TASK_ID for ${#SEEDS[@]} seeds"
  exit 2
fi

SEED="${SEEDS[$TASK_ID]}"
MULTISEED_ROOT="${MULTISEED_ROOT:-journal_results/shenzhen_multihorizon/multiseed}"

RUN_BASE="horizon_gated_repatchtst_p8s4_from_repatch_identity_wd0_lr5e5_epoch300"
SOURCE_RUN="${SOURCE_RUN:-graph_patchtst_ablate_g2_identity_weightdecay0_lr1e4_epoch200}"
SOURCE_DIR="${SOURCE_DIR:-journal_results/shenzhen_multihorizon/$SOURCE_RUN}"

if [ -z "${SOURCE_CKPT:-}" ]; then
  SOURCE_CKPT=$(find "$SOURCE_DIR" -maxdepth 1 -name "*checkpoint.pt" 2>/dev/null | head -n 1 || true)
fi

if [ ! -f "$SOURCE_CKPT" ]; then
  echo "Warm-start checkpoint not found: $SOURCE_CKPT"
  echo "SOURCE_DIR=$SOURCE_DIR"
  echo "Set SOURCE_CKPT=/path/to/checkpoint.pt if your HPC layout differs."
  exit 2
fi

OUT="$MULTISEED_ROOT/$RUN_BASE/seed_${SEED}"
MODEL="journal_dura_${RUN_BASE}_seed${SEED}"

echo "RUN_BASE=$RUN_BASE"
echo "SEED=$SEED"
echo "SOURCE_CKPT=$SOURCE_CKPT"
echo "OUT=$OUT"
echo "MODEL=$MODEL"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

"$PYTHON" scripts/journal/train_multihorizon_raw.py \
  --data-dir data/datasets/ST_EVCDP_v2_canonical/ \
  --output-dir "$OUT" \
  --model-name "$MODEL" \
  --architecture horizon_gated_repatchtst_quantile \
  --warm-start-checkpoint "$SOURCE_CKPT" \
  --seed "$SEED" \
  --use-cuda True \
  --train True \
  --epochs 300 \
  --batch-size 4 \
  --learning-rate 0.00005 \
  --weight-decay 0 \
  --seq-len 48 \
  --hidden-dim 256 \
  --patch-len 8 \
  --patch-stride 4 \
  --transformer-layers 3 \
  --attention-heads 4 \
  --graph-layers 2 \
  --adjacency-mode identity \
  --adjacency-random-seed "$SEED" \
  --dropout 0.1 \
  --horizons 1,3,6,12,24 \
  --quantiles 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.833333333333,0.9,0.95 \
  --save-arrays True
