#!/bin/bash
#SBATCH --job-name=hg_rept
#SBATCH --partition=gpu
#SBATCH --account=kunf0115
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/hg_rept.%A_%a.out
#SBATCH --error=logs/hg_rept.%A_%a.err

cd /home/kunet.ae/100064422/project/forecasting/quantile-MCS

mkdir -p logs

PYTHON=/home/kunet.ae/100064422/project/envs/forecasting-cu118/bin/python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PATCH_DIR="journal_results/shenzhen_multihorizon/patchtst_quantile_hidden256_seq48_p8s4_lr2e4_epoch500"
REPATCH_DIR="journal_results/shenzhen_multihorizon/graph_patchtst_ablate_g2_identity_weightdecay0_lr1e4_epoch200"

PATCH_CKPT=$(find "$PATCH_DIR" -maxdepth 1 -name "*checkpoint.pt" | head -n 1)
REPATCH_CKPT=$(find "$REPATCH_DIR" -maxdepth 1 -name "*checkpoint.pt" | head -n 1)

RUNS=(
  "horizon_gated_repatchtst_p8s4_ws_patch_identity_wd0_lr1e4_epoch200"
  "horizon_gated_repatchtst_p8s4_from_repatch_identity_wd0_lr7e5_epoch300"
  "horizon_gated_repatchtst_p8s4_from_repatch_identity_wd0_lr5e5_epoch300"
)

SOURCE_CKPTS=(
  "$PATCH_CKPT"
  "$REPATCH_CKPT"
  "$REPATCH_CKPT"
)

LRS=("0.0001" "0.00007" "0.00005")
EPOCHS=("200" "300" "300")

RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
SOURCE_CKPT=${SOURCE_CKPTS[$SLURM_ARRAY_TASK_ID]}
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
EPOCH=${EPOCHS[$SLURM_ARRAY_TASK_ID]}

if [ ! -f "$SOURCE_CKPT" ]; then
  echo "Warm-start checkpoint not found: $SOURCE_CKPT"
  echo "PATCH_CKPT=$PATCH_CKPT"
  echo "REPATCH_CKPT=$REPATCH_CKPT"
  exit 2
fi

OUT="journal_results/shenzhen_multihorizon/$RUN"
MODEL="journal_dura_$RUN"

echo "RUN=$RUN"
echo "SOURCE_CKPT=$SOURCE_CKPT"
echo "LR=$LR EPOCHS=$EPOCH"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

$PYTHON scripts/journal/train_multihorizon_raw.py \
  --data-dir data/datasets/ST_EVCDP_v2_canonical/ \
  --output-dir "$OUT" \
  --model-name "$MODEL" \
  --architecture horizon_gated_repatchtst_quantile \
  --warm-start-checkpoint "$SOURCE_CKPT" \
  --use-cuda True \
  --train True \
  --epochs "$EPOCH" \
  --batch-size 4 \
  --learning-rate "$LR" \
  --weight-decay 0 \
  --seq-len 48 \
  --hidden-dim 256 \
  --patch-len 8 \
  --patch-stride 4 \
  --transformer-layers 3 \
  --attention-heads 4 \
  --graph-layers 2 \
  --adjacency-mode identity \
  --dropout 0.1 \
  --horizons 1,3,6,12,24 \
  --quantiles 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.833333333333,0.9,0.95 \
  --save-arrays False
