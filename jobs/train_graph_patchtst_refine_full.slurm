#!/bin/bash
#SBATCH --job-name=gpatch_ref
#SBATCH --partition=gpu
#SBATCH --account=kunf0115
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --array=0-5%4
#SBATCH --output=logs/gpatch_ref.%A_%a.out
#SBATCH --error=logs/gpatch_ref.%A_%a.err

cd /home/kunet.ae/100064422/project/forecasting/quantile-MCS

PYTHON=/home/kunet.ae/100064422/project/envs/forecasting-cu118/bin/python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PATCH_DIR="journal_results/shenzhen_multihorizon/patchtst_quantile_hidden256_seq48_p8s4_lr2e4_epoch500"
G2_DIR="journal_results/shenzhen_multihorizon/graph_patchtst_v1_p8s4_ws_g2_lr1e4_epoch200"
G1_DIR="journal_results/shenzhen_multihorizon/graph_patchtst_v1_p8s4_ws_g1_lr1e4_epoch200"

RUNS=(
  "graph_patchtst_v1_p8s4_ws_g2_lr1e4_epoch400"
  "graph_patchtst_v1_p8s4_from_g2_lr7e5_epoch250"
  "graph_patchtst_v1_p8s4_from_g2_lr5e5_epoch250"
  "graph_patchtst_v1_p8s4_from_g1_lr7e5_epoch250"
  "graph_patchtst_v1_p8s4_ws_g2_lr7e5_epoch300"
  "graph_patchtst_v1_p8s4_ws_g2_lr1e4_dropout005_epoch300"
)

SOURCE_DIRS=(
  "$PATCH_DIR"
  "$G2_DIR"
  "$G2_DIR"
  "$G1_DIR"
  "$PATCH_DIR"
  "$PATCH_DIR"
)

LRS=("0.0001" "0.00007" "0.00005" "0.00007" "0.00007" "0.0001")
GRAPH_LAYERS=("2" "2" "2" "1" "2" "2")
EPOCHS=("400" "250" "250" "250" "300" "300")
DROPOUTS=("0.1" "0.1" "0.1" "0.1" "0.1" "0.05")

RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
SOURCE_DIR=${SOURCE_DIRS[$SLURM_ARRAY_TASK_ID]}
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
GL=${GRAPH_LAYERS[$SLURM_ARRAY_TASK_ID]}
EPOCH=${EPOCHS[$SLURM_ARRAY_TASK_ID]}
DROPOUT=${DROPOUTS[$SLURM_ARRAY_TASK_ID]}

SOURCE_CKPT=$(find "$SOURCE_DIR" -maxdepth 1 -name "*checkpoint.pt" | head -n 1)

if [ ! -f "$SOURCE_CKPT" ]; then
  echo "Checkpoint not found in $SOURCE_DIR"
  exit 2
fi

OUT="journal_results/shenzhen_multihorizon/$RUN"
MODEL="journal_dura_$RUN"

echo "RUN=$RUN"
echo "SOURCE_CKPT=$SOURCE_CKPT"
echo "LR=$LR GRAPH_LAYERS=$GL EPOCHS=$EPOCH DROPOUT=$DROPOUT"

$PYTHON scripts/journal/train_multihorizon_raw.py \
  --data-dir data/datasets/ST_EVCDP_v2_canonical/ \
  --output-dir "$OUT" \
  --model-name "$MODEL" \
  --architecture graph_patchtst_quantile \
  --warm-start-checkpoint "$SOURCE_CKPT" \
  --use-cuda True \
  --train True \
  --epochs "$EPOCH" \
  --batch-size 4 \
  --learning-rate "$LR" \
  --weight-decay 1e-5 \
  --seq-len 48 \
  --hidden-dim 256 \
  --patch-len 8 \
  --patch-stride 4 \
  --transformer-layers 3 \
  --attention-heads 4 \
  --graph-layers "$GL" \
  --adjacency-mode distance \
  --dropout "$DROPOUT" \
  --horizons 1,3,6,12,24 \
  --quantiles 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.833333333333,0.9,0.95 \
  --save-arrays False
