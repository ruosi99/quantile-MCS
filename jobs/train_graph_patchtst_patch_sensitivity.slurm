#!/bin/bash
#SBATCH --job-name=gpatch_sens
#SBATCH --partition=gpu
#SBATCH --account=kunf0115
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3%3
#SBATCH --output=logs/gpatch_sens.%A_%a.out
#SBATCH --error=logs/gpatch_sens.%A_%a.err

cd /home/kunet.ae/100064422/project/forecasting/quantile-MCS

PYTHON=/home/kunet.ae/100064422/project/envs/forecasting-cu118/bin/python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PATCH8_DIR="journal_results/shenzhen_multihorizon/patchtst_quantile_hidden256_seq48_p8s4_lr2e4_epoch500"
PATCH16_DIR="journal_results/shenzhen_multihorizon/patchtst_quantile_hidden256_seq48_p16s8_lr2e4_epoch500"

RUNS=(
  "graph_patchtst_sens_seq48_p16s8_ws_g2_lr1e4_epoch200"
  "graph_patchtst_sens_seq48_p8s4_ws_g1_lr7e5_epoch300"
  "graph_patchtst_sens_seq48_p8s4_ws_g2_weightdecay0_lr1e4_epoch200"
  "graph_patchtst_sens_seq48_p8s4_ws_g2_ff512_lr1e4_epoch200"
)

SOURCE_DIRS=("$PATCH16_DIR" "$PATCH8_DIR" "$PATCH8_DIR" "$PATCH8_DIR")
PATCH_LENS=("16" "8" "8" "8")
PATCH_STRIDES=("8" "4" "4" "4")
GRAPH_LAYERS=("2" "1" "2" "2")
LRS=("0.0001" "0.00007" "0.0001" "0.0001")
EPOCHS=("200" "300" "200" "200")
WEIGHT_DECAYS=("1e-5" "1e-5" "0.0" "1e-5")
FF_DIMS=("0" "0" "0" "512")

RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
SOURCE_DIR=${SOURCE_DIRS[$SLURM_ARRAY_TASK_ID]}
PL=${PATCH_LENS[$SLURM_ARRAY_TASK_ID]}
PS=${PATCH_STRIDES[$SLURM_ARRAY_TASK_ID]}
GL=${GRAPH_LAYERS[$SLURM_ARRAY_TASK_ID]}
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
EPOCH=${EPOCHS[$SLURM_ARRAY_TASK_ID]}
WD=${WEIGHT_DECAYS[$SLURM_ARRAY_TASK_ID]}
FF=${FF_DIMS[$SLURM_ARRAY_TASK_ID]}

SOURCE_CKPT=$(find "$SOURCE_DIR" -maxdepth 1 -name "*checkpoint.pt" | head -n 1)

if [ ! -f "$SOURCE_CKPT" ]; then
  echo "Checkpoint not found in $SOURCE_DIR"
  exit 2
fi

OUT="journal_results/shenzhen_multihorizon/$RUN"
MODEL="journal_dura_$RUN"

echo "RUN=$RUN"
echo "SOURCE_CKPT=$SOURCE_CKPT"
echo "PATCH_LEN=$PL PATCH_STRIDE=$PS GRAPH_LAYERS=$GL LR=$LR EPOCHS=$EPOCH WD=$WD FF=$FF"

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
  --weight-decay "$WD" \
  --seq-len 48 \
  --hidden-dim 256 \
  --patch-len "$PL" \
  --patch-stride "$PS" \
  --transformer-layers 3 \
  --attention-heads 4 \
  --ff-dim "$FF" \
  --graph-layers "$GL" \
  --adjacency-mode distance \
  --dropout 0.1 \
  --horizons 1,3,6,12,24 \
  --quantiles 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.833333333333,0.9,0.95 \
  --save-arrays False
