#!/bin/bash
#SBATCH --job-name=gpatch_ab
#SBATCH --partition=gpu
#SBATCH --account=kunf0115
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --array=0-5%4
#SBATCH --output=logs/gpatch_ab.%A_%a.out
#SBATCH --error=logs/gpatch_ab.%A_%a.err

cd /home/kunet.ae/100064422/project/forecasting/quantile-MCS

PYTHON=/home/kunet.ae/100064422/project/envs/forecasting-cu118/bin/python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PATCH_DIR="journal_results/shenzhen_multihorizon/patchtst_quantile_hidden256_seq48_p8s4_lr2e4_epoch500"
PATCH_CKPT=$(find "$PATCH_DIR" -maxdepth 1 -name "*checkpoint.pt" | head -n 1)

RUNS=(
  "graph_patchtst_ablate_g0_patch_finetune_lr1e4_epoch200"
  "graph_patchtst_ablate_g2_no_warmstart_lr1e4_epoch500"
  "graph_patchtst_ablate_g2_identity_ws_lr1e4_epoch200"
  "graph_patchtst_ablate_g2_random_ws_lr1e4_epoch200"
  "graph_patchtst_ablate_g3_distance_ws_lr1e4_epoch200"
  "graph_patchtst_ablate_g1_identity_ws_lr1e4_epoch200"
)

WARMSTARTS=(
  "$PATCH_CKPT"
  ""
  "$PATCH_CKPT"
  "$PATCH_CKPT"
  "$PATCH_CKPT"
  "$PATCH_CKPT"
)

GRAPH_LAYERS=("0" "2" "2" "2" "3" "1")
ADJ_MODES=("distance" "distance" "identity" "random_permutation" "distance" "identity")
EPOCHS=("200" "500" "200" "200" "200" "200")

RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
WARMSTART=${WARMSTARTS[$SLURM_ARRAY_TASK_ID]}
GL=${GRAPH_LAYERS[$SLURM_ARRAY_TASK_ID]}
ADJ=${ADJ_MODES[$SLURM_ARRAY_TASK_ID]}
EPOCH=${EPOCHS[$SLURM_ARRAY_TASK_ID]}

if [ "$SLURM_ARRAY_TASK_ID" != "1" ] && [ ! -f "$PATCH_CKPT" ]; then
  echo "PatchTST checkpoint not found in $PATCH_DIR"
  exit 2
fi

OUT="journal_results/shenzhen_multihorizon/$RUN"
MODEL="journal_dura_$RUN"

echo "RUN=$RUN"
echo "WARMSTART=$WARMSTART"
echo "GRAPH_LAYERS=$GL ADJACENCY_MODE=$ADJ EPOCHS=$EPOCH"

$PYTHON scripts/journal/train_multihorizon_raw.py \
  --data-dir data/datasets/ST_EVCDP_v2_canonical/ \
  --output-dir "$OUT" \
  --model-name "$MODEL" \
  --architecture graph_patchtst_quantile \
  --warm-start-checkpoint "$WARMSTART" \
  --use-cuda True \
  --train True \
  --epochs "$EPOCH" \
  --batch-size 4 \
  --learning-rate 0.0001 \
  --weight-decay 1e-5 \
  --seq-len 48 \
  --hidden-dim 256 \
  --patch-len 8 \
  --patch-stride 4 \
  --transformer-layers 3 \
  --attention-heads 4 \
  --graph-layers "$GL" \
  --adjacency-mode "$ADJ" \
  --adjacency-random-seed 2023 \
  --dropout 0.1 \
  --horizons 1,3,6,12,24 \
  --quantiles 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.833333333333,0.9,0.95 \
  --save-arrays False
