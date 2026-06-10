#!/bin/bash
#SBATCH --job-name=charged_gpatch
#SBATCH --partition=gpu
#SBATCH --account=kunf0115
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/charged_gpatch.%A_%a.out
#SBATCH --error=logs/charged_gpatch.%A_%a.err

cd /home/kunet.ae/100064422/project/forecasting/quantile-MCS

PYTHON=/home/kunet.ae/100064422/project/envs/forecasting-cu118/bin/python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CITIES=("LOA" "MEL")
CITY=${CITIES[$SLURM_ARRAY_TASK_ID]}
DATA_DIR="CHARGED/data/${CITY}_remove_zero"
SOURCE_RUN="charged_${CITY}_patchtst_source_seq48_p8s4_lr2e4_epoch300_volume"
SOURCE_DIR="journal_results/charged_case_study/${CITY}/${SOURCE_RUN}"
SOURCE_CKPT=$(find "$SOURCE_DIR" -maxdepth 1 -name "*checkpoint.pt" | head -n 1)

if [ ! -f "$SOURCE_CKPT" ]; then
  echo "Checkpoint not found in $SOURCE_DIR"
  echo "Submit jobs/train_charged_patchtst_source_case_study.slurm first."
  exit 2
fi

RUN="charged_${CITY}_graph_patchtst_case_volume_seq48_p8s4_ws_g2_weightdecay0_lr1e4_epoch200"
OUT="journal_results/charged_case_study/${CITY}/${RUN}"
MODEL="journal_charged_${CITY}_volume_graph_patchtst_seq48_p8s4_ws_g2_weightdecay0_lr1e4_epoch200"

echo "CITY=$CITY"
echo "DATA_DIR=$DATA_DIR"
echo "SOURCE_CKPT=$SOURCE_CKPT"
echo "OUT=$OUT"

$PYTHON scripts/journal/train_multihorizon_raw.py \
  --data-dir "$DATA_DIR" \
  --dataset-family charged_site \
  --target-feature volume \
  --charged-adjacency-k 8 \
  --output-dir "$OUT" \
  --model-name "$MODEL" \
  --architecture graph_patchtst_quantile \
  --warm-start-checkpoint "$SOURCE_CKPT" \
  --use-cuda True \
  --train True \
  --epochs 200 \
  --batch-size 4 \
  --learning-rate 0.0001 \
  --weight-decay 0.0 \
  --seq-len 48 \
  --hidden-dim 256 \
  --patch-len 8 \
  --patch-stride 4 \
  --transformer-layers 3 \
  --attention-heads 4 \
  --ff-dim 0 \
  --graph-layers 2 \
  --adjacency-mode distance \
  --dropout 0.1 \
  --horizons 1,3,6,12,24 \
  --quantiles 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.833333333333,0.9,0.95 \
  --save-arrays True
