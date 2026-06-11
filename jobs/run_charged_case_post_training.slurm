#!/bin/bash
#SBATCH --job-name=charged_post
#SBATCH --partition=gpu
#SBATCH --account=kunf0115
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=0-06:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/charged_post.%A_%a.out
#SBATCH --error=logs/charged_post.%A_%a.err

cd /home/kunet.ae/100064422/project/forecasting/quantile-MCS

PYTHON=/home/kunet.ae/100064422/project/envs/forecasting-cu118/bin/python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CITIES=("LOA" "MEL")
CITY=${CITIES[$SLURM_ARRAY_TASK_ID]}
RUN="charged_${CITY}_graph_patchtst_case_volume_seq48_p8s4_ws_g2_weightdecay0_lr1e4_epoch200"
STAGE1_DIR="journal_results/charged_case_study/${CITY}/${RUN}"
STAGE2_DIR="journal_results/charged_case_study/${CITY}/stage2_cqr/${RUN}"
STAGE4_DIR="journal_results/charged_case_study/${CITY}/stage4_decision/${RUN}"

if [ ! -f "$STAGE1_DIR/run_metadata.json" ]; then
  echo "Missing Stage 1 metadata: $STAGE1_DIR/run_metadata.json"
  exit 2
fi
if [ ! -f "$STAGE1_DIR/predict_quantiles.npy" ]; then
  echo "Missing Stage 1 test arrays: $STAGE1_DIR/predict_quantiles.npy"
  echo "The GraphPatchTST case-study run must be launched with --save-arrays True."
  exit 2
fi

echo "CITY=$CITY"
echo "STAGE1_DIR=$STAGE1_DIR"
echo "STAGE2_DIR=$STAGE2_DIR"
echo "STAGE4_DIR=$STAGE4_DIR"

$PYTHON scripts/journal/calibrate_multihorizon_cqr.py \
  --stage1-output-dir "$STAGE1_DIR" \
  --output-dir "$STAGE2_DIR" \
  --use-cuda True \
  --deltas 0.1,0.2,0.4

$PYTHON scripts/journal/evaluate_stage4_decision.py \
  --stage1-output-dir "$STAGE1_DIR" \
  --stage2-output-dir "$STAGE2_DIR" \
  --output-dir "$STAGE4_DIR" \
  --use-cuda True \
  --cost-ratios 1:1,3:1,5:1,9:1,19:1 \
  --machine charged_case_study_hpc
