#!/bin/bash
#SBATCH --job-name=base_ms
#SBATCH --partition=gpu
#SBATCH --account=kunf0115
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --array=0-34%5
#SBATCH --output=logs/base_ms.%A_%a.out
#SBATCH --error=logs/base_ms.%A_%a.err

set -euo pipefail

# Default array 0-34 runs all seed-independent baselines plus PatchTST.
# After PatchTST seed checkpoints finish, submit GraphPatchTST with:
#   sbatch --array=35-39%2 jobs/train_multiseed_baselines.slurm

REPO_ROOT="${REPO_ROOT:-/home/kunet.ae/100064422/project/forecasting/quantile-MCS}"
PYTHON="${PYTHON:-/home/kunet.ae/100064422/project/envs/forecasting-cu118/bin/python}"

cd "$REPO_ROOT"
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SEEDS=(2023 2024 2025 2026 2027)
SEED_COUNT="${#SEEDS[@]}"

MODEL_KEYS=(
  "pag_informer_warmstart"
  "lstm_quantile"
  "tft_quantile"
  "patchtst_quantile_hidden256_seq48_p8s4_lr2e4_epoch500"
  "nlinear_quantile"
  "dlinear_quantile"
  "multi_scale_temporal_graph_quantile"
  "graph_patchtst_seed_patch_ws_g2_lr1e4_epoch200"
)

ARCHS=(
  "pag_informer"
  "lstm_quantile"
  "tft_quantile"
  "patchtst_quantile"
  "nlinear_quantile"
  "dlinear_quantile"
  "multi_scale_temporal_graph_quantile"
  "graph_patchtst_quantile"
)

SEQ_LENS=(24 24 24 48 48 48 24 48)
SHORT_SEQS=(24 24 24 24 24 24 24 24)
HIDDEN_DIMS=(128 128 128 256 128 128 128 256)
TEMPORAL_LAYERS=(1 1 1 1 1 1 1 1)
GRAPH_LAYERS=(2 2 2 0 2 2 2 2)
PATCH_LENS=(8 8 8 8 8 8 8 8)
PATCH_STRIDES=(4 4 4 4 4 4 4 4)
MOVING_AVGS=(7 7 7 7 7 7 7 7)
TRANSFORMER_LAYERS=(3 3 3 3 3 3 3 3)
ATTENTION_HEADS=(4 4 4 4 4 4 4 4)
FF_DIMS=(0 0 0 0 0 0 0 0)
DROPOUTS=(0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1)
LRS=(0.001 0.001 0.001 0.0002 0.001 0.001 0.001 0.0001)
WEIGHT_DECAYS=(1e-5 1e-5 1e-5 1e-5 1e-5 1e-5 1e-5 1e-5)
EPOCHS=(200 200 200 500 200 200 200 200)
ADJ_MODES=(distance distance distance distance distance distance distance distance)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL_IDX=$((TASK_ID / SEED_COUNT))
SEED_IDX=$((TASK_ID % SEED_COUNT))

if [ "$MODEL_IDX" -ge "${#MODEL_KEYS[@]}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=$TASK_ID"
  echo "Max valid task id is $((${#MODEL_KEYS[@]} * SEED_COUNT - 1))"
  exit 2
fi

SEED="${SEEDS[$SEED_IDX]}"
MODEL_KEY="${MODEL_KEYS[$MODEL_IDX]}"
ARCH="${ARCHS[$MODEL_IDX]}"
MULTISEED_ROOT="${MULTISEED_ROOT:-journal_results/shenzhen_multihorizon/multiseed}"
OUT="$MULTISEED_ROOT/$MODEL_KEY/seed_${SEED}"
MODEL="journal_dura_${MODEL_KEY}_seed${SEED}"
WARMSTART=""

if [ "$MODEL_KEY" = "pag_informer_warmstart" ]; then
  WARMSTART="${PAG_WARMSTART_CKPT:-quantile_model/dura_pag_informer_quantile_on_pretrain_results/dura_pag_informer_quantile_on_pretrain_1_bs8_completed.pt}"
fi

if [ "$MODEL_KEY" = "graph_patchtst_seed_patch_ws_g2_lr1e4_epoch200" ]; then
  PATCH_SEED_DIR="$MULTISEED_ROOT/patchtst_quantile_hidden256_seq48_p8s4_lr2e4_epoch500/seed_${SEED}"
  WARMSTART=$(find "$PATCH_SEED_DIR" -maxdepth 1 -name "*checkpoint.pt" 2>/dev/null | head -n 1 || true)
  if [ ! -f "$WARMSTART" ]; then
    echo "Seed-specific PatchTST checkpoint not found for GraphPatchTST."
    echo "Expected under: $PATCH_SEED_DIR"
    echo "Run array 15-19 first, then resubmit GraphPatchTST with array 35-39."
    exit 2
  fi
fi

if [ -n "$WARMSTART" ] && [ ! -f "$WARMSTART" ]; then
  echo "Warm-start checkpoint not found: $WARMSTART"
  exit 2
fi

SAVE_ARRAYS="${SAVE_ARRAYS:-False}"

echo "TASK_ID=$TASK_ID"
echo "MODEL_KEY=$MODEL_KEY"
echo "ARCH=$ARCH"
echo "SEED=$SEED"
echo "OUT=$OUT"
echo "MODEL=$MODEL"
echo "WARMSTART=$WARMSTART"
echo "SAVE_ARRAYS=$SAVE_ARRAYS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

"$PYTHON" scripts/journal/train_multihorizon_raw.py \
  --data-dir data/datasets/ST_EVCDP_v2_canonical/ \
  --output-dir "$OUT" \
  --model-name "$MODEL" \
  --architecture "$ARCH" \
  --warm-start-checkpoint "$WARMSTART" \
  --seed "$SEED" \
  --use-cuda True \
  --train True \
  --epochs "${EPOCHS[$MODEL_IDX]}" \
  --batch-size 4 \
  --learning-rate "${LRS[$MODEL_IDX]}" \
  --weight-decay "${WEIGHT_DECAYS[$MODEL_IDX]}" \
  --seq-len "${SEQ_LENS[$MODEL_IDX]}" \
  --short-seq "${SHORT_SEQS[$MODEL_IDX]}" \
  --hidden-dim "${HIDDEN_DIMS[$MODEL_IDX]}" \
  --temporal-layers "${TEMPORAL_LAYERS[$MODEL_IDX]}" \
  --graph-layers "${GRAPH_LAYERS[$MODEL_IDX]}" \
  --patch-len "${PATCH_LENS[$MODEL_IDX]}" \
  --patch-stride "${PATCH_STRIDES[$MODEL_IDX]}" \
  --moving-avg "${MOVING_AVGS[$MODEL_IDX]}" \
  --transformer-layers "${TRANSFORMER_LAYERS[$MODEL_IDX]}" \
  --attention-heads "${ATTENTION_HEADS[$MODEL_IDX]}" \
  --ff-dim "${FF_DIMS[$MODEL_IDX]}" \
  --adjacency-mode "${ADJ_MODES[$MODEL_IDX]}" \
  --adjacency-random-seed "$SEED" \
  --dropout "${DROPOUTS[$MODEL_IDX]}" \
  --horizons 1,3,6,12,24 \
  --quantiles 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.833333333333,0.9,0.95 \
  --save-arrays "$SAVE_ARRAYS"
