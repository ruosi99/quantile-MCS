# Journal Stage 1 Multi-Horizon Raw Baseline

## Record Metadata
- Record version: `stage1-multihorizon-raw-scaffold-2026-04-30`
- Record date: `2026-04-30 14:44:56 +04:00`
- Time zone: `Asia/Dubai`
- Machine used for smoke run: `Lenovo`
- Git branch: `multi_horizon_journal`
- Base commit before local Stage 1 edits: `b861229`
- Stage: `1.1`
- Status: code path implemented and smoke-tested; full training run is still pending

## Goal
Stage 1 adds a direct multi-horizon raw forecasting path for the journal branch.

The target output contract is:

```text
predict_quantiles: (T, N, H, Q)
label_list:        (T, N, H)
predict_point_q50: (T, N, H)
```

For the current Shenzhen setup:

```text
N = 1682
H = [1, 3, 6, 12, 24]
Q = 13
```

## What Was Implemented
- Added multi-horizon window creation:
  - `utils/model_training/training_utils.py`
  - `create_multi_horizon_rnn_data`
  - `CreateMultiHorizonDataset`
- Extended quantile loss:
  - `utils/model_training/loss_functions.py`
  - `QuantileLoss` now supports both `(B,N,Q)` and `(B,N,H,Q)`.
- Extended model output:
  - `utils/model_training/models.py`
  - `PAGInformerQuantile(..., horizons=[...])` can output `(B,N,H,Q)`.
  - The old single-step behavior remains `(B,N,Q)` when `horizons` is omitted or `[1]`.
- Added Stage 1 training/evaluation entry:
  - `scripts/journal/train_multihorizon_raw.py`
- Added rerunnable Lenovo launcher:
  - `scripts/journal/run_stage1_multihorizon_raw.sh`
- Added tests:
  - `tests/test_journal_stage1_multihorizon.py`
- Updated the generated multi-horizon run spec:
  - `docs/journal_stage0/journal_multihorizon_stub_run_spec.json`

## Smoke Run
Smoke command used on Lenovo:

```bash
conda run -n py12 python scripts/journal/train_multihorizon_raw.py \
  --data-dir data/datasets/ST_EVCDP_v2_canonical/ \
  --output-dir journal_results/shenzhen_multihorizon/smoke_raw \
  --model-name journal_dura_pag_informer_quantile_multihorizon_smoke \
  --use-cuda True \
  --train True \
  --epochs 1 \
  --batch-size 2 \
  --seq-len 24 \
  --horizons 1,3,6,12,24 \
  --quantiles 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.833333333333,0.9,0.95 \
  --max-train-batches 1 \
  --max-valid-batches 1 \
  --max-test-batches 1
```

Smoke output:

```text
predict_quantiles_shape: [2, 1682, 5, 13]
label_shape: [2, 1682, 5]
device: cuda:0
gpu_name: NVIDIA GeForce RTX 3060 Laptop GPU
best_valid_loss: 1.1696498394012451
```

Generated smoke files:

```text
journal_results/shenzhen_multihorizon/smoke_raw/predict_quantiles.npy
journal_results/shenzhen_multihorizon/smoke_raw/label_list.npy
journal_results/shenzhen_multihorizon/smoke_raw/predict_point_q50.npy
journal_results/shenzhen_multihorizon/smoke_raw/point_metrics_by_horizon.csv
journal_results/shenzhen_multihorizon/smoke_raw/raw_interval_metrics_by_horizon.csv
journal_results/shenzhen_multihorizon/smoke_raw/quantile_crossing_metrics_by_horizon.csv
journal_results/shenzhen_multihorizon/smoke_raw/run_metadata.json
journal_results/shenzhen_multihorizon/smoke_raw/*_checkpoint.pt
```

These smoke outputs are ignored by Git through `journal_results/`.

## Verification
Focused test command:

```bash
conda run -n py12 python -m pytest \
  tests/test_journal_stage1_multihorizon.py \
  tests/test_stage0_runtime.py \
  tests/test_journal_contracts.py \
  -q
```

Result:

```text
15 passed
```

## Full Stage 1 Run
The full Lenovo run command is:

```bash
bash scripts/journal/run_stage1_multihorizon_raw.sh
```

Expected full-run output directory:

```text
journal_results/shenzhen_multihorizon/raw/
```

Expected full-run files:

```text
predict_quantiles.npy
label_list.npy
predict_point_q50.npy
point_metrics_by_horizon.csv
raw_interval_metrics_by_horizon.csv
quantile_crossing_metrics_by_horizon.csv
run_metadata.json
*_checkpoint.pt
```

## Notes And Boundaries
- Stage 1 is a raw direct multi-horizon forecasting baseline.
- Stage 1 does not yet implement global CQR, horizon-wise CQR, stratified conformal calibration, or one-sided decision calibration.
- Calibration work should begin in Stage 2 after the full Stage 1 run is reviewed.
- Conference-facing canonical outputs remain untouched.
