# Journal Stage 1 Multi-Horizon Raw Baseline

## Record Metadata
- Record version: `stage1-multihorizon-raw-full-2026-05-01`
- Record date: `2026-05-01`
- Time zone: `Asia/Dubai`
- Machine used for smoke run: `Lenovo`
- Git branch: `multi_horizon_journal`
- Base commit before local Stage 1 edits: `b861229`
- Stage: `1.1`
- Status: code path implemented, smoke-tested, and full Lenovo training run completed

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

Full Lenovo run metadata:

```text
created_at: 2026-05-01T03:30:20
branch: multi_horizon_journal
commit: b861229
output_dir: journal_results/shenzhen_multihorizon/raw/
device: cuda:0
gpu_name: NVIDIA GeForce RTX 3060 Laptop GPU
batch_size: 4
epochs: 200
training_time: 12:39:50
best_valid_loss: 0.012310015896992118
best_epoch: 195
final_epoch_train_loss: 0.01801140
final_epoch_valid_loss: 0.01266819
predict_quantiles_shape: [829, 1682, 5, 13]
label_shape: [829, 1682, 5]
```

Full-run point metrics:

| Horizon | MAE | RMSE | MAPE | R2 | MedAE |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.245080 | 0.860865 | 20.429586 | 0.975357 | 0.057590 |
| 3 | 0.336551 | 1.150000 | 27.475218 | 0.956131 | 0.061777 |
| 6 | 0.522251 | 1.680275 | 41.130406 | 0.906643 | 0.075787 |
| 12 | 0.561272 | 1.614462 | 41.741566 | 0.914005 | 0.066863 |
| 24 | 0.519250 | 1.424259 | 37.235874 | 0.932507 | 0.079594 |

Raw 90 percent interval coverage before CQR:

| Horizon | PICP | MPIW | WIS |
|---:|---:|---:|---:|
| 1 | 0.469888 | 0.963270 | 1.064522 |
| 3 | 0.638815 | 1.771716 | 1.875286 |
| 6 | 0.661366 | 2.703414 | 2.853122 |
| 12 | 0.670741 | 4.473385 | 4.549090 |
| 24 | 0.668473 | 4.399555 | 4.458932 |

Quantile crossing:

```text
crossing_rate = 0.0 for all horizons
```

Main interpretation:

- The direct multi-horizon path is functional and stable.
- Raw intervals are substantially under-covered, especially for H=1.
- H=1 point accuracy is much weaker than the dedicated canonical single-step model, so Stage 1A is mainly a baseline for the journal multi-horizon path rather than a replacement for the canonical single-step result.

## Notes And Boundaries
- Stage 1 is a raw direct multi-horizon forecasting baseline.
- Stage 1 does not implement global CQR, horizon-wise CQR, stratified conformal calibration, or one-sided decision calibration.
- Calibration work is handled by Stage 2.
- Conference-facing canonical outputs remain untouched.
