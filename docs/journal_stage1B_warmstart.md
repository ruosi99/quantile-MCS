# Journal Stage 1B Warm-Start Multi-Horizon Baseline

## Record Metadata
- Record version: `stage1B-warmstart-full-2026-05-01`
- Record date: `2026-05-01`
- Machine target: `Lenovo`
- Git branch: `multi_horizon_journal`
- Stage: `1.1B`
- Status: warm-start code path implemented and full Lenovo run completed

## Purpose
Stage 1B tests whether the strong canonical single-step checkpoint can provide a better
initialization for the direct multi-horizon journal model.

Stage 1A trained the multi-horizon model from random initialization:

```text
journal_results/shenzhen_multihorizon/raw/
```

Stage 1B keeps the same multi-horizon target and output contract, but initializes all
shape-compatible parameters from the canonical H=[1] model:

```text
quantile_model/dura_pag_informer_quantile_on_pretrain_results/dura_pag_informer_quantile_on_pretrain_1_bs8_completed.pt
```

## Expected Difference From Stage 1A
Stage 1A:

```text
random initialization -> direct multi-horizon training
```

Stage 1B:

```text
canonical H=[1] checkpoint -> load matching backbone/head-prefix weights -> direct multi-horizon training
```

The output layer is intentionally not fully loaded because the old checkpoint predicts
7 single-horizon quantiles while the Stage 1B model predicts 5 horizons times 13 quantiles.

Expected skipped keys include:

```text
quantile_head.2.weight
quantile_head.2.bias
```

## Code Entry Points
- Python entry:
  - `scripts/journal/train_multihorizon_raw.py`
- Lenovo launcher:
  - `scripts/journal/run_stage1_multihorizon_warmstart.sh`

The training script now accepts:

```text
--warm-start-checkpoint <checkpoint-path>
```

The loader copies only parameters whose names and shapes match between the source and target models.
It writes the loaded/skipped key report into `run_metadata.json` under:

```json
"warm_start": { ... }
```

## Full Run Command
Run on Lenovo:

```bash
bash scripts/journal/run_stage1_multihorizon_warmstart.sh
```

Expected output directory:

```text
journal_results/shenzhen_multihorizon/warmstart_raw/
```

Expected output files:

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
created_at: 2026-05-01T21:16:40
branch: multi_horizon_journal
commit: 67b0431
output_dir: journal_results/shenzhen_multihorizon/warmstart_raw/
device: cuda:0
gpu_name: NVIDIA GeForce RTX 3060 Laptop GPU
batch_size: 4
epochs: 200
training_time: 12:25:23
best_valid_loss: 0.012632330812081911
best_epoch: 193
final_epoch_train_loss: 0.01812896
final_epoch_valid_loss: 0.01267220
predict_quantiles_shape: [829, 1682, 5, 13]
label_shape: [829, 1682, 5]
```

Warm-start loading report:

```text
loaded_key_count: 128
skipped_key_count: 2
skipped keys:
  - quantile_head.2.weight, shape [7, 128] -> [65, 128]
  - quantile_head.2.bias, shape [7] -> [65]
```

The skipped keys are expected because Stage 1B expands the output layer from single-horizon
7-quantile prediction to 5-horizon by 13-quantile prediction.

Full-run point metrics:

| Horizon | MAE | RMSE | MAPE | R2 | MedAE |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.203582 | 0.808564 | 18.125527 | 0.978260 | 0.030795 |
| 3 | 0.328695 | 1.302555 | 28.034877 | 0.943719 | 0.037187 |
| 6 | 0.500057 | 1.692275 | 42.596524 | 0.905305 | 0.031806 |
| 12 | 0.570441 | 1.660818 | 41.809863 | 0.908995 | 0.043695 |
| 24 | 0.517300 | 1.471221 | 39.743868 | 0.927982 | 0.068399 |

Raw 90 percent interval coverage before CQR:

| Horizon | PICP | MPIW | WIS |
|---:|---:|---:|---:|
| 1 | 0.604806 | 1.031343 | 1.112340 |
| 3 | 0.632331 | 1.957082 | 2.067156 |
| 6 | 0.645196 | 3.050152 | 3.185698 |
| 12 | 0.655198 | 4.077807 | 4.190385 |
| 24 | 0.657056 | 4.465956 | 4.529381 |

Quantile crossing:

```text
crossing_rate = 0.0 for all horizons
```

## Comparison Target
Compare Stage 1B against Stage 1A:

```text
journal_results/shenzhen_multihorizon/raw/point_metrics_by_horizon.csv
journal_results/shenzhen_multihorizon/warmstart_raw/point_metrics_by_horizon.csv
```

Primary expected improvement:

```text
H=1 MAE, RMSE, MAPE, R2
```

Secondary expected improvement:

```text
H=3 and H=6 point metrics
```

The warm-start run should also preserve:

```text
quantile_crossing_metrics_by_horizon.csv -> crossing_rate = 0.0
```

Observed comparison against Stage 1A:

```text
Average MAE: 0.436881 -> 0.424015, about 4.45 percent lower
Average RMSE: 1.345972 -> 1.387087, about 2.81 percent higher
Average MedAE: 0.068322 -> 0.042377, about 38.62 percent lower
H1 MAE: 0.245080 -> 0.203582, about 16.93 percent lower
```

Interpretation:

- Warm-start improves H=1 and the typical absolute error profile.
- It does not fully dominate Stage 1A because RMSE and some longer-horizon metrics worsen.
- It remains far weaker than the dedicated canonical single-step H=1 model, so it should be treated as the stronger multi-horizon baseline, not as evidence that multi-horizon replaces the single-step canonical result.
- Because the paper is unlikely to include Stage 1A versus Stage 1B as a central ablation, later Stage 2 calibration is run on Stage 1B warm-start as the representative multi-horizon forecast model.

## Notes
- Stage 1B does not modify the Stage 1A result directory.
- Stage 1B is not conformal calibration.
- Stage 2 adds global and horizon-wise CQR after Stage 1B.
