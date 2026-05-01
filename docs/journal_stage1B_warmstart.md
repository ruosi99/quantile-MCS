# Journal Stage 1B Warm-Start Multi-Horizon Baseline

## Record Metadata
- Record version: `stage1B-warmstart-scaffold-2026-05-01`
- Record date: `2026-05-01`
- Machine target: `Lenovo`
- Git branch: `multi_horizon_journal`
- Stage: `1.1B`
- Status: warm-start code path implemented; full Lenovo run pending

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

## Notes
- Stage 1B does not modify the Stage 1A result directory.
- Stage 1B is not conformal calibration.
- Stage 2 should still add global and horizon-wise CQR after Stage 1B is reviewed.
