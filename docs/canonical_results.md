# Canonical Results

## Canonical Experiment

- Experiment name: `dura_pag_informer_quantile_on_pretrain`
- Model family: `PAGInformerQuantile`
- Task focus: probabilistic EV charging demand forecasting on duration demand
- Evaluation setting: saved point forecasts plus conformalized quantile intervals at `delta = 0.1, 0.2, 0.4`

## Canonical Result Folder

- Canonical result folder: `quantile_model/dura_pag_informer_quantile_on_pretrain_results`

This is the current paper result folder that should be reused for tables, figures, and follow-up analysis unless a newer validated folder replaces it.

## Currently Trustworthy Metrics

- Point forecasting metrics from `dura_pag_informer_quantile_on_pretrain_1bs8_point_q50.csv`
  - `MSE`, `RMSE`, `MAPE`, `RAE`, `MAE`, `R2`, `MedAE`, `EVS`
- Interval metrics from the canonical saved interval arrays
  - `PICP` for `delta = 0.1, 0.2, 0.4`
  - `MPIW` for `delta = 0.1, 0.2, 0.4`
- Raw-vs-calibrated coverage comparison
  - raw 90% interval coverage is about `62.21%`
  - calibrated 90% interval coverage is about `88.51%`
- Quantile monotonicity
  - saved quantile predictions are monotonic with zero detected crossing violations

## Not Yet Fully Trustworthy

- Full checkpoint reproducibility on the current machine
  - Reason: the available Python environments currently do not include the required `torch` runtime, so inference-only rerun from checkpoint has not yet been executed here.
- Groupwise conformal details at `delta = 0.4`
  - Reason: the saved `cqr_s_hat_delta0.4.npy` contains a negative group value, so the group calibration behavior should still be treated cautiously even though the saved interval arrays themselves are usable.

## WIS Note

- The original saved `WIS` values for `delta = 0.2` and `delta = 0.4` were produced with the wrong default `delta=0.1` inside interval evaluation.
- This has now been fixed in `train.py` so future reruns use the correct `delta` for each interval setting.
- Corrected `WIS` values should be taken from the refreshed canonical interval CSV files after recomputation.
