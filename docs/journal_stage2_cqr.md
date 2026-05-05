# Journal Stage 2 CQR Calibration

## Record Metadata
- Record version: `stage2-cqr-warmstart-full-2026-05-01`
- Record date: `2026-05-01`
- Machine target: `Lenovo`
- Git branch: `multi_horizon_journal`
- Stage: `2.1-2.2`
- Status: global and horizon-wise CQR code implemented; smoke test passed; full Lenovo warm-start run completed

## Purpose
Stage 2 calibrates the direct multi-horizon quantile forecasts produced in Stage 1.

It does not retrain the model. It loads a Stage 1 checkpoint, rebuilds the calibration and
test splits, computes conformalized quantile regression thresholds on the calibration split,
and evaluates calibrated intervals on the test split.

The first two Stage 2 calibration baselines are:

```text
global_cqr   -> one threshold per delta, pooled across time, station, and horizon
horizon_cqr  -> one threshold per delta and horizon
```

These are the baseline calibration methods that later stratified calibration should beat.

## Code Entry Points
- Python entry:
  - `scripts/journal/calibrate_multihorizon_cqr.py`
- Lenovo launchers:
  - `scripts/journal/run_stage2_cqr_raw.sh`
  - `scripts/journal/run_stage2_cqr_warmstart.sh`
- Tests:
  - `tests/test_journal_stage2_cqr.py`

## Important Implementation Note
The legacy `utils/model_training/conformal.py` function `apply_cqr_interval()` has previously
been used for an ablation where `s_hat` was forced to zero.

Stage 2 intentionally uses a journal-specific CQR implementation instead of that legacy apply
function. The Stage 2 test suite checks that nonzero `s_hat` values actually expand the interval.

The applied interval rule is:

```text
L_cal = L_raw - s_hat
U_cal = U_raw + s_hat
```

with nonnegative clipping after calibration.

## Full Run Commands
Run from the repository root on Lenovo.

For Stage 1A from-scratch raw:

```bash
bash scripts/journal/run_stage2_cqr_raw.sh
```

Expected output:

```text
journal_results/shenzhen_multihorizon/stage2_cqr/raw/
```

This run is optional for now. The current paper-facing Stage 2 record uses the warm-start
multi-horizon model because it is the stronger representative Stage 1 baseline.

For Stage 1B warm-start raw:

```bash
bash scripts/journal/run_stage2_cqr_warmstart.sh
```

Expected output:

```text
journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw/
```

## Output Files
Each Stage 2 output directory should contain:

```text
stage2_metadata.json
cqr_thresholds.csv
calibrated_interval_metrics_by_horizon.csv
raw_vs_cqr_comparison_by_horizon.csv
global_summary.csv
```

No large calibrated interval arrays are saved by default. The saved Stage 1 predictions plus the
threshold CSVs are enough to reconstruct intervals later if needed.

## Smoke Test
The implemented smoke run used one calibration batch and one test batch:

```powershell
conda run -n py12 python scripts/journal/calibrate_multihorizon_cqr.py `
  --stage1-output-dir journal_results/shenzhen_multihorizon/raw `
  --output-dir journal_results/shenzhen_multihorizon/stage2_cqr/smoke_raw `
  --use-cuda False `
  --max-calib-batches 1 `
  --max-test-batches 1
```

Smoke output confirmed:

```text
calib_quantiles = (4, 1682, 5, 13)
calib_labels    = (4, 1682, 5)
test_quantiles  = (4, 1682, 5, 13)
test_labels     = (4, 1682, 5)
```

The smoke metrics showed clear coverage improvement over raw intervals, but these values should
not be reported as experiment results because only one batch was used.

## Verification
Stage 2 unit tests and Stage 1 regression tests passed:

```text
conda run -n py12 python -m pytest tests/test_journal_stage2_cqr.py tests/test_journal_stage1_multihorizon.py -q
10 passed
```

## Success Criteria For Full Runs
For each Stage 1 input result folder:

1. `raw` PICP should show the known under-coverage.
2. `global_cqr` should substantially improve average PICP toward the nominal coverage.
3. `horizon_cqr` should reduce horizon-level ACE or worst-horizon under-coverage relative to `global_cqr`.
4. MPIW and WIS should be checked to ensure coverage improvement is not only caused by excessive interval inflation.

The primary reporting target is `delta=0.1` because it corresponds to 90 percent nominal coverage.

## Full Warm-Start Run
Full Lenovo run command:

```bash
bash scripts/journal/run_stage2_cqr_warmstart.sh
```

Full run metadata:

```text
created_at: 2026-05-01T22:53:49
branch: multi_horizon_journal
commit: ed02522
stage1_output_dir: journal_results/shenzhen_multihorizon/warmstart_raw/
output_dir: journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw/
device: cuda:0
gpu_name: NVIDIA GeForce RTX 3060 Laptop GPU
calib_quantiles_shape: [828, 1682, 5, 13]
calib_labels_shape: [828, 1682, 5]
test_quantiles_shape: [829, 1682, 5, 13]
test_labels_shape: [829, 1682, 5]
```

Summary by method and nominal coverage:

| Method | Delta | Nominal | Mean PICP | Min PICP | Max PICP | Mean ACE | Max ACE | Mean MPIW | Mean WIS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.1 | 0.9 | 0.638917 | 0.604806 | 0.657056 | 0.261083 | 0.295194 | 2.916468 | 3.016992 |
| global_cqr | 0.1 | 0.9 | 0.913388 | 0.892134 | 0.924940 | 0.016534 | 0.024940 | 2.916553 | 3.017066 |
| horizon_cqr | 0.1 | 0.9 | 0.913738 | 0.882695 | 0.930853 | 0.020660 | 0.030853 | 2.916619 | 3.017127 |
| raw | 0.2 | 0.8 | 0.588075 | 0.561845 | 0.613981 | 0.211925 | 0.238155 | 1.850629 | 1.948117 |
| global_cqr | 0.2 | 0.8 | 0.807232 | 0.787943 | 0.844638 | 0.015898 | 0.044638 | 1.850664 | 1.948143 |
| horizon_cqr | 0.2 | 0.8 | 0.807510 | 0.775446 | 0.838567 | 0.025575 | 0.038567 | 1.850663 | 1.948142 |
| raw | 0.4 | 0.6 | 0.489752 | 0.471420 | 0.510661 | 0.110248 | 0.128580 | 0.975355 | 1.142472 |
| global_cqr | 0.4 | 0.6 | 0.595050 | 0.522632 | 0.727126 | 0.061282 | 0.127126 | 0.975373 | 1.142482 |
| horizon_cqr | 0.4 | 0.6 | 0.594848 | 0.563507 | 0.626280 | 0.023555 | 0.036493 | 0.975382 | 1.142488 |

Main interpretation:

- Raw warm-start intervals are strongly under-covered at all nominal coverage levels.
- CQR corrects most of the under-coverage.
- At the primary 90 percent target, global CQR is already very strong and slightly cleaner than horizon-wise CQR by mean/max ACE.
- At 60 percent nominal coverage, horizon-wise CQR is more stable across horizons than global CQR.
- Width inflation is extremely small, so the calibration result needs a zero-boundary explanation rather than a simple "wider interval" explanation.

## Zero-Boundary Diagnostic
The CQR thresholds are tiny:

| Method | Delta | Horizon | s_hat |
|---|---:|---:|---:|
| global_cqr | 0.1 | all | 0.0000481755 |
| global_cqr | 0.2 | all | 0.0000189398 |
| global_cqr | 0.4 | all | 0.0000091120 |

This initially looks surprising because PICP improves a lot while MPIW barely changes.

The reason is that many raw misses occur at the nonnegative boundary: true demand is exactly
zero, while the learned lower quantile is slightly above zero. A very small CQR correction moves
the lower endpoint below zero, and nonnegative clipping maps it to zero. This rescues many zero
demand observations without materially increasing interval width.

For `delta = 0.1` global CQR:

| Horizon | Raw PICP | CQR PICP | Rescued Share | Rescued Zero Share | Lower Miss Share | Upper Miss Share |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.604806 | 0.892134 | 0.287328 | 0.287311 | 0.352092 | 0.043102 |
| 3 | 0.632331 | 0.918925 | 0.286594 | 0.286585 | 0.330537 | 0.037132 |
| 6 | 0.645196 | 0.924940 | 0.279744 | 0.279737 | 0.321979 | 0.032825 |
| 12 | 0.655198 | 0.917449 | 0.262252 | 0.262250 | 0.312748 | 0.032054 |
| 24 | 0.657056 | 0.913492 | 0.256436 | 0.256433 | 0.309714 | 0.033231 |

The rescued share is almost entirely zero-demand samples. This should be explicitly mentioned
in any paper or group-meeting discussion because otherwise the combination of large PICP gains
and near-zero MPIW change may look suspicious.

## Hourly Diagnostic And Adaptive Calibration Decision
Hourly diagnostics were computed using target-hour alignment:

```text
target_index = test_start + window_index + seq_len + horizon - 1
test_start = 7883
seq_len = 24
```

For `delta = 0.1`, global CQR improves the worst hourly cell dramatically but does not make all
hour-by-horizon cells identical:

| Method | Mean hourly ACE | Max hourly ACE | Min hourly PICP | Max hourly PICP | Worst under-covered cell |
|---|---:|---:|---:|---:|---|
| raw | 0.261065 | 0.363909 | 0.536091 | 0.686377 | H1 at 10:00, PICP 0.536091 |
| global_cqr | 0.025218 | 0.080178 | 0.819822 | 0.975012 | H1 at 10:00, PICP 0.819822 |
| horizon_cqr | 0.029890 | 0.088466 | 0.811534 | 0.960883 | H1 at 10:00, PICP 0.811534 |

For lower nominal coverage, horizon-wise CQR is more useful:

| Delta | Method | Mean hourly ACE | Max hourly ACE | Min hourly PICP | Max hourly PICP |
|---:|---|---:|---:|---:|---:|
| 0.2 | global_cqr | 0.027478 | 0.123815 | 0.727009 | 0.923815 |
| 0.2 | horizon_cqr | 0.037296 | 0.104230 | 0.695770 | 0.865194 |
| 0.4 | global_cqr | 0.067987 | 0.219433 | 0.478274 | 0.819433 |
| 0.4 | horizon_cqr | 0.035839 | 0.109156 | 0.490844 | 0.671700 |

Adaptive or stratified calibration is therefore not needed to "save" marginal 90 percent coverage;
global CQR already does that. Its revised role should be narrower and more honest:

```text
Use adaptive/stratified calibration to target localized reliability, worst-cell ACE,
and lower-coverage settings, not as the main mechanism for fixing marginal coverage.
```

This changes the journal framing. The next adaptive calibration experiment should be treated as
a diagnostic/worst-cell reliability extension. If it cannot improve worst-cell ACE or localized
under-coverage without unacceptable width inflation, the paper should not claim adaptive
calibration as the central contribution.
