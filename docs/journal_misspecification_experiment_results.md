# Cost-Ratio Misspecification Experiment Results

## Metadata
- Date: 2026-05-07
- Machine: Lenovo
- Branch: `multi_horizon_journal`
- Commit at execution: `5fdc1e4`
- Phase executed: Phase 1 only
- Output directory: `journal_results/shenzhen_multihorizon/misspecification/`

## Code Changes
This run added the Phase 1 misspecification gate implementation:

- `scripts/journal/build_misspecification_experiments.py`
- `scripts/journal/run_misspecification_experiments_warmstart.sh`
- `tests/test_journal_misspecification.py`

The script reads the existing warm-start Stage 1 forecast tensors, Stage 4 decision outputs, and reinforcement station/date mappings. It keeps the same tensor convention as the previous journal scripts:

- `predict_quantiles.npy`: `(test_window, station, horizon, quantile)`
- `label_list.npy`: `(test_window, station, horizon)`

Phase 1 evaluates raw cost-aligned quantile decisions under cost-ratio misspecification:

- rows: assumed cost ratio used to choose the deployed quantile
- columns: true cost ratio used to evaluate asymmetric newsvendor cost
- regret baseline: the correctly aligned raw quantile for the true cost ratio

The phase uses the primary project grid:

- `1:1`, `3:1`, `5:1`, `9:1`, `19:1`

No extra interpolated ratios were enabled for the official Phase 1 gate run.

## Verification
The new unit tests passed:

```powershell
conda run -n py12 python -m pytest tests/test_journal_misspecification.py -q -p no:cacheprovider
```

Result:

- `5 passed`
- one existing `torch_geometric` deprecation warning

## Run Command
```powershell
conda run -n py12 python scripts/journal/build_misspecification_experiments.py `
  --stage1-output-dir journal_results/shenzhen_multihorizon/warmstart_raw `
  --stage4-output-dir journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw `
  --reinforcement-dir journal_results/shenzhen_multihorizon/reinforcement `
  --data-dir data/datasets/ST_EVCDP_v2_canonical `
  --output-dir journal_results/shenzhen_multihorizon/misspecification `
  --phases 1 `
  --bootstrap-samples 300 `
  --bootstrap-seed 20260507 `
  --machine Lenovo
```

## Generated Files
- `misspecification_cost_matrix.csv`
- `misspecification_regret_matrix.csv`
- `misspecification_pct_regret_matrix.csv`
- `misspecification_by_horizon.csv`
- `misspecification_by_station_group.csv`
- `misspecification_asymmetry.csv`
- `misspecification_bootstrap_ci.csv`
- `misspecification_regret_heatmap.png`
- `misspecification_by_horizon_heatmap.png`
- `misspecification_asymmetry_bar.png`
- `misspecification_metadata.json`

## Gate Result
Phase 1 gate result: `PASSED`.

Gate rule from the plan:

- `PASSED` if max off-diagonal percentage regret > 10%
- `MARGINAL` if 5%-10%
- `FAILED` if < 5%

Observed maximum off-diagonal percentage regret:

- `206.249%`
- assumed cost ratio: `19:1`
- true cost ratio: `1:1`
- direction: overestimate

Minimum off-diagonal percentage regret:

- `-3.777%`
- this occurs because in a small number of empirical cells the wrong quantile can slightly outperform the nominal true-aligned quantile, reflecting forecast/model imperfection rather than a theoretical violation.

## Aggregate Percentage Regret Matrix
Rows are assumed ratios; columns are true ratios.

| assumed | 1:1 | 3:1 | 5:1 | 9:1 | 19:1 |
|---|---:|---:|---:|---:|---:|
| 1:1 | 0.000 | 20.413 | 42.041 | 79.898 | 157.927 |
| 3:1 | 21.124 | 0.000 | 0.483 | 9.797 | 38.835 |
| 5:1 | 49.625 | 8.072 | 0.000 | -0.814 | 12.983 |
| 9:1 | 108.043 | 34.188 | 13.947 | 0.000 | -3.777 |
| 19:1 | 206.249 | 86.531 | 50.565 | 21.268 | 0.000 |

## Asymmetry Summary
The misspecification effect is large in both directions.

| direction | mean pct regret | max pct regret | positive regret share |
|---|---:|---:|---:|
| overestimate | 59.961 | 206.249 | 1.000 |
| underestimate | 35.779 | 157.927 | 0.800 |

The largest loss is overestimating cost asymmetry: choosing a high quantile such as `q0.95` when the true cost ratio is symmetric `1:1` leads to severe over-provision cost.

Underestimating asymmetry is also important: using `q0.5` when the true ratio is `19:1` causes `157.927%` aggregate percentage regret.

## Horizon Decomposition
The largest by-horizon regret cells are:

- H24, assumed `19:1`, true `1:1`: `258.684%`
- H12, assumed `19:1`, true `1:1`: `224.169%`
- H6, assumed `19:1`, true `1:1`: `184.160%`
- H12, assumed `1:1`, true `19:1`: `169.892%`
- H3, assumed `19:1`, true `1:1`: `166.840%`

This suggests the misspecification issue is not confined to H1. It is at least as visible at longer horizons, especially when a high target quantile is used under a truly symmetric cost setting.

## Station-Group Decomposition
The station-group table shows very high percentage regret in ultra-sparse stations, but the absolute cost scale there can be small.

Largest observed station-group percentage regrets include:

- `G4_high_zero`, assumed `19:1`, true `1:1`: `1186.163%`
- `G4_high_zero`, assumed `19:1`, true `3:1`: `451.279%`
- `G4_high_zero`, assumed `9:1`, true `1:1`: `320.758%`
- `G1_low_zero`, assumed `19:1`, true `1:1`: `300.656%`

Interpretation:

- high-zero stations amplify percentage regret because the true-aligned baseline cost is tiny
- low-zero stations still show large regret when cost asymmetry is heavily overestimated
- the effect is therefore not only a zero-inflation artifact

## Day-Level Bootstrap
The day-level bootstrap used 36 target-date units from `reinforcement/test_window_manifest.csv` and 300 samples.

Top percentage-regret CIs:

| assumed | true | direction | point | 95% CI |
|---|---|---|---:|---:|
| 19:1 | 1:1 | overestimate | 209.602 | [199.951, 221.040] |
| 1:1 | 19:1 | underestimate | 161.181 | [151.821, 169.865] |
| 9:1 | 1:1 | overestimate | 109.728 | [103.386, 115.581] |
| 19:1 | 3:1 | overestimate | 88.913 | [83.356, 94.595] |
| 1:1 | 9:1 | underestimate | 81.718 | [76.735, 88.399] |

The confidence intervals are far above the 10% gate threshold, so the Phase 1 pass is stable rather than noise from a few windows.

## Interpretation
Phase 1 changes the paper-direction assessment.

Before this run, cost-ratio misspecification was only a possible robustness concern. The result now shows it is operationally large:

- cost-aligned quantile choice is valuable, but it is sensitive to the assumed cost ratio
- both underestimating and overestimating the ratio can be costly
- overestimating asymmetry is especially damaging when the true ratio is close to symmetric
- the effect survives day-level bootstrap

This supports continuing to Phase 2, but it does not require broad Stage 3 calibration work.

## Recommended Next Step
Proceed to Phase 2 only after review/approval:

- compare robust quantile-selection rules under cost-ratio uncertainty
- keep strategies snapped to the existing cost-ratio grid
- do not start Phase 3 unless Phase 2 shows meaningful strategy differences

## Phase 2 Update: Robust Quantile Selection
Phase 2 was implemented and executed on Lenovo on `2026-05-07`.

Additional code changes:

- `scripts/journal/build_misspecification_experiments.py` now supports `--phases 2`
- `scripts/journal/run_misspecification_experiments_warmstart.sh` now accepts `MISSPECIFICATION_PHASES`, defaulting to `1`
- `tests/test_journal_misspecification.py` now covers Phase 2 strategy rules and by-horizon matrix reconstruction

The experiment follows the actual Phase 1 output schema:

- `assumed_cost_ratio`
- `true_cost_ratio`
- `expected_cost`
- `regret_vs_true_aligned`
- `pct_regret_vs_true_aligned`

Verification after Phase 2 changes:

```powershell
conda run -n py12 python -m pytest tests/test_journal_misspecification.py -q -p no:cacheprovider
```

Result:

- `11 passed`
- one existing `torch_geometric` deprecation warning

### Phase 2 Run Command
```powershell
conda run -n py12 python scripts/journal/build_misspecification_experiments.py `
  --stage1-output-dir journal_results/shenzhen_multihorizon/warmstart_raw `
  --stage4-output-dir journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw `
  --reinforcement-dir journal_results/shenzhen_multihorizon/reinforcement `
  --data-dir data/datasets/ST_EVCDP_v2_canonical `
  --output-dir journal_results/shenzhen_multihorizon/misspecification `
  --phases 2 `
  --bootstrap-samples 300 `
  --bootstrap-seed 20260507 `
  --machine Lenovo
```

Phase 2 reads the Phase 1 matrices only. It does not reload the raw forecast tensors, retrain the model, interpolate new ratios, or use calibration/CQR.

### Additional Phase 2 Outputs
- `robust_strategy_comparison.csv`
- `robust_strategy_comparison_by_horizon.csv`
- `robust_strategy_comparison_chart.png`

### Aggregate Phase 2 Results
Worst-case percentage regret by scenario and strategy:

| scenario | strategy | chosen ratio | worst-case pct regret | mean pct regret | worst true ratio |
|---|---|---:|---:|---:|---:|
| narrow | minimax_regret | 5:1 | 0.000 | -0.407 | 5:1 |
| narrow | expected_cost | 5:1 | 0.000 | -0.407 | 5:1 |
| narrow | midpoint | 9:1 | 13.947 | 6.973 | 5:1 |
| narrow | conservative | 9:1 | 13.947 | 6.973 | 5:1 |
| medium | minimax_regret | 5:1 | 12.983 | 5.060 | 19:1 |
| medium | expected_cost | 5:1 | 12.983 | 5.060 | 19:1 |
| medium | midpoint | 9:1 | 34.188 | 11.089 | 3:1 |
| medium | conservative | 19:1 | 86.531 | 39.591 | 3:1 |
| wide | minimax_regret | 5:1 | 49.625 | 13.973 | 1:1 |
| wide | expected_cost | 5:1 | 49.625 | 13.973 | 1:1 |
| wide | midpoint | 9:1 | 108.043 | 30.480 | 1:1 |
| wide | conservative | 19:1 | 206.249 | 72.923 | 1:1 |

Main result:

- `minimax_regret` and `expected_cost` both choose `5:1` in all aggregate scenarios
- this is the best worst-case choice in all three scenarios
- midpoint and conservative heuristics can be much worse
- the wide-scenario gap between best and worst strategy is `156.624` percentage points

### By-Horizon Results
The best worst-case strategy is horizon-dependent in the medium scenario:

- H1 medium: midpoint/minimax/expected all choose `9:1`
- H3 medium: midpoint/minimax/expected all choose `9:1`
- H6 medium: midpoint/minimax/expected all choose `9:1`
- H12 medium: minimax/expected choose `5:1`
- H24 medium: minimax/expected choose `5:1`

For the wide scenario, `minimax_regret`/`expected_cost` choose `5:1` across all horizons, with worst-case percentage regret rising at longer horizons:

- H1: `38.602%`
- H3: `39.252%`
- H6: `46.586%`
- H12: `56.065%`
- H24: `56.391%`

### Interpretation After Phase 2
Phase 2 is paper-relevant.

It shows that after Phase 1 identifies cost-ratio misspecification as material, a simple robust rule can reduce the worst-case exposure substantially:

- wide scenario conservative `19:1`: `206.249%` worst-case pct regret
- wide scenario midpoint `9:1`: `108.043%`
- wide scenario minimax/expected `5:1`: `49.625%`

This supports a decision-robustness subsection:

- cost-aligned quantile selection is valuable
- but exact cost-ratio assumptions are risky
- robust selection over plausible cost-ratio ranges can materially reduce worst-case regret

Phase 2 still does not require broad Stage 3 calibration. Phase 3 should remain conditional and should only be considered if a time-varying cost profile is needed for a specific paper claim.
