# Journal Reinforcement Experiments

## Metadata
- Date: 2026-05-06
- Machine: Lenovo, Windows GPU experiment machine
- Branch: `multi_horizon_journal`
- Commit at run time: `57dcf8a`
- Output directory: `journal_results/shenzhen_multihorizon/reinforcement/`

## Purpose
This reinforcement package strengthens the current journal narrative without reopening broad Stage 3.

The package covers four additions:

1. station-level zero-inflation sensitivity
2. deployment strategy comparison
3. day-level bootstrap confidence intervals
4. paper-facing deployment diagnostic flowchart

No Stage 3 localized or broad stratified calibration was implemented.

## Code Added
- `scripts/journal/build_reinforcement_experiments.py`
- `scripts/journal/run_reinforcement_experiments_warmstart.sh`
- `tests/test_journal_reinforcement.py`

## Verification
Test command:

```powershell
conda run -n py12 python -m pytest tests/test_journal_reinforcement.py tests/test_journal_paper_assets.py tests/test_journal_stage4_decision.py -q -p no:cacheprovider
```

Result:

- `13 passed`
- `1 warning`

## Full Lenovo Command
```powershell
conda run -n py12 python scripts/journal/build_reinforcement_experiments.py --stage1-output-dir journal_results/shenzhen_multihorizon/warmstart_raw --stage2-output-dir journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw --stage4-output-dir journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw --data-dir data/datasets/ST_EVCDP_v2_canonical --output-dir journal_results/shenzhen_multihorizon/reinforcement --deltas 0.1 --primary-delta 0.1 --cost-ratios 1:1,3:1,5:1,9:1,19:1 --station-groups 4 --bootstrap-samples 300 --bootstrap-seed 20260506 --machine Lenovo
```

## Generated Files
- `station_zero_ratio_by_station.csv`
- `station_zero_group_summary.csv`
- `station_zero_inflation_sensitivity.csv`
- `station_zero_ratio_vs_zbr_share.csv`
- `station_zero_group_decision_attribution.csv`
- `deployment_strategy_comparison.csv`
- `deployment_strategy_by_horizon.csv`
- `test_window_manifest.csv`
- `bootstrap_ci_day_level.csv`
- `station_zero_ratio_vs_zbr_share.png`
- `station_group_decision_attribution.png`
- `deployment_strategy_cost_bar.png`
- `deployment_diagnostic_flowchart.png`
- `reinforcement_metadata.json`

## Experiment 1: Station-Level Zero-Inflation Sensitivity

### Design
Stations were grouped by station-level zero ratio computed from the Stage 1B warm-start test labels.

Four quantile-based station groups were used:

| Group | Station count | Min zero ratio | Mean zero ratio | Max zero ratio |
|---|---:|---:|---:|---:|
| `G1_low_zero` | 421 | 0.000000 | 0.000000 | 0.000000 |
| `G2_mid_zero` | 420 | 0.000000 | 0.006270 | 0.039807 |
| `G3_mid_zero` | 420 | 0.040772 | 0.247664 | 0.761641 |
| `G4_high_zero` | 421 | 0.762364 | 0.990316 | 1.000000 |

### Result Summary
Global CQR, 90 percent nominal coverage:

| Group | Mean zero ratio | Mean PICP all | Mean PICP positive | Mean ZBR share | Min positive PICP |
|---|---:|---:|---:|---:|---:|
| `G1_low_zero` | 0.000000 | 0.978477 | 0.978477 | 0.000000 | 0.956795 |
| `G2_mid_zero` | 0.006270 | 0.936126 | 0.941744 | 0.938029 | 0.907473 |
| `G3_mid_zero` | 0.247664 | 0.766331 | 0.845687 | 0.999823 | 0.741535 |
| `G4_high_zero` | 0.990316 | 0.972324 | 0.589354 | 1.000000 | 0.426819 |

### Interpretation
The station-level analysis clarifies the earlier aggregate result.

- `ZBR_share` near 1 does not mean the full dataset is almost all zero.
- Boundary rescue is concentrated in sparse station regimes.
- Low-zero stations have no meaningful zero-boundary rescue and already show strong positive-demand coverage.
- Mid-zero and high-zero stations reveal the real boundary effect.
- The highest-zero group has very low positive-demand coverage, but positive events are rare and absolute decision cost is small.

This strengthens the paper narrative because the boundary effect is not presented as a blanket claim over all stations; it is a station-sparsity-sensitive reliability phenomenon.

## Experiment 2: Deployment Strategy Comparison

### Strategies
Four deployment strategies were compared:

- `median_decision`: use raw `q0.5`
- `symmetric_interval_upper_bound`: use the 90 percent global-CQR upper interval bound, `q0.95 + s_hat`
- `raw_cost_aligned_quantile`: use the raw target quantile implied by the cost ratio
- `one_sided_refined_cost_aligned_quantile`: use the cost-aligned target quantile plus horizon-wise one-sided conformal refinement

### Result Summary

| Strategy | Cost ratio | Expected cost | Reduction vs median | Reduction vs symmetric upper |
|---|---|---:|---:|---:|
| `median_decision` | `3:1` | 0.868041 | 0.000000 | 0.476678 |
| `symmetric_interval_upper_bound` | `3:1` | 1.344719 | -0.476678 | 0.000000 |
| `raw_cost_aligned_quantile` | `3:1` | 0.720889 | 0.147152 | 0.623830 |
| `one_sided_refined_cost_aligned_quantile` | `3:1` | 0.720752 | 0.147288 | 0.623966 |
| `median_decision` | `9:1` | 2.200118 | 0.000000 | -0.717003 |
| `symmetric_interval_upper_bound` | `9:1` | 1.483115 | 0.717003 | 0.000000 |
| `raw_cost_aligned_quantile` | `9:1` | 1.222980 | 0.977138 | 0.260135 |
| `one_sided_refined_cost_aligned_quantile` | `9:1` | 1.222530 | 0.977588 | 0.260585 |
| `median_decision` | `19:1` | 4.420247 | 0.000000 | -2.706471 |
| `symmetric_interval_upper_bound` | `19:1` | 1.713776 | 2.706471 | 0.000000 |
| `raw_cost_aligned_quantile` | `19:1` | 1.713762 | 2.706485 | 0.000014 |
| `one_sided_refined_cost_aligned_quantile` | `19:1` | 1.712642 | 2.707605 | 0.001134 |

### Interpretation
This experiment turns the Stage 4 attribution result into a deployment-facing statement.

- Using a symmetric upper interval bound is not a generally optimal deployment rule.
- At low-to-moderate asymmetric costs, the symmetric upper bound over-orders and is worse than the cost-aligned quantile.
- At `19:1`, the symmetric upper bound is effectively aligned with `q0.95`, so it becomes close to the cost-aligned strategy.
- One-sided refinement remains a small but consistent cost refiner.

Paper-facing message:

> Calibration intervals are useful for reliability, but the deployed decision rule must still be aligned with the operating cost ratio.

## Experiment 3: Day-Level Bootstrap Upgrade

### Design
The script recovered target timestamps from `duration.csv` and the Stage 1 split contract.

The test-window target dates span:

- first target date: `2023-07-27`
- last target date: `2023-08-31`
- unique target dates: `36`

This enabled day-level bootstrap instead of only test-window bootstrap.

### Selected Day-Level Confidence Intervals

| Metric | Cost ratio | Point | 95 percent CI |
|---|---|---:|---:|
| raw reduction vs median | `3:1` | 0.148920 | [0.138788, 0.162179] |
| raw reduction vs median | `5:1` | 0.391273 | [0.365930, 0.413714] |
| raw reduction vs median | `9:1` | 0.982100 | [0.932793, 1.035425] |
| raw reduction vs median | `19:1` | 2.716846 | [2.594846, 2.835888] |
| refined reduction vs symmetric | `3:1` | 0.627075 | [0.598053, 0.658343] |
| refined reduction vs symmetric | `9:1` | 0.262503 | [0.244510, 0.285202] |
| one-sided gain vs raw | `3:1` | 0.000136 | [0.000129, 0.000143] |
| one-sided gain vs raw | `19:1` | 0.001142 | [0.000909, 0.001413] |
| quantile-choice share | `3:1` | 0.999037 | [0.998971, 0.999114] |
| quantile-choice share | `19:1` | 0.999562 | [0.999451, 0.999665] |

### Interpretation
The day-level bootstrap supports the earlier test-window bootstrap conclusions.

- The cost-aligned quantile improvement over median remains stable.
- The one-sided calibration gain remains positive but tiny.
- Quantile-choice share remains essentially `0.999+` for asymmetric ratios.

## Experiment 4: Deployment Diagnostic Flowchart

### Output
The script generated:

- `deployment_diagnostic_flowchart.png`

### Flow
The flowchart encodes the current paper recommendation:

1. train the multi-horizon probabilistic forecaster
2. run boundary-aware reliability diagnostics
3. choose the cost-aligned target quantile
4. apply lightweight one-sided calibration
5. use targeted local calibration only for diagnostically identified hard cells

### Interpretation
This packages the empirical findings into a practical deployment protocol.

It also preserves the current decision:

- do not run broad Stage 3 by default
- use localized calibration only if hard positive-demand cells must be specifically addressed

## Overall Summary
The reinforcement package strengthens the current paper direction.

Key findings:

- Station-level sparsity matters: zero-boundary rescue is not universal, but it dominates in sparse station regimes.
- Low-zero stations already have strong positive-demand reliability.
- High-zero stations expose boundary behavior and rare positive-demand undercoverage, but their absolute decision costs are small.
- Deployment should not rely on symmetric interval upper bounds unless the operating cost ratio really corresponds to the upper quantile.
- Cost-aligned quantile selection remains the main value generator.
- One-sided calibration is a lightweight, stable, but small decision-cost refiner.
- Day-level bootstrap confirms that the main decision-value conclusion is stable.

Recommended next step:

- polish figures/tables for the paper
- decide whether the high-zero rare-positive behavior needs a small discussion or appendix table
- keep broad Stage 3 out of the main path unless a targeted H1 positive-demand experiment becomes necessary

