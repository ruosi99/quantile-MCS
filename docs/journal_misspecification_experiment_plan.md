# Cost-Ratio Misspecification Experiments For The Current Journal Branch

## Metadata
- Date: 2026-05-07
- Branch: `multi_horizon_journal`
- Target machine for execution: `Lenovo`
- Authoring machine: `Dell`
- Output directory: `journal_results/shenzhen_multihorizon/misspecification/`

## Why This Exists
This document rewrites the generic misspecification experiment idea so it fits the current project state.

It is intended to answer one paper-direction question:

> If the operator does not know the exact under-provision to over-provision cost ratio, is the current journal story still stable, or does cost-ratio misspecification become important enough to shift the paper's center of gravity?

This package is not meant to reopen broad Stage 3 calibration work.
It is a decision-focused extension that builds directly on the existing Shenzhen warm-start experiment stack.

## Current Project Context
These experiments must be implemented against the outputs that already exist in this repository:

- Stage 1 warm-start forecasts:
  - `journal_results/shenzhen_multihorizon/warmstart_raw/`
- Stage 4 decision outputs:
  - `journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw/`
- reinforcement outputs:
  - `journal_results/shenzhen_multihorizon/reinforcement/`
- paper-facing evidence assets:
  - `journal_results/shenzhen_multihorizon/paper_assets/`

The current branch already established:

- multi-horizon direct quantile forecasts with horizons `H = [1, 3, 6, 12, 24]`
- quantile grid
  - `[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.833333333333, 0.9, 0.95]`
- boundary-aware reliability evidence
- decision-value attribution evidence
- reinforcement outputs for station-level zero-inflation sensitivity and deployment strategy comparison

This means the misspecification study should be positioned as a paper-direction gate, not as a foundational experiment.

## Main Research Question
The package should test the following:

1. How expensive is it if the operator chooses the wrong cost-aligned quantile because the true cost ratio is misspecified?
2. Is the regret from misspecification large enough to become a new paper-significant result?
3. If misspecification matters, which robust quantile-selection rule is preferable under cost-ratio uncertainty?
4. Does time-varying cost structure strengthen the case for multi-horizon probabilistic forecasts?

## Recommended Execution Logic
Do not treat all three phases as equally necessary.

Use this sequence:

1. Run Phase 1 first as a gate.
2. Only if Phase 1 shows practically meaningful misspecification regret, continue to Phase 2.
3. Only if Phase 1 and Phase 2 both suggest real paper value, continue to Phase 3.

This is important because the current paper already has a strong main narrative:

- marginal coverage can mislead under zero-inflation and nonnegativity
- most decision value comes from cost-aligned quantile choice
- calibration is a lightweight safeguard

The misspecification package should only move the paper center if it adds clearly new and decision-relevant evidence.

## Code To Create
- `scripts/journal/build_misspecification_experiments.py`
- `scripts/journal/run_misspecification_experiments_warmstart.sh`
- `tests/test_journal_misspecification.py`

## Execution Environment
- Conda environment: `py12`
- Python: `3.12`
- Dependencies:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`

## Required Inputs

### Stage 1 warm-start arrays
Read these files from:
`journal_results/shenzhen_multihorizon/warmstart_raw/`

- `predict_quantiles.npy`
- `label_list.npy`
- `run_metadata.json`

These are the actual project inputs. Do not assume flattened arrays named `y_true` or `q_forecasts`.

Current known shapes from `run_metadata.json`:

- `predict_quantiles.npy`: `(829, 1682, 5, 13)`
- `label_list.npy`: `(829, 1682, 5)`

Interpretation:

- axis 0: test windows
- axis 1: stations
- axis 2: horizons
- axis 3: quantiles

Horizons are:
- `[1, 3, 6, 12, 24]`

Quantile levels are:
- `[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.833333333333, 0.9, 0.95]`

### Stage 4 decision outputs
Read these files from:
`journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw/`

- `decision_summary_by_method_and_cost_ratio.csv`
- `decision_metrics_by_horizon.csv`
- `decision_regret_vs_oracle.csv`
- `decision_one_sided_thresholds.csv`
- `decision_eval_metadata.json`

These should be used for consistency checks and to reuse method naming conventions.

### Reinforcement outputs
Read these files from:
`journal_results/shenzhen_multihorizon/reinforcement/`

- `station_zero_group_summary.csv`
- `station_zero_ratio_by_station.csv`
- `station_zero_inflation_sensitivity.csv`
- `station_zero_group_decision_attribution.csv`
- `deployment_strategy_comparison.csv`
- `deployment_strategy_by_horizon.csv`
- `test_window_manifest.csv`
- `bootstrap_ci_day_level.csv`

These files now exist and should be treated as the authoritative source for:

- station group definitions
- test-window to date mapping
- current deployment strategy baselines

Do not rebuild station groups from scratch unless the script explicitly detects that the reinforcement files are missing.

### Dataset directory
Use:
`data/datasets/ST_EVCDP_v2_canonical/`

This may be needed only for optional timestamp validation. The main temporal mapping should come from:
- `journal_results/shenzhen_multihorizon/reinforcement/test_window_manifest.csv`

## Shared Loading Rules

### Tensor handling
All misspecification computations should be based on the same tensor convention already used in:
- `scripts/journal/evaluate_stage4_decision.py`
- `scripts/journal/build_paper_evidence_assets.py`

That means:

- operate from `(test_window, station, horizon, quantile)` tensors
- only flatten after building the correct masks for horizon, station group, or target hour

Do not write new indexing logic that assumes a different sample order.

### Cost ratios
The current project already uses these exact cost ratios in Stage 4:

- `1:1`
- `3:1`
- `5:1`
- `9:1`
- `19:1`

These should remain the primary grid because they align with the current quantile grid and existing decision outputs.

`2:1` may be included as an optional interpolated ratio, but it should not be promoted into the primary paper matrix by default.

Recommended default grid for Phase 1:

```python
COST_RATIO_GRID = [
    {"label": "1:1",  "r": 1.0,  "tau": 0.5},
    {"label": "3:1",  "r": 3.0,  "tau": 0.75},
    {"label": "5:1",  "r": 5.0,  "tau": 0.8333333333333334},
    {"label": "9:1",  "r": 9.0,  "tau": 0.9},
    {"label": "19:1", "r": 19.0, "tau": 0.95},
]
```

Optional expanded grid only if needed later:

```python
OPTIONAL_EXTRA_RATIOS = [
    {"label": "2:1",  "r": 2.0,  "tau": 0.666666666667},
    {"label": "7:1",  "r": 7.0,  "tau": 0.875},
    {"label": "12:1", "r": 12.0, "tau": 0.923076923077},
]
```

## CLI Interface
The new script should accept:

```text
--stage1-output-dir
--stage4-output-dir
--reinforcement-dir
--data-dir
--output-dir
--phases
--include-interpolated-ratios
--bootstrap-samples
--bootstrap-seed
--machine
```

Recommended defaults:

- `--stage1-output-dir journal_results/shenzhen_multihorizon/warmstart_raw`
- `--stage4-output-dir journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw`
- `--reinforcement-dir journal_results/shenzhen_multihorizon/reinforcement`
- `--data-dir data/datasets/ST_EVCDP_v2_canonical`
- `--output-dir journal_results/shenzhen_multihorizon/misspecification`
- `--phases 1`
- `--include-interpolated-ratios false`
- `--bootstrap-samples 300`
- `--bootstrap-seed 20260507`
- `--machine Lenovo`

Defaulting to `--phases 1` is deliberate. Phase 1 is the gate.

## Phase 1: Cost-Ratio Misspecification Matrix

### Purpose
Measure the decision regret incurred when the operator chooses a quantile based on an assumed cost ratio, but the true cost ratio is different.

### Project-specific method choice
Use the raw cost-aligned quantile decisions as the base decision rule.

For Phase 1, do not make one-sided calibration the main object.
The current paper direction already showed that calibration has tiny incremental decision gain.

So the misspecification matrix should primarily answer:

> If the operator picks the wrong raw target quantile because the cost ratio is wrong, how much regret follows?

Optional extension:
- repeat the same matrix for `horizon_cqr` one-sided refined decisions only if the raw matrix turns out to be important

### Decision construction
For each assumed cost ratio:

1. map ratio to `tau = r / (1 + r)`
2. extract or interpolate the corresponding quantile from `predict_quantiles.npy`
3. clip decisions to be nonnegative
4. evaluate cost under each true cost ratio

### Aggregation unit
Use the same atomic unit as Stage 4:

- sample-window × station × horizon

Then average costs over all included atomic observations.

### Required Phase 1 outputs
- aggregate cost matrix
- aggregate regret matrix
- aggregate percentage regret matrix
- by-horizon long-format matrix
- by-station-group long-format matrix
- asymmetry summary
- selected bootstrap confidence intervals
- aggregate regret heatmap
- by-horizon heatmap panel
- asymmetry bar chart

### Station-group decomposition
Use the existing reinforcement station groups rather than redefining them.

Expected source:
- `station_zero_group_summary.csv`
- `station_zero_ratio_by_station.csv`

If the file layout contains per-station group assignments separately, use those directly.
If `station_zero_group_summary.csv` is only a summary table, then derive the actual station masks from `station_zero_ratio_by_station.csv`.

### Bootstrap convention
Use day-level bootstrap based on:
- `reinforcement/test_window_manifest.csv`

This is better than the current paper-assets test-window bootstrap and is now available in-project.

### Gate metric
Use the aggregate percentage regret matrix as the primary gate.

Recommended decision rule:

- `PASSED` if max off-diagonal percentage regret > `10%`
- `MARGINAL` if max off-diagonal percentage regret is between `5%` and `10%`
- `FAILED` if max off-diagonal percentage regret < `5%`

Interpretation:

- `PASSED`: misspecification is large enough to justify a paper-facing extension
- `MARGINAL`: useful as a smaller robustness/decision subsection
- `FAILED`: do not shift the paper center; keep this as appendix or omit

### Additional paper-facing statistic
Compute underestimate-vs-overestimate asymmetry.

This is especially relevant if:
- assuming too small a ratio is systematically more expensive than assuming too large a ratio

If that pattern is strong, it gives a cleaner operational claim than the raw matrix alone.

## Phase 2: Robust Quantile Selection Under Cost-Ratio Uncertainty

### Precondition
Run only if Phase 1 is `PASSED` or meaningfully `MARGINAL`.

Current Lenovo Phase 1 result:

- gate status: `PASSED`
- max off-diagonal percentage regret: `206.249%`
- output directory: `journal_results/shenzhen_multihorizon/misspecification/`

This means Phase 2 is now justified.

### Purpose
If exact cost ratios are not known, compare simple robust selection rules for choosing one deployed quantile.

Phase 2 answers:

> If the operator only knows the cost ratio lies in a plausible range, which quantile-selection rule minimizes worst-case or expected regret?

### What Phase 2 does not do
- It does not retrain the model.
- It does not introduce new quantile levels.
- It does not use interpolation by default.
- It does not use calibration or CQR.
- It does not rebuild station groups or bootstrap logic.
- It operates on Phase 1 cost and regret matrices.

### Required Phase 2 inputs
Read these already-generated Phase 1 outputs from:
`journal_results/shenzhen_multihorizon/misspecification/`

- `misspecification_cost_matrix.csv`
- `misspecification_regret_matrix.csv`
- `misspecification_pct_regret_matrix.csv`
- `misspecification_by_horizon.csv`

Actual Phase 1 column names are:

- matrix files:
  - first column: `assumed_cost_ratio`
  - remaining columns: true cost-ratio labels
- by-horizon file:
  - `horizon`
  - `assumed_cost_ratio`
  - `true_cost_ratio`
  - `expected_cost`
  - `regret_vs_true_aligned`
  - `pct_regret_vs_true_aligned`

### Recommended scenarios
Keep the scenario set small and tied to the current project grid:

```python
UNCERTAINTY_SCENARIOS = [
    {
        "name": "narrow",
        "r_low": 5.0,
        "r_high": 9.0,
        "valid_grid_indices": [2, 3],  # 5:1, 9:1
    },
    {
        "name": "medium",
        "r_low": 3.0,
        "r_high": 19.0,
        "valid_grid_indices": [1, 2, 3, 4],  # 3:1, 5:1, 9:1, 19:1
    },
    {
        "name": "wide",
        "r_low": 1.0,
        "r_high": 19.0,
        "valid_grid_indices": [0, 1, 2, 3, 4],  # all five ratios
    },
]
```

`valid_grid_indices` defines the true cost-ratio uncertainty set.

Important design decision:

- true cost ratios are restricted to the scenario range
- the deployed assumed ratio can be any entry in the full five-point grid

This follows the robust-optimization interpretation: the uncertainty set constrains the true state, while the operator's action set remains the full deployed quantile grid.

### Recommended strategies
- midpoint
- conservative
- minimax_regret
- expected_cost

Use the following exact definitions:

#### `midpoint`
Choose the grid ratio whose `r` value is closest to the arithmetic midpoint of `[r_low, r_high]`.

Tie-breaking rule:

- if two grid entries are equally close, choose the higher `r`

Expected choices:

- narrow: midpoint 7.0, tie between `5:1` and `9:1`, choose `9:1`
- medium: midpoint 11.0, choose `9:1`
- wide: midpoint 10.0, choose `9:1`

#### `conservative`
Choose the highest cost ratio inside the scenario range.

Expected choices:

- narrow: `9:1`
- medium: `19:1`
- wide: `19:1`

#### `minimax_regret`
For each possible deployed ratio in the full five-point grid, compute the worst absolute regret across the true ratios in the scenario range.

Choose the deployed ratio with the smallest worst-case absolute regret.

Use `misspecification_regret_matrix.csv`, not the percentage-regret matrix, for the strategy selection itself.

#### `expected_cost`
For each possible deployed ratio in the full five-point grid, compute the mean expected cost across the true ratios in the scenario range.

Assume a discrete uniform prior over the grid ratios inside the scenario range.

Choose the deployed ratio with the smallest mean expected cost.

Use `misspecification_cost_matrix.csv` for the strategy selection.

### Important project constraint
Strategies should snap to the current cost-ratio grid by default.
Do not introduce dense new interpolated ratio grids unless Phase 1 shows this is necessary.

For the official Phase 2 run, keep:

- `--include-interpolated-ratios false`
- no optional `2:1`, `7:1`, or `12:1` entries

### Required Phase 2 outputs
- robust strategy comparison summary
- by-horizon robust strategy summary
- grouped bar figure

Concrete output files:

- `robust_strategy_comparison.csv`
- `robust_strategy_comparison_by_horizon.csv`
- `robust_strategy_comparison_chart.png`

`robust_strategy_comparison.csv` should have one row per scenario-strategy pair:

- 3 scenarios x 4 strategies = 12 rows

Required columns:

- `scenario`
- `strategy`
- `chosen_ratio_label`
- `chosen_ratio_r`
- `chosen_tau`
- `mean_cost`
- `worst_case_cost`
- `best_case_cost`
- `mean_regret`
- `worst_case_regret`
- `mean_pct_regret`
- `worst_case_pct_regret`
- `mean_oracle_cost`
- `worst_true_cost_ratio`

`robust_strategy_comparison_by_horizon.csv` should repeat the same evaluation per horizon:

- 5 horizons x 3 scenarios x 4 strategies = 60 rows

The figure should plot:

- x-axis: uncertainty scenario
- bars: strategies
- y-axis: `worst_case_pct_regret`
- annotation: selected `chosen_ratio_label`

### Evaluation procedure
For each scenario and strategy:

1. select one deployed assumed ratio
2. evaluate that row across the scenario's valid true-ratio columns
3. compute:
   - mean and worst-case expected cost
   - mean and worst-case absolute regret
   - mean and worst-case percentage regret
   - mean oracle cost from the corresponding diagonal entries
   - worst true ratio for the selected strategy

For by-horizon analysis, reconstruct 5x5 matrices from `misspecification_by_horizon.csv` using the actual Phase 1 column names listed above.

### Phase 2 success criteria
Phase 2 is paper-worthy if at least one of the following is true:

1. `minimax_regret` or `expected_cost` selects a different ratio than simple midpoint/conservative heuristics.
2. The worst-case percentage regret gap between the best and worst strategy is large, for example greater than 20 percentage points in the wide scenario.
3. The preferred strategy changes across horizons, suggesting horizon-specific robust deployment choices.

If all strategies converge to nearly the same choices, Phase 2 should be framed as a simple robustness result rather than a major new paper direction.

### Why Phase 2 is conditional
If Phase 1 regret is small, Phase 2 will not change the paper meaningfully.
In that case this should not be built.

## Phase 3: Time-Varying Cost-Ratio Scenarios

### Precondition
Run only if:
- Phase 1 is `PASSED` or strong `MARGINAL`
- and Phase 2 suggests robust strategy differences are nontrivial

### Purpose
Show whether varying future cost structure across target hours creates a genuinely multi-horizon decision advantage.

### Why this is the most speculative phase
This phase is the furthest from the current validated paper center.

It is only worth doing if we learn from Phase 1 that:
- misspecification regret is material
- and from Phase 2 that operator uncertainty meaningfully changes the preferred quantile rule

### Time mapping requirement
This phase must use:
- `reinforcement/test_window_manifest.csv`

to map each test window and horizon to the actual target hour.

Do not infer this loosely from array order.

### Recommended profiles
Keep the same three profile families as the generic plan:
- flat baseline
- peak-sensitive profile
- moderate variation profile

But treat them as demonstration scenarios, not as a core paper claim unless the effects are large.

### Required Phase 3 outputs
- profile definitions
- strategy comparison summary
- hourly breakdown
- multi-horizon demonstration table
- cost-profile figure
- strategy comparison figure
- hourly breakdown figure

## Recommended Project Positioning Of The Whole Package

### If Phase 1 fails
Do not move the paper center.

Use the result, at most, as:
- one short appendix robustness statement:
  - cost-aligned quantile decisions are naturally robust to moderate cost-ratio misspecification on Shenzhen

### If Phase 1 is marginal
Keep the current paper main line unchanged.

Possible use:
- one short subsection in Results/Discussion
- optional Phase 2 only
- no need for Phase 3 unless the story becomes much stronger than expected

### If Phase 1 passes strongly
Then the paper may gain a new decision-robustness subsection:

- quantile choice is the main value generator
- but that value is sensitive to cost-ratio misspecification
- therefore robust quantile selection becomes operationally relevant

Even then, this should still be a decision-focused extension, not a replacement for the current boundary-aware reliability narrative.

## Required Test Coverage
At minimum:

- exact quantile extraction
- interpolation between quantiles
- asymmetric cost computation
- diagonal regret is zero
- regret is nonnegative up to numerical tolerance
- station-group filtering preserves counts
- day-level bootstrap resamples valid days from `test_window_manifest.csv`
- time-varying target-hour mapping is consistent with horizon offsets

## Required Output Files

### Phase 1
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

### Phase 2
- `robust_strategy_comparison.csv`
- `robust_strategy_comparison_by_horizon.csv`
- `robust_strategy_comparison_chart.png`

### Phase 3
- `time_varying_profile_definitions.csv`
- `time_varying_strategy_comparison.csv`
- `time_varying_hourly_breakdown.csv`
- `time_varying_multi_horizon_demo.csv`
- `time_varying_cost_profile_and_quantile.png`
- `time_varying_strategy_comparison_bar.png`
- `time_varying_hourly_breakdown.png`

### Metadata
- `misspecification_metadata.json`

## Full Run Command Template

### Gate-only default run
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

### Phase 2 robust-selection run after Phase 1 passes
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

### Full run only if Phase 1 justifies it
```powershell
conda run -n py12 python scripts/journal/build_misspecification_experiments.py `
  --stage1-output-dir journal_results/shenzhen_multihorizon/warmstart_raw `
  --stage4-output-dir journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw `
  --reinforcement-dir journal_results/shenzhen_multihorizon/reinforcement `
  --data-dir data/datasets/ST_EVCDP_v2_canonical `
  --output-dir journal_results/shenzhen_multihorizon/misspecification `
  --phases 1,2,3 `
  --bootstrap-samples 300 `
  --bootstrap-seed 20260507 `
  --machine Lenovo
```

## Verification Command
```powershell
conda run -n py12 python -m pytest tests/test_journal_misspecification.py -q -p no:cacheprovider
```

## Final Recommendation Before Lenovo Execution
Start with Phase 1 only.

The current paper already has a coherent main narrative. The misspecification package should only be allowed to expand the paper if the Phase 1 regret matrix shows a real operational vulnerability.

So the practical Lenovo order should be:

1. implement script and tests
2. run `--phases 1`
3. inspect the gate result
4. only then decide whether to launch `--phases 1,2,3`
