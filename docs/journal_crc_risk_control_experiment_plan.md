# Conformal Risk-Controlled Quantile Deployment For The Current Journal Branch

## Metadata
- Date: 2026-05-14
- Branch: `multi_horizon_journal`
- Authoring machine: `Dell`
- Target execution machine: `Lenovo`
- Planned output directory:
  - `journal_results/shenzhen_multihorizon/crc_risk_control/`

## Purpose
This document rewrites the CRC experiment idea so it matches the current repository state, result folders, and no-leakage requirements.

The experiment should answer:

> Can conformal risk control provide a credible deployment layer when operators trust service-risk targets more than exact monetary cost ratios?

This is not intended to replace the current paper center by default.
It is a decision-layer comparison experiment that should be evaluated against the already established story:

- boundary-aware reliability diagnostics
- decision-value attribution
- deployment strategy comparison
- misspecification and robust quantile selection

## Current Project State Relevant To CRC

### Existing result folders
The current branch already has these relevant outputs:

- `journal_results/shenzhen_multihorizon/warmstart_raw/`
- `journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw/`
- `journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw/`
- `journal_results/shenzhen_multihorizon/reinforcement/`
- `journal_results/shenzhen_multihorizon/paper_assets/`
- `journal_results/shenzhen_multihorizon/misspecification/`

### Existing scripts that should be reused conceptually
- `scripts/journal/train_multihorizon_raw.py`
- `scripts/journal/calibrate_multihorizon_cqr.py`
- `scripts/journal/evaluate_stage4_decision.py`
- `scripts/journal/build_paper_evidence_assets.py`
- `scripts/journal/build_reinforcement_experiments.py`

### Current saved Stage 1 arrays
Available directly on disk:

- `predict_quantiles.npy`
- `label_list.npy`
- `run_metadata.json`

from:
- `journal_results/shenzhen_multihorizon/warmstart_raw/`

Known shapes from current metadata:

- `predict_quantiles`: `(829, 1682, 5, 13)`
- `label_list`: `(829, 1682, 5)`

Interpretation:
- axis 0: test windows
- axis 1: stations
- axis 2: horizons
- axis 3: quantiles

Horizons:
- `[1, 3, 6, 12, 24]`

Quantiles:
- `[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.833333333333, 0.9, 0.95]`

## Critical Execution Constraint: No Leakage
CRC cannot be calibrated on the final test outputs.

This is the most important adaptation relative to generic prompts.

### What is not available directly
The repository does **not** currently store calibration predictions and labels as ready-made `.npy` files beside the saved test arrays.

That means CRC cannot be implemented as:
- load test predictions
- choose CRC quantiles on test
- report test performance

That would violate the split rule.

### What is available indirectly
The Stage 4 script already reconstructs calibration predictions on the fly by:

1. loading the Stage 1 checkpoint
2. rebuilding the calibration loader using the saved split contract
3. running inference on the calibration split

Therefore CRC must follow the same approach.

### Required split logic
Use this logic:

1. Model training is already complete.
2. Rebuild the original calibration split from the dataset and metadata.
3. Run inference on the calibration split to obtain:
   - `qhat_calib`
   - `y_calib`
4. Use the saved Stage 1 test outputs as:
   - `qhat_test`
   - `y_test`
5. Perform CRC selection on calibration only.
6. Report final results on test only.

If this cannot be done cleanly, stop and fix the export path first.

## Scientific Positioning
CRC should be framed as:

- a risk-budget-based deployment alternative
- not a replacement for the boundary-aware reliability story
- not a replacement for cost-aligned quantile deployment unless results clearly justify it

The right comparison question is:

> When monetary cost ratios are uncertain or hard to specify, can CRC deliver controlled shortage risk with better resource efficiency than conservative interval-upper-bound deployment?

## Version-1 Implementation Rules
These should be treated as fixed for the first CRC run.

1. Use raw quantile candidates only.
   - Do not stack Stage 2 CQR before CRC selection.
   - Stage 2 remains a baseline source, not a CRC candidate generator.

2. Reconstruct calibration predictions and labels from the original calibration split.
   - Never select CRC quantiles using test predictions or labels.

3. For horizon-wise CRC, flatten calibration samples over `(T_calib, N)` within each horizon.
   - Interpret the resulting guarantee as marginal station-window risk control for that horizon.

4. Candidates are ordered from most conservative to least conservative.
   - Select the least conservative feasible candidate, i.e. the largest candidate index `j` such that `crc_upper_j <= alpha`.

5. Use bounded-loss correction:
   - `crc_upper = (n / (n + 1)) * empirical_risk + B / (n + 1)`
   - with `B = 1.0` for version-1 losses

6. Weighted shortage loss must remain bounded in `[0, 1]`.
   - use:
     - `L = min(w * max(y - a, 0) / M_h, 1)`
   - do not use:
     - `w * min(max(y - a, 0) / M_h, 1)`
     unless the loss bound is adjusted accordingly

7. Use shortage alphas `0.03, 0.05, 0.10` in the first run.
   - add `0.01` only as a second-run sensitivity

8. Save calibration reproducibility artifacts.
   - save `qhat_calib` and `y_calib` if feasible
   - otherwise save at minimum:
     - hashes
     - shapes
     - split identifiers
     - timestamp metadata

9. Safety-margin baseline is optional for a smoke run, but required for paper-facing CRC results if CRC is retained in the main manuscript.

10. Add a `global_crc` baseline if implementation is straightforward.
   - this means one selected quantile pooled across all horizons
   - compare it with horizon-wise CRC

## Loss Design Constraints
Do not use the full asymmetric newsvendor cost as the CRC-controlled loss.

Reason:
- it is not monotone in the decision level
- CRC needs bounded monotone losses over ordered candidates

Use only losses in `[0, 1]` that become no larger as decisions become more conservative.

## Recommended Losses For First Implementation

### Mandatory loss 1: violation loss
```text
L_viol(a, y) = 1{y > a}
```

### Mandatory loss 2: normalized shortage loss
```text
L_short(a, y) = min(max(y - a, 0) / M_h, 1)
```

Use:
- `M_h = q95` of positive calibration labels for that horizon

This must be computed from calibration labels only.
Do not use test labels to set `M_h`.

### Optional loss 3: weighted normalized shortage
Only include this in the first implementation if target-hour metadata for both calibration and test can be reconstructed cleanly.

Suggested weighting:
- hours `16` to `21`: weight `2.0`
- all other hours: weight `1.0`

If calibration-side target-hour metadata cannot be reconstructed robustly, skip weighted loss and record that choice in metadata.

Implementation form:

```text
L = min(w * max(y - a, 0) / M_h, 1)
```

This preserves the bounded-loss assumption with `B = 1.0`.

## Candidate Decision Family

### First implementation scope
Use only the existing quantile grid:

```text
a_tau(x) = qhat(x, tau)
```

ordered from most conservative to least conservative:

```text
0.95, 0.9, 0.8333, 0.8, 0.75, ..., 0.05
```

This keeps the first implementation aligned with current project artifacts and existing deployment baselines.

Version-1 constraint:
- use raw quantile candidates only
- do not apply Stage 2 CQR thresholds before CRC selection

### Do not implement this in version 1
Do not implement the additive safety-offset family:

```text
a_lambda(x) = qhat(x, tau0) + lambda
```

unless the first CRC run shows that the method almost always chooses the most conservative available quantile and the grid is too coarse to be useful.

## CRC Selection Rule

For each:
- loss type
- target alpha
- horizon

construct a calibration loss table:

```text
loss_table: (num_calibration_samples, num_candidates)
```

Candidates must be ordered from:
- lowest risk / most conservative
to
- highest risk / least conservative

For candidate `j`, compute:

```text
risk_hat_j = mean(loss_table[:, j])
crc_upper_j = (n / (n + 1)) * risk_hat_j + 1 / (n + 1)
```

Selection rule:
- choose the least conservative candidate whose CRC upper bound is `<= alpha`

Because candidates are ordered from most conservative to least conservative, this means selecting the largest feasible candidate index.

Selection status:
- `no_candidate_satisfies_alpha`
- `all_candidates_satisfy_alpha`
- `interior_candidate_selected`

Save for each selection:
- loss name
- alpha
- horizon
- selected quantile
- candidate rank
- empirical calibration risk
- CRC upper bound
- selection status
- number of calibration samples

## Required Inputs

### Stage 1 test outputs
From:
- `journal_results/shenzhen_multihorizon/warmstart_raw/`

Use:
- `predict_quantiles.npy`
- `label_list.npy`
- `run_metadata.json`

### Stage 2 outputs
From:
- `journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw/`

Use only for baseline reconstruction or cross-checking:
- `cqr_thresholds.csv`
- `stage2_metadata.json`

CRC should not depend on Stage 2 thresholds for policy selection in version 1.

### Stage 4 outputs
From:
- `journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw/`

Use for baseline comparison and consistency:
- `decision_summary_by_method_and_cost_ratio.csv`
- `decision_metrics_by_horizon.csv`
- `decision_regret_vs_oracle.csv`
- `decision_one_sided_thresholds.csv`
- `decision_eval_metadata.json`

### Reinforcement outputs
From:
- `journal_results/shenzhen_multihorizon/reinforcement/`

Use:
- `test_window_manifest.csv`
- `bootstrap_ci_day_level.csv`
- deployment strategy outputs as comparison references

These are useful for:
- day-level bootstrap
- test target-date mapping
- paper-facing baseline alignment

### Dataset inputs
From:
- `data/datasets/ST_EVCDP_v2_canonical/`

Use only as needed to rebuild calibration loaders and recover calibration-side timestamps if required.

## Baselines

### Mandatory baselines
1. `median_decision`
   - `a = q0.5`

2. `symmetric_90_upper`
   - reconstruct from raw `q0.95` plus Stage 2 global CQR threshold

3. `raw_cost_aligned_quantile`
   - `3:1 -> q0.75`
   - `5:1 -> q0.8333`
   - `9:1 -> q0.9`
   - `19:1 -> q0.95`

4. `one_sided_refined_cost_aligned_quantile`
   - reconstruct from Stage 4 one-sided thresholds

5. `global_crc`
   - one selected quantile pooled across all horizons
   - include if implementation remains straightforward

### Optional baseline
6. `safety_margin`
   - `a = q0.5 + m_h`
   - select `m_h` on calibration only

This baseline is useful, but it is not required for the very first run if implementation scope must be kept tight.
If CRC is retained in the paper-facing main text, add this baseline before final comparison tables are frozen.

### Do not add in version 1 by default
- minimax-regret deployment from misspecification
- expected-cost deployment from misspecification

Those can be added later if CRC looks promising and the comparison matrix needs expansion.

## Alpha Grids

### Violation-risk targets
Recommended:
- `0.05`
- `0.10`
- `0.20`

### Shortage-risk targets
Recommended:
- `0.03`
- `0.05`
- `0.10`

Use `0.01` only as a second-run sensitivity.

These are sensible first-pass grids.

If CRC always selects the same extreme quantile for all alphas, the result is still informative and should be reported honestly.

## Evaluation Metrics On Test

### Risk metrics
- `test_violation_rate`
- `test_normalized_shortage_risk`
- `test_weighted_shortage_risk` if implemented
- `shortage_frequency`
- `mean_shortage`
- `cvar95_shortage`

### Resource-use metrics
- `mean_decision_level`
- `mean_overage`
- `mean_absolute_overage`
- optional utilization proxy

### Cost metrics
Evaluate downstream asymmetric cost under:
- `3:1`
- `5:1`
- `9:1`
- `19:1`

This keeps CRC directly comparable to the current decision experiments.

### Reporting levels
Always report:
- by horizon
- pooled over horizons

## Bootstrap

### Required bootstrap rule
Use day-level bootstrap if possible.

Current source:
- `journal_results/shenzhen_multihorizon/reinforcement/test_window_manifest.csv`

If calibration/test target-date mapping can be aligned cleanly, use day-level bootstrap.
If not, fall back to window-level bootstrap and record a warning in metadata.

### Metrics worth bootstrapping
- test risk for each CRC loss and alpha
- expected cost under each cost ratio
- risk reduction versus `symmetric_90_upper`
- overage reduction versus `symmetric_90_upper`
- `cvar95_shortage`

## Outputs To Save

### CSV files
- `crc_selection_table.csv`
- `crc_test_risk_summary.csv`
- `crc_by_horizon.csv`
- `crc_baseline_comparison.csv`
- `crc_risk_efficiency_frontier.csv`
- `crc_cost_ratio_evaluation.csv`
- `crc_bootstrap_ci_day_level.csv`
- `crc_metadata.json`

Calibration reproducibility files:
- `crc_calibration_manifest.json`
- `crc_calibration_hashes.json`

Optional if storage is acceptable:
- `qhat_calib.npy`
- `y_calib.npy`

Optional if weighted loss is implemented:
- `crc_weighted_shortage_summary.csv`

### Figures
- `crc_test_risk_vs_alpha.png`
- `crc_selected_quantiles_by_alpha.png`
- `crc_risk_efficiency_frontier.png`
- `crc_expected_cost_by_policy.png`
- `crc_shortage_cvar_by_policy.png`

### Markdown report
- `crc_experiment_report.md`

## Files To Add
- `scripts/journal/build_crc_risk_control_experiments.py`
- `scripts/journal/run_crc_risk_control_warmstart.sh`
- `tests/test_journal_crc_risk_control.py`

## CLI Contract
The new script should support:

```text
--stage1-output-dir
--stage2-output-dir
--stage4-output-dir
--reinforcement-dir
--paper-assets-dir
--output-dir
--alphas-violation
--alphas-shortage
--cost-ratios
--bootstrap-samples
--bootstrap-seed
--machine
```

## Recommended First Run
The first Lenovo run should be intentionally narrow:

- losses:
  - violation
  - normalized shortage
- alphas:
  - violation: `0.05,0.10,0.20`
  - shortage: `0.03,0.05,0.10`
- baselines:
  - mandatory four only
  - plus `global_crc` if straightforward
- bootstrap:
  - 300 samples

Do not include weighted loss or safety-margin baseline in the very first run unless the implementation remains straightforward.

## Unit Tests Required

At minimum:

1. shape-loading test
2. loss boundedness test
3. monotonicity of candidate risk ordering under shortage-based losses
4. CRC selector synthetic test
5. no-leakage metadata test
6. baseline presence test
7. output-file existence test on a smoke fixture
8. calibration manifest or hash output test

## Scientific Acceptance Criteria
This CRC package is worth keeping only if it answers at least one of these convincingly:

1. CRC attains empirical test risk near or below the target alpha across horizons.
2. CRC is less conservative than `symmetric_90_upper` while maintaining shortage-risk control.
3. CRC improves overage or mean decision level relative to conservative interval-upper-bound deployment.
4. CRC offers a credible alternative when monetary cost ratios are hard to specify.

## How To Interpret Outcomes

### If CRC is strong
Use it as:
- a comparison module
- an alternative deployment layer when service-risk targets are more credible than monetary cost ratios

Do not automatically replace the current main paper narrative.

### If CRC is mixed
This is still publishable as a comparison result.

Likely interpretation:
- cost-aligned quantiles remain better for expected-cost minimization when the cost ratio is reliable
- CRC is useful as a reliability safeguard or service-level deployment tool

### If CRC is weak
If CRC always chooses the most conservative quantile or gives poor cost/resource efficiency:
- keep it out of the main text
- report briefly as appendix or omit

## Expected Command Template
```powershell
conda run -n py12 python scripts/journal/build_crc_risk_control_experiments.py `
  --stage1-output-dir journal_results/shenzhen_multihorizon/warmstart_raw `
  --stage2-output-dir journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw `
  --stage4-output-dir journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw `
  --reinforcement-dir journal_results/shenzhen_multihorizon/reinforcement `
  --paper-assets-dir journal_results/shenzhen_multihorizon/paper_assets `
  --output-dir journal_results/shenzhen_multihorizon/crc_risk_control `
  --alphas-violation 0.05,0.10,0.20 `
  --alphas-shortage 0.03,0.05,0.10 `
  --cost-ratios 3:1,5:1,9:1,19:1 `
  --bootstrap-samples 300 `
  --bootstrap-seed 20260520 `
  --machine Lenovo
```

## Final Execution Recommendation
This experiment is executable on the current branch, but only if it is treated as:

- a calibration-split reconstruction problem first
- and a CRC comparison experiment second

The biggest engineering risk is not the CRC formula.
It is correctly rebuilding calibration predictions without leakage and keeping the tensor/time indexing aligned with the existing journal pipeline.
