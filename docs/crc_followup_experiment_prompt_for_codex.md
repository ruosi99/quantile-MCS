# CRC Follow-Up Experiments for the Journal Branch

## Metadata
- Date: 2026-05-14
- Target branch: `codex/CRC_risk_control` or current journal branch after merging CRC v1
- Target machine: Lenovo
- Primary output directory:
  - `journal_results/shenzhen_multihorizon/crc_risk_control_followup/`
- Upstream CRC v1 output directory:
  - `journal_results/shenzhen_multihorizon/crc_risk_control/`

## Background
CRC v1 has already been implemented and run. The current v1 implementation follows the no-leakage rule:

- calibration predictions are reconstructed from the Stage 1 checkpoint and calibration split;
- CRC quantile selections are made on calibration predictions and labels only;
- final policy evaluation uses saved Stage 1 test predictions and labels;
- `selection_uses_test_labels` is false.

CRC v1 scope:

- losses: violation and normalized shortage;
- candidate family: raw quantile grid only;
- baselines: median, symmetric 90 upper, raw cost-aligned quantiles, one-sided refined cost-aligned quantiles, global CRC, horizon-wise CRC;
- weighted loss: not implemented;
- safety-margin baseline: not implemented;
- calibration arrays were not saved, only hashes and manifest were saved.

Key CRC v1 findings:

- Violation-risk CRC is scientifically useful and should be retained for further evaluation.
- Horizon-wise violation CRC at alpha = 0.10 achieves pooled test violation around 0.09 with much lower overage than symmetric 90 upper.
- Symmetric 90 upper is very conservative: low violation but high overage.
- Normalized-shortage CRC controls shortage magnitude but allows high violation frequency; it should not be framed as a service reliability policy.
- H1 strict alpha = 0.05 control is mixed: pooled risk looks good, but H1 can exceed alpha on test.

The goal of this follow-up is to decide whether CRC can be retained as a main paper contribution, and if so, whether it should be framed as:

> a service-risk-controlled deployment layer for operators who trust risk budgets more than exact monetary cost ratios.

This is NOT a request to retrain the forecasting model.

---

## Scientific Goal
Add the missing reviewer-defense experiments for CRC:

1. Compare CRC with deterministic safety-margin baselines.
2. Compare CRC-corrected selection with plain empirical-risk selection.
3. Diagnose H1 strict-control sensitivity and temporal/block robustness.
4. Optionally evaluate peak-hour or target-hour group-wise CRC if timestamp metadata can be reconstructed cleanly.
5. Update paper-facing risk-efficiency and cost evaluation tables/figures.

The final interpretation should be:

- Cost-aligned critical-fractile quantiles remain appropriate when the monetary cost ratio is known and trusted.
- CRC is an alternative deployment layer when the operator specifies service-risk targets, e.g., violation probability or shortage-risk budget, instead of an exact monetary cost ratio.
- CRC should not be claimed to minimize expected asymmetric cost.
- CRC should be compared against CQR upper endpoints, one-sided conformal decisions, deterministic safety margins, and raw cost-aligned quantiles.

---

## Important Theory Constraints
CRC controls bounded monotone risk. Use CRC only with losses that are no larger when the provisioning decision becomes more conservative.

Allowed primary CRC loss:

```text
L_viol(a, y) = 1{y > a}
```

Allowed secondary CRC loss:

```text
L_short(a, y) = min(max(y - a, 0) / M_h, 1)
```

Do NOT use the full asymmetric newsvendor cost as a CRC-controlled loss, because it is not monotone in decision level `a`.

Asymmetric costs should be used only for evaluation after the policy is selected.

---

## Required No-Leakage Protocol
Do not select any policy on test labels.

Use this split logic:

1. Load trained Stage 1 multi-horizon quantile model.
2. Reconstruct calibration predictions and labels from the calibration split.
3. Use calibration predictions and labels only for:
   - CRC quantile selection;
   - empirical-risk selector selection;
   - safety-margin selection;
   - block/day-level selector selection;
   - group-wise selector selection.
4. Use saved Stage 1 test predictions and labels only for final evaluation.
5. Save split metadata, hashes, selected policies, and selection source metadata.

If calibration predictions cannot be reconstructed cleanly, stop and fix the export path. Do not fall back to selecting on test data.

Strongly recommended: save the reconstructed calibration arrays for reproducibility, unless storage is a concern.

Suggested files:

```text
crc_calib_quantiles.npy      # optional but recommended
crc_calib_labels.npy         # optional but recommended
crc_calib_manifest.json
crc_calib_hashes.json
```

---

## Existing Inputs
Use the same inputs as CRC v1:

```text
--stage1-output-dir journal_results/shenzhen_multihorizon/warmstart_raw
--stage2-output-dir journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw
--stage4-output-dir journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw
--reinforcement-dir journal_results/shenzhen_multihorizon/reinforcement
--paper-assets-dir journal_results/shenzhen_multihorizon/paper_assets
--crc-v1-dir journal_results/shenzhen_multihorizon/crc_risk_control
```

Known Stage 1 test array shapes:

```text
test_quantiles: (829, 1682, 5, 13)
test_labels:    (829, 1682, 5)
```

Known calibration reconstruction shape from CRC v1:

```text
calib_quantiles: (828, 1682, 5, 13)
calib_labels:    (828, 1682, 5)
```

Horizons:

```text
[1, 3, 6, 12, 24]
```

Quantiles:

```text
[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.833333333333, 0.9, 0.95]
```

---

## New Script To Add
Add one of the following:

Preferred:

```text
scripts/journal/build_crc_followup_experiments.py
scripts/journal/run_crc_followup_experiments_warmstart.sh
```

Alternative:

- extend `scripts/journal/build_crc_risk_control_experiments.py` with `--followup` flags.

Add tests:

```text
tests/test_journal_crc_followup.py
```

---

# Experiment 1: Deterministic Safety-Margin Baseline

## Purpose
Reviewer-defense baseline. A reviewer will ask whether a deterministic median forecast plus a tuned safety margin can achieve the same service-risk control as CRC.

## Policies
Use the median quantile as the base forecast:

```text
a_m(x) = q0.5(x) + m
```

Implement two versions:

### 1. Global safety margin
One margin shared across all horizons:

```text
m_global
```

### 2. Horizon-wise safety margin
One margin per horizon:

```text
m_h, h in [1, 3, 6, 12, 24]
```

## Candidate margin grid
Construct candidate margins from calibration residuals only.

For each horizon h:

```text
residual_h = max(y_calib_h - q0.5_calib_h, 0)
```

Recommended candidate margin quantiles:

```text
[0.0, 0.25, 0.50, 0.75, 0.80, 0.833333, 0.90, 0.95, 0.975, 0.99]
```

Use unique sorted candidate margins.

For global margin, pool residuals across all horizons and station-window samples.

## Selection rules
Implement both:

### 1. Empirical-risk safety margin
Select the smallest margin whose calibration empirical violation risk is <= alpha.

### 2. CRC-corrected safety margin
Select the smallest margin whose CRC-corrected risk upper bound is <= alpha.

Use bounded loss correction:

```text
crc_upper = (n / (n + 1)) * empirical_risk + 1.0 / (n + 1)
```

## Alphas
Use violation-risk alphas:

```text
[0.05, 0.10, 0.20]
```

## Evaluation metrics on test
For each policy and alpha:

- test violation rate;
- mean decision;
- mean overage;
- mean shortage;
- normalized shortage risk;
- CVaR95 shortage;
- expected asymmetric cost at cost ratios 3:1, 5:1, 9:1, 19:1;
- by-horizon versions of all core metrics.

## Outputs

```text
crc_safety_margin_selection.csv
crc_safety_margin_test_summary.csv
crc_safety_margin_by_horizon.csv
crc_safety_margin_vs_crc.png
```

Required columns for `crc_safety_margin_test_summary.csv`:

```text
policy, selection_rule, alpha, margin_scope, selected_margin, selected_quantile_equivalent_optional,
test_violation_rate, mean_decision, mean_overage, mean_shortage, cvar95_shortage,
expected_cost_3_1, expected_cost_5_1, expected_cost_9_1, expected_cost_19_1
```

---

# Experiment 2: Empirical-Risk Selector vs CRC-Corrected Selector

## Purpose
Show whether the CRC correction matters beyond simple validation-threshold selection.

## Candidate family
Use the same raw quantile candidates as CRC v1:

```text
a_tau(x) = qhat(x, tau)
```

Candidates ordered from most conservative to least conservative:

```text
0.95, 0.9, 0.833333, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05
```

## Selectors

### 1. Empirical-risk selector
Choose the least conservative quantile whose calibration empirical violation risk is <= alpha.

### 2. CRC-corrected selector
Choose the least conservative quantile whose CRC upper bound is <= alpha.

## Important implementation detail
Because candidates are ordered from most conservative to least conservative:

```text
select the largest index j such that risk_bound_j <= alpha
```

Equivalently, scan from least conservative to most conservative and stop at the first feasible candidate.

## Alphas

```text
[0.05, 0.10, 0.20]
```

## Scope
Implement both:

- global selector;
- horizon-wise selector.

## Evaluation
For each selector:

- selected quantile;
- calibration empirical risk;
- CRC upper bound;
- test violation rate;
- mean overage;
- mean decision;
- CVaR95 shortage;
- expected costs at 3:1, 5:1, 9:1, 19:1.

## Outputs

```text
crc_empirical_vs_corrected_selection.csv
crc_empirical_vs_corrected_test_summary.csv
crc_empirical_vs_crc_selector.png
```

## Expected interpretation
If CRC and empirical selectors are nearly identical, report that the calibration set is large and the CRC correction is small, but CRC still provides the principled finite-sample risk-control protocol.

If empirical selectors are more aggressive and violate alpha more often on test, highlight the value of CRC correction.

---

# Experiment 3: H1 and Temporal/Blocked Robustness

## Purpose
CRC v1 showed pooled violation-risk control is strong, but strict H1 alpha = 0.05 test control was mixed. This experiment diagnoses whether stricter H1 alpha or block-level selection improves robustness.

## Experiment 3A: H1 alpha sensitivity
Run H1-only horizon-wise CRC violation selection for:

```text
alpha in [0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
```

Use raw quantile candidates.

Report:

- selected quantile;
- calibration empirical risk;
- CRC upper bound;
- H1 test violation;
- H1 mean overage;
- H1 CVaR95 shortage;
- H1 expected cost at 3:1, 5:1, 9:1, 19:1.

Output:

```text
crc_h1_alpha_sensitivity.csv
crc_h1_alpha_sensitivity.png
```

## Experiment 3B: Block-aggregated CRC selection
Current CRC v1 interprets each station-window pair as a marginal calibration sample. This can be optimistic under station correlation.

Implement block-level selection where each calibration sample is a time block.

Preferred block definition:

1. If target-date metadata can be reconstructed cleanly, use target-day blocks.
2. Otherwise, use time-window index blocks, aggregating over stations.

For each block b, candidate tau, and horizon h:

```text
block_loss[b, tau, h] = mean over stations of 1{y > q_tau}
```

Then apply CRC selection over the block-level loss table.

Run for:

```text
alpha in [0.05, 0.10, 0.20]
```

Scope:

- horizon-wise block CRC;
- optional global block CRC.

Evaluate on test using the same block definition if metadata exists. Also report pooled station-window metrics for comparison.

Outputs:

```text
crc_blocked_selection.csv
crc_blocked_test_summary.csv
crc_blocked_by_horizon.csv
crc_blocked_vs_station_window.png
```

Required metadata:

```text
block_definition, num_calib_blocks, num_test_blocks, target_date_available, fallback_used
```

## Interpretation
If block CRC is more conservative but improves H1 alpha control, frame it as a robustness option for temporally correlated operations.

If block CRC is too conservative, keep it in appendix and state that station-window CRC is the main deployment layer.

---

# Experiment 4: Peak-Hour / Target-Hour Group-Wise CRC (Optional but Valuable)

## Purpose
Make CRC more operational: operators may want stricter service-risk control during evening peaks.

## Prerequisite
Target-hour metadata must be reconstructed cleanly for both calibration and test.

If target-hour metadata cannot be reconstructed, skip this experiment and record the reason in metadata.

## Groups
Use target-hour groups:

```text
peak: target hour in [16, 17, 18, 19, 20, 21]
non_peak: all other target hours
```

Optional expanded groups:

```text
overnight: 00-05
morning:   06-10
midday:    11-15
evening:   16-21
late:      22-23
```

## Selection
For each group and horizon, select raw quantile candidates using CRC violation risk.

Use minimum sample fallback:

```text
if group calibration samples < min_samples, fall back to horizon-wise CRC selection
```

Recommended minimum:

```text
min_samples = 10000 station-window samples
```

## Alphas

```text
[0.05, 0.10, 0.20]
```

## Evaluation
Report for each group:

- selected quantile;
- calibration CRC upper bound;
- test violation rate;
- mean overage;
- CVaR95 shortage;
- expected costs at 3:1, 5:1, 9:1, 19:1;
- fallback flag and sample size.

Outputs:

```text
crc_peak_hour_group_selection.csv
crc_peak_hour_group_test_summary.csv
crc_peak_hour_group_by_horizon.csv
crc_peak_hour_crc.png
```

## Interpretation
If peak-hour CRC reduces peak violation risk with acceptable overage, this becomes a strong paper-facing operational result.

If it does not improve over horizon-wise CRC, place it in appendix and state that horizon-wise CRC is sufficient in the current dataset.

---

# Experiment 5: Updated Risk-Efficiency Frontier and Paper-Facing Summary

## Purpose
Merge CRC v1 and follow-up outputs into one paper-facing set of tables and figures.

## Policies to include
At minimum:

- median decision;
- symmetric 90 upper;
- raw cost-aligned quantiles for 3:1, 5:1, 9:1, 19:1;
- one-sided refined cost-aligned quantiles;
- global CRC violation alpha = 0.05, 0.10, 0.20;
- horizon-wise CRC violation alpha = 0.05, 0.10, 0.20;
- empirical-risk raw-quantile selector;
- deterministic safety-margin global and horizon-wise;
- block CRC if implemented;
- peak-hour CRC if implemented.

## Metrics
For every policy:

- test violation rate;
- normalized shortage risk;
- mean decision;
- mean overage;
- mean shortage;
- CVaR95 shortage;
- expected cost at 3:1, 5:1, 9:1, 19:1;
- by-horizon metrics;
- day-level bootstrap CI where feasible.

## Figures
Generate:

```text
crc_followup_risk_efficiency_frontier.png
crc_followup_expected_cost_by_policy.png
crc_followup_shortage_cvar_by_policy.png
crc_followup_violation_vs_overage.png
crc_followup_policy_decision_flow.png   # optional if plotting helper exists
```

Suggested risk-efficiency plot:

- x-axis: mean overage or mean decision;
- y-axis: test violation rate or CVaR95 shortage;
- color: policy family;
- annotate symmetric 90 upper, horizon CRC alpha 0.10, median, and best safety margin.

## Tables
Generate:

```text
crc_followup_policy_comparison.csv
crc_followup_risk_efficiency_table.csv
crc_followup_cost_table.csv
crc_followup_bootstrap_ci.csv
crc_followup_report.md
```

---

## Bootstrap Requirements
Use day-level bootstrap if target-date metadata is available.

If only test-window manifest is available, use the same day-level grouping used in reinforcement experiments.

Minimum bootstrap samples:

```text
300
```

Compute 95% CI for:

- overage reduction vs symmetric 90 upper;
- violation difference vs alpha;
- expected cost difference vs symmetric 90 upper;
- expected cost difference vs safety margin;
- CVaR95 shortage difference if computationally feasible.

If bootstrap for CVaR is unstable, record it but do not use it as a primary claim.

---

## Tests To Add
Add unit and integration tests in:

```text
tests/test_journal_crc_followup.py
```

Required tests:

1. No test-label selection:
   - selected policies must be computed from calibration arrays only.
2. Safety-margin monotonicity:
   - violation loss must be non-increasing as margin increases.
3. Raw-quantile candidate monotonicity:
   - violation loss must be non-increasing as quantile level increases.
4. CRC selector correctness:
   - synthetic example where q0.95, q0.9, q0.833 are feasible but q0.75 is not; selector must choose q0.833, not q0.95.
5. Empirical vs CRC correction:
   - CRC feasible set must be a subset of empirical feasible set for bounded loss B = 1.
6. Block aggregation shape:
   - block-level loss table shape must be `(num_blocks, num_candidates)` for each horizon.
7. Output existence:
   - all required CSV, JSON, and MD report files are created.
8. Bounded loss:
   - all CRC losses must be within [0, 1].

Run tests with:

```powershell
conda run -n py12 python -m pytest tests/test_journal_crc_followup.py -q -p no:cacheprovider
```

---

## Suggested Full Command
Update paths if needed.

```powershell
conda run -n py12 python scripts/journal/build_crc_followup_experiments.py `
  --stage1-output-dir journal_results/shenzhen_multihorizon/warmstart_raw `
  --stage2-output-dir journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw `
  --stage4-output-dir journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw `
  --reinforcement-dir journal_results/shenzhen_multihorizon/reinforcement `
  --paper-assets-dir journal_results/shenzhen_multihorizon/paper_assets `
  --crc-v1-dir journal_results/shenzhen_multihorizon/crc_risk_control `
  --output-dir journal_results/shenzhen_multihorizon/crc_risk_control_followup `
  --alphas-violation 0.05,0.10,0.20 `
  --h1-alphas 0.03,0.04,0.05,0.06,0.08,0.10 `
  --cost-ratios 3:1,5:1,9:1,19:1 `
  --bootstrap-samples 300 `
  --bootstrap-seed 20260521 `
  --machine Lenovo
```

Optional flags:

```text
--enable-block-crc
--enable-peak-hour-crc
--save-calibration-arrays
--min-group-samples 10000
```

---

## Final Report Structure
The script should write:

```text
crc_followup_report.md
```

with these sections:

1. No-leakage verification.
2. Summary of v1 CRC results reused.
3. Safety-margin baseline comparison.
4. Empirical-risk selector vs CRC-corrected selector.
5. H1 strict-control and block robustness.
6. Peak-hour/group-wise CRC if implemented.
7. Updated risk-efficiency frontier.
8. Paper interpretation.
9. Recommended manuscript positioning.

Recommended conclusion language:

```text
Violation-risk CRC should be retained as a service-risk-target deployment layer. It does not replace cost-aligned critical-fractile decisions when monetary cost ratios are reliable. Instead, it provides a principled alternative when operators specify an acceptable service-violation risk rather than an exact shortage-to-overage cost ratio.
```

---

## Manuscript Interpretation Rules
Use these rules when writing the final report.

Do say:

- CRC controls a bounded monotone operational risk, such as service violation.
- CRC can reduce over-provision relative to conservative interval-upper-bound deployment at the same or acceptable risk level.
- Cost-aligned quantiles remain the expected-cost benchmark when cost ratios are trusted.
- Safety-margin comparison is required before claiming CRC is operationally superior.

Do not say:

- CRC directly minimizes asymmetric expected cost.
- CRC replaces CQR in all settings.
- CRC guarantees every horizon's empirical test risk is below alpha.
- Normalized-shortage CRC is a service-violation-control policy.
- Test labels were used to select policies.

---

## Go / No-Go Criteria For Main-Paper CRC
CRC can enter the main paper if at least two of the following hold:

1. Horizon-wise CRC alpha = 0.10 or 0.20 controls pooled test violation below target and substantially reduces overage versus symmetric 90 upper.
2. CRC-corrected selector is more reliable than empirical-risk selector, or empirically similar with a clear finite-sample protocol justification.
3. CRC performs competitively against tuned safety margins in risk-efficiency frontier.
4. Block or H1 robustness analysis explains the H1 alpha = 0.05 exception.
5. Peak-hour/group-wise CRC provides interpretable operational gains.

If fewer than two hold, keep CRC as an appendix or robustness experiment and maintain the main narrative around cost-aligned quantile deployment and cost-ratio uncertainty.
