# Preference Mapping And Risk-Screened Deployment For The Current Journal Branch

## Metadata
- Date: 2026-05-20
- Branch: `multi_horizon_journal`
- Authoring machine: `Dell`
- Target execution machine: `Lenovo`
- Planned output directory:
  - `journal_results/shenzhen_multihorizon/preference_mapping/`

## Purpose
This document rewrites the generic preference-mapping prompt so it fits the current repository, the current result folders, and the paper direction already established on this branch.

The central question is:

> Given multi-horizon EV charging probabilistic forecasts, how should an operator deploy station-level provisioning decisions when preferences are expressed either as a monetary cost ratio or as a service-risk budget?

The current branch already has:

- cost-aware quantile deployment results
- one-sided refinement results
- deployment strategy comparison
- CRC risk-controlled deployment results
- CRC follow-up reviewer-defense results

So this package should not re-derive everything from scratch. Its purpose is to:

1. empirically map cost-ratio language to risk-budget language
2. show where the two preference languages agree or disagree
3. test a risk-screened deployment rule that reconciles them

## Current Project Context

### Existing result folders
These folders already exist and should be reused:

- `journal_results/shenzhen_multihorizon/warmstart_raw/`
- `journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw/`
- `journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw/`
- `journal_results/shenzhen_multihorizon/reinforcement/`
- `journal_results/shenzhen_multihorizon/paper_assets/`
- `journal_results/shenzhen_multihorizon/crc_risk_control/`
- `journal_results/shenzhen_multihorizon/crc_risk_control_followup/`
- `journal_results/shenzhen_multihorizon/misspecification/`

### Current Stage 1 test arrays
Known current files:

- `predict_quantiles.npy`
- `label_list.npy`

Known shapes:

- `predict_quantiles`: `(829, 1682, 5, 13)`
- `label_list`: `(829, 1682, 5)`

Horizons:
- `[1, 3, 6, 12, 24]`

Quantile grid:
- `[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.833333333333, 0.9, 0.95]`

### Existing scripts that should be reused conceptually
- `scripts/journal/evaluate_stage4_decision.py`
- `scripts/journal/build_paper_evidence_assets.py`
- `scripts/journal/build_reinforcement_experiments.py`
- `scripts/journal/build_crc_risk_control_experiments.py`
- `scripts/journal/build_crc_followup_experiments.py`

## Paper Direction Constraint
This experiment should extend the current paper narrative, not replace it.

Current main line:

- boundary-aware reliability diagnostics
- decision-value attribution
- deployment strategy comparison
- service-risk deployment via CRC

Preference mapping should therefore be framed as:

- a deployment-layer bridge between cost-aware and risk-aware policy selection
- not as a new forecasting model
- not as a replacement for the boundary-aware reliability story

## Critical No-Leakage Rule
No policy selection may use final test labels.

Allowed:

- cost-aware quantile policies may be determined analytically from the cost ratio
- CRC policies may be imported from existing CRC result folders if their original selection was calibration-only
- new risk-screened policies must be selected on calibration only if they require label-based optimization
- test labels may be used only for final reporting

### Practical implication for this experiment
This package should prefer reusing already-selected policies from:

- `crc_risk_control/`
- `crc_risk_control_followup/`

rather than recomputing CRC from scratch.

Only the new risk-screened feasible-cost-min policy requires calibration-side selection logic.

## Existing Assets That Should Be Reused Directly

### Cost-aware side
From `stage4_decision/warmstart_raw/`:

- `decision_summary_by_method_and_cost_ratio.csv`
- `decision_metrics_by_horizon.csv`
- `decision_regret_vs_oracle.csv`
- `decision_one_sided_thresholds.csv`
- `decision_eval_metadata.json`

These already define:

- raw cost-aligned quantile policies
- one-sided refined cost-aligned quantile policies
- their test performance

### Risk-aware side
From `crc_risk_control/`:

- `crc_selection_table.csv`
- `crc_test_risk_summary.csv`
- `crc_by_horizon.csv`
- `crc_baseline_comparison.csv`
- `crc_cost_ratio_evaluation.csv`
- `crc_bootstrap_ci_day_level.csv`
- `crc_metadata.json`

From `crc_risk_control_followup/`:

- `crc_followup_policy_comparison.csv`
- `crc_followup_risk_efficiency_table.csv`
- `crc_followup_bootstrap_ci.csv`
- `crc_h1_alpha_sensitivity.csv`
- `crc_safety_margin_test_summary.csv`
- `crc_empirical_vs_corrected_test_summary.csv`
- `crc_blocked_test_summary.csv`
- `crc_followup_metadata.json`

### Timestamp and bootstrap support
From `reinforcement/`:

- `test_window_manifest.csv`
- `bootstrap_ci_day_level.csv`

This should be reused for day-level bootstrap rather than recreated independently.

## New Files To Add
- `scripts/journal/build_preference_mapping_experiments.py`
- `scripts/journal/run_preference_mapping_experiments_warmstart.sh`
- `tests/test_journal_preference_mapping.py`

Output directory:
- `journal_results/shenzhen_multihorizon/preference_mapping/`

Paper-facing report:
- `preference_mapping_report.md`

## Execution Design: Gate And Scope
This package should **not** be implemented as one large monolithic run by default.

### Phase A: Mapping and conflict analysis
Mandatory first phase:

1. cost-to-risk mapping
2. risk-to-cost mapping
3. preference equivalence frontier
4. preference conflict map

These can mostly reuse existing outputs plus lightweight derived metrics.

### Phase A gate
Only continue beyond Phase A if at least one of the following holds:

- cost-aware and risk-aware policies show a non-trivial empirical mismatch in selected quantiles or realized violation
- the conflict map reveals recurring settings where cost-only policies violate the intended service-risk budget
- the preliminary frontier suggests that a risk-screened policy could improve the risk-cost trade-off relative to both parents

If Phase A shows near-complete equivalence between the cost-aware and risk-aware branches, keep this package as appendix-grade robustness and do not automatically expand to all later variants.

### Phase B: Risk-screened deployment
Only after Phase A is stable:

5. risk-screened deployment

### Phase C: Aggregate procurement-style proxy
Optional only, default off:

6. aggregate procurement-style evaluation

This should not be in the first Lenovo run by default.

## Core Definitions

### Cost ratio to quantile
For cost ratio `r = c_u / c_o`:

```text
tau(r) = r / (1 + r)
```

Main exact grid:

- `1:1 -> 0.5`
- `3:1 -> 0.75`
- `5:1 -> 0.833333333333`
- `9:1 -> 0.9`
- `19:1 -> 0.95`

### Quantile to implied cost ratio
For a quantile `tau`:

```text
r_implied = tau / (1 - tau)
```

### Violation risk

```text
violation = 1{y > a}
```

### Overage and shortage

```text
overage  = max(a - y, 0)
shortage = max(y - a, 0)
```

### Expected asymmetric cost

```text
cost_r(a, y) = r * max(y - a, 0) + max(a - y, 0)
```

This is only for evaluation and calibration-side feasible-set cost comparison.

## Experiment 1: Cost-To-Risk Mapping

### Purpose
Translate monetary cost-ratio preferences into empirical service-risk outcomes.

Question:

> If the operator chooses the quantile implied by a known cost ratio, what actual violation risk does that policy produce on the test set?

### Policies
Use these exact raw quantiles:

- `1:1 -> q0.5`
- `3:1 -> q0.75`
- `5:1 -> q0.833333333333`
- `9:1 -> q0.9`
- `19:1 -> q0.95`

Optionally also include the one-sided refined versions from Stage 4 in the same output table, but keep raw cost-aligned policies as the primary mapping line.

### Metrics
Pooled and by horizon:

- deployed quantile
- nominal risk `1 - tau`
- actual violation rate
- violation gap
- mean overage
- mean shortage
- shortage CVaR95
- expected cost under own ratio
- expected cost under all main true cost ratios

### Outputs
- `cost_to_risk_mapping.csv`
- `cost_to_risk_mapping_by_horizon.csv`
- `cost_to_risk_mapping.png`

## Experiment 2: Risk-To-Cost Mapping

### Purpose
Translate service-risk-budget preferences into implied monetary preferences.

Question:

> If the operator specifies a service-risk budget alpha, what quantile does CRC deploy, and what cost ratio does that imply?

### Required CRC families
Keep the first pass focused on:

- `global_crc_violation`
- `horizon_crc_violation`

Do not make normalized-shortage CRC the primary mapping family in version 1, because we already know from CRC v1 that it is not equivalent to service-violation control.

Normalized-shortage CRC may be included as appendix-ready sensitivity if desired.

### Main alphas
- `0.05`
- `0.10`
- `0.20`

### H1 sensitivity
Reuse H1-specific follow-up results:
- `0.03, 0.04, 0.05, 0.06, 0.08, 0.10`

### Metrics
- selected quantile
- implied cost ratio
- actual test violation rate
- mean overage
- mean shortage
- shortage CVaR95
- expected cost under all main cost ratios
- selection status

### Outputs
- `risk_to_cost_mapping.csv`
- `risk_to_cost_mapping_by_horizon.csv`
- `risk_to_cost_mapping.png`
- `h1_risk_to_cost_mapping.csv`

## Experiment 3: Preference Equivalence Frontier

### Purpose
Place cost-aware and risk-aware policies in the same empirical deployment space.

Question:

> Are monetary cost ratios and service-risk budgets empirically equivalent preference languages on this dataset?

### Primary policy set for the main frontier
Keep the first version compact:

- `median_decision`
- `symmetric_90_upper`
- `raw_cost_aligned_quantile_{1:1,3:1,5:1,9:1,19:1}`
- `one_sided_refined_cost_aligned_quantile_{3:1,5:1,9:1,19:1}`
- `global_crc_violation_alpha0.05/0.10/0.20`
- `horizon_crc_violation_alpha0.05/0.10/0.20`

### Secondary policy set for appendix or extended frontier
Include only if figure clutter remains manageable:

- safety-margin follow-up policies
- empirical selector policies
- block CRC policies

### Required figures
1. `risk_resource_frontier.png`
   - x-axis: actual violation rate
   - y-axis: mean overage

2. `risk_cost_frontier_by_ratio.png`
   - x-axis: actual violation rate
   - y-axis: expected cost
   - separate panel per true cost ratio

### Outputs
- `preference_equivalence_frontier.csv`
- `risk_resource_frontier.png`
- `risk_cost_frontier_by_ratio.png`

## Experiment 4: Preference Conflict Map

### Purpose
Explicitly diagnose disagreement between cost-aware and risk-aware policy recommendations.

Question:

> For each cost ratio and risk budget, do the cost-only and CRC-only branches recommend the same quantile? If not, who is more conservative and what is the practical consequence?

### Pairing design
For each:
- cost ratio in `{1:1,3:1,5:1,9:1,19:1}`
- alpha in `{0.05,0.10,0.20}`

compare:

- `tau_cost`
- `tau_crc`
- `tau_gap = tau_crc - tau_cost`

### Required outputs per pair
- cost-policy actual violation
- CRC-policy actual violation
- cost-policy expected cost
- CRC-policy expected cost
- cost-policy mean overage
- CRC-policy mean overage
- conflict type

### Conflict types
Use a simple interpretable set:

- `aligned`
- `cost_policy_risk_feasible`
- `cost_policy_too_aggressive`
- `crc_more_conservative_than_needed`
- `crc_less_conservative_than_cost_policy`

### Outputs
- `preference_conflict_map.csv`
- `preference_conflict_heatmap_tau_gap.png`
- `preference_conflict_heatmap_violation_gap.png`

## Experiment 5: Risk-Screened Deployment

### Purpose
Build a deployment rule that reconciles monetary cost preferences and service-risk budgets.

### Version-1 rule hierarchy
This is where the prompt needs a stricter project adaptation.

#### Variant A: main rule for first implementation
Use the simple max-tau rule as the mandatory version-1 risk-screened policy:

```text
tau_screened = max(tau_cost, tau_crc_required)
```

This rule:
- is easy to interpret
- is deterministic
- does not introduce a new optimization layer beyond combining two existing preference languages

#### Variant B: second-layer extension
Calibration-side feasible-cost-min is useful, but should be treated as a second-pass enhancement, not the first mandatory implementation.

Use it only after Variant A is stable.

Definition:
1. identify CRC-feasible quantiles on calibration
2. among feasible candidates, choose the lowest calibration-side asymmetric cost under the specified cost ratio
3. evaluate on test

This variant is scientifically valuable, but it is a stronger new policy definition than simple preference reconciliation.

### First-run scope discipline
For the first Lenovo implementation and run:

- treat `risk_screened_max_tau` as mandatory
- treat `risk_screened_feasible_cost_min` as optional second-pass work
- keep the main comparison set limited to `raw_cost_aligned`, `one_sided_refined_cost_aligned`, `global_crc_violation`, and `horizon_crc_violation`
- keep safety-margin, empirical-selector, and block-CRC families out of the main tables unless they are explicitly needed for appendix or reviewer-defense summaries

### First-run required policy comparison
For each cost ratio and alpha:

- `cost_only`
- `crc_only`
- `risk_screened_max_tau`
- `symmetric_90_upper`

### Second-run optional additions
- `risk_screened_feasible_cost_min`
- safety-margin baseline

### Metrics
- selected quantile
- test violation
- risk target satisfied
- expected cost under the true cost ratio
- mean overage
- mean shortage
- shortage CVaR95
- cost increase vs cost-only
- overage reduction vs CRC-only
- violation reduction vs cost-only

### Outputs
- `risk_screened_deployment.csv`
- `risk_screened_by_horizon.csv`
- `risk_screened_policy_comparison.csv`
- `risk_screened_frontier.png`

## Experiment 6: Aggregate Procurement-Style Proxy

### Status
Optional only.
Default off for the first Lenovo run.

### Reason
This adds execution and framing complexity and is not needed to answer whether preference mapping itself is valuable.

### When to enable
Only after Experiments 1 to 5 are stable and only if the paper needs a small Applied Energy style aggregate provisioning proxy.

### Required caveat
If enabled, the report must explicitly state:

- summed station-level quantiles are not the true aggregate quantile
- this is an aggregate consequence proxy, not a formal market procurement model

## Bootstrap

### Reuse existing day-level bootstrap logic
Use the current reinforcement / CRC manifest logic where possible.

At minimum, bootstrap:

1. overage reduction of CRC / risk-screened vs `symmetric_90_upper`
2. expected cost difference between cost-only, CRC-only, and risk-screened
3. violation reduction when risk-screening corrects a too-aggressive cost-only policy
4. key frontier points

### Output
- `preference_mapping_bootstrap_ci.csv`

## Figures Required For First Paper-Facing Version

Mandatory:
- `cost_to_risk_mapping.png`
- `risk_to_cost_mapping.png`
- `risk_resource_frontier.png`
- `risk_cost_frontier_by_ratio.png`
- `preference_conflict_heatmap_tau_gap.png`
- `preference_conflict_heatmap_violation_gap.png`
- `risk_screened_frontier.png`

Optional:
- `aggregate_procurement_frontier.png`

## Tests To Add
- `tests/test_journal_preference_mapping.py`

Minimum tests:

1. exact cost-ratio to quantile mapping
2. implied ratio from quantile
3. no-test-label-selection metadata
4. max-rule screened tau is at least as conservative as both parents
5. feasible-cost-min selection uses calibration-only source tags
6. required frontier columns exist
7. required output files exist after smoke run
8. nonnegativity of cost / overage / shortage / CVaR

## Smoke Run
Support:

- `--max-calib-batches 1`
- `--max-test-windows 8`
- `--bootstrap-samples 20`

## Full Run Recommendation

### First full run
Do not enable aggregate procurement.

```bash
conda run -n py12 python scripts/journal/build_preference_mapping_experiments.py \
  --stage1-output-dir journal_results/shenzhen_multihorizon/warmstart_raw \
  --stage2-output-dir journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw \
  --stage4-output-dir journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw \
  --reinforcement-dir journal_results/shenzhen_multihorizon/reinforcement \
  --paper-assets-dir journal_results/shenzhen_multihorizon/paper_assets \
  --crc-v1-dir journal_results/shenzhen_multihorizon/crc_risk_control \
  --crc-followup-dir journal_results/shenzhen_multihorizon/crc_risk_control_followup \
  --output-dir journal_results/shenzhen_multihorizon/preference_mapping \
  --cost-ratios 1:1,3:1,5:1,9:1,19:1 \
  --alphas-violation 0.05,0.10,0.20 \
  --h1-alphas 0.03,0.04,0.05,0.06,0.08,0.10 \
  --bootstrap-samples 300 \
  --bootstrap-seed 20260601 \
  --enable-risk-screened true \
  --enable-aggregate-procurement false \
  --machine Lenovo
```

### Second run only if needed
If Variant A shows value, then add:
- feasible-cost-min
- optional aggregate proxy

## How To Judge Whether This Experiment Is Worth Keeping

### Keep as main paper contribution extension if:
- cost-aware and risk-aware policies show meaningful empirical disagreement
- risk-screened policies resolve that disagreement with better risk-cost trade-offs
- bootstrap confirms the main differences

### Keep only as robustness / appendix if:
- cost-aware and risk-aware policies are nearly equivalent on the main horizons
- risk-screened deployment collapses to one of the parents in most settings

## Final Paper-Facing Question This Package Must Answer
The final report should explicitly answer:

1. Are monetary cost-ratio preferences and service-risk-budget preferences empirically equivalent on this dataset?
2. When do they select the same quantile?
3. When does the cost-aware branch violate the risk budget?
4. When is CRC more conservative than economically necessary?
5. Does risk-screened deployment resolve the conflict effectively?
6. Is that resolution stable under day-level bootstrap?

## Final Execution Recommendation
This package is close to executable on the current branch, but it should be implemented in two layers:

1. mapping + frontier + conflict analysis
2. risk-screened deployment

Do not start with aggregate procurement.
Do not re-run all CRC logic from scratch unless necessary.
Prefer reusing current CRC / follow-up outputs and only reconstruct calibration-side selection where the new risk-screened feasible-cost-min rule requires it.
