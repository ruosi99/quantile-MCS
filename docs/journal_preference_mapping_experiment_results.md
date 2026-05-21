# Preference Mapping Experiment Results

Date: 2026-05-20
Machine: Lenovo
Branch recorded by run: `codex/CRC_risk_control`
Commit recorded by run: `49164f7`
Output directory: `journal_results/shenzhen_multihorizon/preference_mapping/`

This document records the first preference-mapping run. It is a new result note and does not overwrite earlier Stage 4, CRC, CRC follow-up, misspecification, or paper-assets records.

## Purpose

This experiment tests whether monetary cost-ratio preferences and service-risk-budget preferences are empirically equivalent deployment languages for the current multi-horizon EV charging forecast outputs.

It uses existing outputs from:

- `warmstart_raw/`
- `stage2_cqr/warmstart_raw/`
- `stage4_decision/warmstart_raw/`
- `crc_risk_control/`
- `crc_risk_control_followup/`
- `reinforcement/`

No new forecasting model is trained. Cost-aware policies are selected analytically from the cost ratio, CRC policies are imported from existing calibration-only CRC outputs, and test labels are used only for final evaluation.

## Code Added

- `scripts/journal/build_preference_mapping_experiments.py`
- `scripts/journal/run_preference_mapping_experiments_warmstart.sh`
- `tests/test_journal_preference_mapping.py`

## Verification

Unit test command:

```powershell
conda run -n py12 python -m pytest tests/test_journal_preference_mapping.py -q -p no:cacheprovider
```

Result: `9 passed, 1 warning`.

Smoke command used 8 test windows and 20 bootstrap samples. It completed successfully and wrote:

- `journal_results/shenzhen_multihorizon/preference_mapping_smoke/`

Full command:

```powershell
conda run -n py12 python scripts/journal/build_preference_mapping_experiments.py --stage1-output-dir journal_results/shenzhen_multihorizon/warmstart_raw --stage2-output-dir journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw --stage4-output-dir journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw --reinforcement-dir journal_results/shenzhen_multihorizon/reinforcement --paper-assets-dir journal_results/shenzhen_multihorizon/paper_assets --crc-v1-dir journal_results/shenzhen_multihorizon/crc_risk_control --crc-followup-dir journal_results/shenzhen_multihorizon/crc_risk_control_followup --output-dir journal_results/shenzhen_multihorizon/preference_mapping --cost-ratios 1:1,3:1,5:1,9:1,19:1 --alphas-violation 0.05,0.10,0.20 --h1-alphas 0.03,0.04,0.05,0.06,0.08,0.10 --bootstrap-samples 300 --bootstrap-seed 20260601 --enable-risk-screened true --enable-aggregate-procurement false --machine Lenovo
```

Full run completed with test prediction shape `[829, 1682, 5, 13]` and test label shape `[829, 1682, 5]`.

## Output Files

Core tables:

- `cost_to_risk_mapping.csv`
- `cost_to_risk_mapping_by_horizon.csv`
- `risk_to_cost_mapping.csv`
- `risk_to_cost_mapping_by_horizon.csv`
- `h1_risk_to_cost_mapping.csv`
- `preference_equivalence_frontier.csv`
- `preference_conflict_map.csv`
- `risk_screened_deployment.csv`
- `risk_screened_by_horizon.csv`
- `risk_screened_policy_comparison.csv`
- `preference_mapping_bootstrap_ci.csv`
- `preference_mapping_metadata.json`
- `preference_mapping_report.md`

Figures:

- `cost_to_risk_mapping.png`
- `risk_to_cost_mapping.png`
- `risk_resource_frontier.png`
- `risk_cost_frontier_by_ratio.png`
- `preference_conflict_heatmap_tau_gap.png`
- `preference_conflict_heatmap_violation_gap.png`
- `risk_screened_frontier.png`

## Phase A Gate

Phase A passed.

Recorded gate:

```json
{
  "phase_a_gate_pass": true,
  "reason": "nontrivial preference mismatch",
  "too_aggressive_pair_count": 15,
  "nontrivial_tau_gap_pair_count": 24,
  "max_cost_policy_violation_minus_alpha": 0.3951074242422069
}
```

Interpretation:

- Cost-aware and risk-aware preferences are not empirically equivalent.
- There are repeated settings where the cost-only policy violates the requested service-risk budget.
- Running Phase B `risk_screened_max_tau` was justified.
- Aggregate procurement remained disabled, as planned.

## Cost-To-Risk Mapping

Raw cost-aligned quantile results:

| Cost ratio | Quantile | Nominal risk | Test violation | Mean overage | Mean shortage | Own expected cost |
|---|---:|---:|---:|---:|---:|---:|
| 1:1 | 0.500000 | 0.500000 | 0.445107 | 0.202002 | 0.222013 | 0.424015 |
| 3:1 | 0.750000 | 0.250000 | 0.145192 | 0.409935 | 0.103651 | 0.720889 |
| 5:1 | 0.833333 | 0.166667 | 0.102679 | 0.562110 | 0.072323 | 0.923726 |
| 9:1 | 0.900000 | 0.100000 | 0.064492 | 0.839526 | 0.042606 | 1.222980 |
| 19:1 | 0.950000 | 0.050000 | 0.035669 | 1.275474 | 0.023068 | 1.713762 |

Interpretation:

- Higher cost ratios produce more conservative quantiles and lower realized violation, as expected.
- The realized violation is below the nominal tail probability for all listed cost ratios.
- Cost-ratio language alone does not directly encode a desired service-risk budget; for example, `5:1` gives realized violation `0.102679`, which is close to but above a strict `0.10` service target.

## Risk-To-Cost Mapping

Violation-CRC results:

| CRC scope | Alpha | Mean selected quantile | Implied cost ratio | Test violation | Mean overage | Mean shortage |
|---|---:|---:|---:|---:|---:|---:|
| global | 0.05 | 0.950000 | 19.000000 | 0.035669 | 1.275474 | 0.023068 |
| horizon | 0.05 | 0.940000 | 15.666667 | 0.040367 | 1.247997 | 0.024716 |
| global | 0.10 | 0.833333 | 5.000000 | 0.102679 | 0.562110 | 0.072323 |
| horizon | 0.10 | 0.840000 | 5.250000 | 0.090430 | 0.709945 | 0.058760 |
| global | 0.20 | 0.700000 | 2.333333 | 0.165967 | 0.378273 | 0.112867 |
| horizon | 0.20 | 0.700000 | 2.333333 | 0.165967 | 0.378273 | 0.112867 |

Interpretation:

- `alpha=0.05` maps to a very conservative economic preference, approximately `16:1` to `19:1`.
- `alpha=0.10` maps closely to `5:1`, but the global policy lands slightly above the target on test (`0.102679`), while horizon CRC stays below (`0.090430`).
- `alpha=0.20` maps to a much less conservative preference, roughly `2.33:1`.

## H1 Risk-To-Cost Sensitivity

H1-specific mapping from the CRC follow-up:

| Alpha | Selected quantile | Implied cost ratio | H1 test violation | H1 mean overage | Expected cost 5:1 |
|---:|---:|---:|---:|---:|---:|
| 0.03 | 0.95 | 19.00 | 0.043102 | 0.477604 | 0.539516 |
| 0.04 | 0.95 | 19.00 | 0.043102 | 0.477604 | 0.539516 |
| 0.05 | 0.90 | 9.00 | 0.066595 | 0.340217 | 0.443328 |
| 0.06 | 0.90 | 9.00 | 0.066595 | 0.340217 | 0.443328 |
| 0.08 | 0.833333 | 5.00 | 0.098348 | 0.251299 | 0.405643 |
| 0.10 | 0.80 | 4.00 | 0.112161 | 0.231822 | 0.402077 |

Interpretation:

- H1 remains a stricter robustness case.
- If H1 must stay below 5 percent realized violation, alpha needs to be tightened to `0.03` or `0.04`, which selects q0.95.

## Preference Conflict Map

Conflict counts:

| CRC scope | Conflict type | Count |
|---|---|---:|
| global | aligned | 2 |
| global | cost_policy_risk_feasible | 6 |
| global | cost_policy_too_aggressive | 7 |
| horizon | cost_policy_risk_feasible | 7 |
| horizon | cost_policy_too_aggressive | 8 |

Examples:

- `1:1` is too aggressive for all tested alpha values; realized cost-policy violation is `0.445107`.
- `3:1` is too aggressive for alpha `0.05` and `0.10`.
- `5:1` is too aggressive for alpha `0.05`, and for horizon alpha `0.10` the risk-screened rule tightens the policy enough to pass.
- `9:1` is too aggressive for alpha `0.05` but feasible for alpha `0.10` and `0.20`.
- `19:1` is already conservative enough for all tested alpha values.

## Risk-Screened Deployment

The first implemented reconciliation rule is:

```text
risk_screened_max_tau = max(tau_cost, tau_crc_required)
```

Main findings:

- When cost-only is already conservative enough, risk-screened collapses to the cost-only policy.
- When the risk budget is stricter than the economic preference, risk-screened collapses to the CRC-required quantile.
- For `5:1, alpha=0.10`, global CRC and cost-only both use q0.833333 and test violation is `0.102679`, slightly above the nominal `0.10` threshold. Horizon risk-screening raises the mean selected quantile to `0.86`, lowers violation to `0.084078`, and raises expected cost from `0.923726` to `1.006842`.
- For `5:1, alpha=0.05`, risk-screened becomes q0.95/global or near q0.94/horizon, reducing violation from `0.102679` to `0.035669` or `0.040367`, but increasing 5:1 expected cost by about `0.467087` or `0.447849`.

Policy target satisfaction counts in `risk_screened_policy_comparison.csv`:

| Policy role | Satisfies target | Count |
|---|---:|---:|
| cost_only | false | 16 |
| cost_only | true | 14 |
| crc_only | false | 5 |
| crc_only | true | 25 |
| risk_screened_max_tau | false | 3 |
| risk_screened_max_tau | true | 27 |
| symmetric_90_upper | true | 30 |

Interpretation:

- Risk screening fixes most, but not all, realized test violations because policies are selected without test labels and the realized test sample can slightly exceed the calibration target.
- Horizon risk screening is more robust than global risk screening around alpha `0.10`.
- Symmetric 90 upper is safest but often economically expensive, so it should remain a conservative benchmark rather than the preferred deployment rule.

## Paper Direction

This experiment is worth keeping as a deployment-layer bridge, not as a new model contribution.

Suggested framing:

- Monetary cost ratios and service-risk budgets are related but not interchangeable.
- The mapping is interpretable: `5:1` approximately corresponds to a 10 percent violation budget, while `19:1` corresponds to approximately 5 percent.
- Preference conflict is real in the current test results, especially for low and moderate cost ratios under strict risk budgets.
- A simple risk-screened max-tau rule provides a transparent reconciliation mechanism.
- Horizon-wise risk screening should be preferred over global screening when the paper discusses practical service-risk control.

Recommended next step:

- Use the generated figures and tables as paper-facing evidence.
- Do not run aggregate procurement yet.
- Consider a small follow-up only if we want to replace `max_tau` with the optional calibration-side feasible-cost-min rule.

## Sync Notes

To inspect the full result on Dell, copy:

```text
journal_results/shenzhen_multihorizon/preference_mapping/
```

The smoke directory is optional:

```text
journal_results/shenzhen_multihorizon/preference_mapping_smoke/
```

