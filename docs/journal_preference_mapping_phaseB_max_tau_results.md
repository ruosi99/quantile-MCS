# Preference Mapping Phase B: Max-Tau Risk-Screened Deployment

Date: 2026-05-20
Machine: Lenovo
Source output directory: `journal_results/shenzhen_multihorizon/preference_mapping/`
Focused Phase B output directory: `journal_results/shenzhen_multihorizon/preference_mapping_phaseB_max_tau/`

This note records the narrowed Phase B first-run experiment. The scope is intentionally limited to the interpretable `risk_screened_max_tau` rule:

```text
tau_screened = max(tau_cost, tau_crc_required)
```

No complex Phase B variants were added. In particular, this run excludes:

- `risk_screened_feasible_cost_min`
- safety-margin extensions
- aggregate procurement proxy

## Inputs

The focused Phase B folder was derived from the completed preference-mapping run:

- `risk_screened_policy_comparison.csv`
- `preference_mapping_bootstrap_ci.csv`
- `risk_screened_frontier.png`

The policy comparison keeps only the first-run comparison set:

- `cost_only`
- `crc_only`
- `risk_screened_max_tau`
- `symmetric_90_upper`

## Outputs

Focused output files:

- `phaseB_max_tau_metadata.json`
- `risk_screened_max_tau_parent_comparison.csv`
- `risk_screened_max_tau_only.csv`
- `risk_screened_target_satisfaction_counts.csv`
- `preference_mapping_bootstrap_ci.csv`
- `risk_screened_frontier.png`

## Target Satisfaction

Target satisfaction counts:

| Policy role | Target satisfied | Count |
|---|---:|---:|
| cost_only | false | 16 |
| cost_only | true | 14 |
| crc_only | false | 5 |
| crc_only | true | 25 |
| risk_screened_max_tau | false | 3 |
| risk_screened_max_tau | true | 27 |
| symmetric_90_upper | true | 30 |

Interpretation:

- Cost-only violates the requested service-risk budget in many settings.
- The max-tau screened rule improves target satisfaction from `14/30` to `27/30`.
- The remaining failures are cases where the calibration-selected quantile still slightly exceeds the risk target on the final test set.
- Symmetric 90 upper is the most conservative benchmark, but it is not the preferred economic policy because it often increases overage cost.

## Key Examples

For `5:1, alpha=0.10`:

| Scope | Policy | Selected quantile | Test violation | Target satisfied | Own expected cost | Cost increase vs cost-only |
|---|---|---:|---:|---|---:|---:|
| global | cost_only | 0.833333 | 0.102679 | false | 0.923726 | 0.000000 |
| global | risk_screened_max_tau | 0.833333 | 0.102679 | false | 0.923726 | 0.000000 |
| horizon | risk_screened_max_tau | 0.860000 | 0.084078 | true | 1.006842 | 0.083116 |

For `5:1, alpha=0.05`:

| Scope | Policy | Selected quantile | Test violation | Target satisfied | Own expected cost | Cost increase vs cost-only |
|---|---|---:|---:|---|---:|---:|
| global | cost_only | 0.833333 | 0.102679 | false | 0.923726 | 0.000000 |
| global | risk_screened_max_tau | 0.950000 | 0.035669 | true | 1.390813 | 0.467087 |
| horizon | risk_screened_max_tau | 0.940000 | 0.040367 | true | 1.371575 | 0.447849 |

For `5:1, alpha=0.20`:

| Scope | Policy | Selected quantile | Test violation | Target satisfied | Own expected cost | Cost increase vs cost-only |
|---|---|---:|---:|---|---:|---:|
| global | cost_only | 0.833333 | 0.102679 | true | 0.923726 | 0.000000 |
| global | risk_screened_max_tau | 0.833333 | 0.102679 | true | 0.923726 | 0.000000 |
| horizon | risk_screened_max_tau | 0.833333 | 0.102679 | true | 0.923726 | 0.000000 |

## Main Conclusion

The first-run Phase B result supports keeping `risk_screened_max_tau` as the main reconciliation rule.

It is useful because it is transparent:

- if cost preference is already conservative enough, it leaves the cost-only policy unchanged;
- if the service-risk budget is stricter, it raises the deployed quantile to the CRC-required level;
- horizon-wise screening is preferable around practical thresholds such as `5:1, alpha=0.10`, where global screening remains slightly above the target but horizon screening passes.

This phase does not justify adding the more complex feasible-cost-min rule yet. The current evidence is enough for a paper-facing first version of risk-screened deployment.

