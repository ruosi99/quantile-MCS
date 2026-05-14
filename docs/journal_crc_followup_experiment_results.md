# CRC Follow-Up Experiment Results

Date: 2026-05-14
Machine: Lenovo
Branch: `codex/CRC_risk_control`
Commit: `e8d0c9c`
Output directory: `journal_results/shenzhen_multihorizon/crc_risk_control_followup/`

This document records the CRC follow-up experiments only. It was added as a new result note so the earlier CRC v1 results, stage documents, and previous result directories are not overwritten or edited.

## Purpose

The follow-up experiments were designed to defend the CRC deployment layer from three likely reviewer questions:

1. Is CRC doing something meaningfully different from a simple deterministic safety margin?
2. Does the finite-sample CRC correction matter, or is empirical calibration enough on this dataset?
3. Is the H1 short-horizon control issue a real robustness concern, and can a block-level CRC view change the decision?

The experiments used the warm-start multi-horizon model outputs and reconstructed calibration predictions from the Stage 1 checkpoint. Test labels were used only after the policy was selected.

## Code Added

New files added for this follow-up:

- `scripts/journal/build_crc_followup_experiments.py`
- `scripts/journal/run_crc_followup_experiments_warmstart.sh`
- `tests/test_journal_crc_followup.py`

The script reuses the existing CRC v1 helper functions where possible, then adds:

- deterministic safety-margin selection from median prediction plus calibration residual margin
- empirical-risk selector vs CRC-corrected selector for raw quantile policies
- H1-only alpha sensitivity analysis
- horizon-wise block CRC using calibration window blocks
- combined paper-facing summary tables and figures

Peak-hour/group-wise CRC was not executed in this follow-up. The calibration target-hour metadata was not reconstructed here, so forcing a peak-hour split would add indexing risk without a clear gain for this round.

## Verification

Unit tests:

```powershell
conda run -n py12 python -m pytest tests/test_journal_crc_followup.py -q -p no:cacheprovider
```

Result: `8 passed, 1 warning`.

Smoke run:

```powershell
conda run -n py12 python scripts/journal/build_crc_followup_experiments.py --stage1-output-dir journal_results/shenzhen_multihorizon/warmstart_raw --stage2-output-dir journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw --stage4-output-dir journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw --reinforcement-dir journal_results/shenzhen_multihorizon/reinforcement --paper-assets-dir journal_results/shenzhen_multihorizon/paper_assets --crc-v1-dir journal_results/shenzhen_multihorizon/crc_risk_control --output-dir journal_results/shenzhen_multihorizon/crc_risk_control_followup_smoke --alphas-violation 0.05,0.10 --h1-alphas 0.04,0.05,0.10 --cost-ratios 3:1,5:1 --bootstrap-samples 20 --bootstrap-seed 20260521 --max-calib-batches 1 --max-test-windows 8 --enable-block-crc true --machine Lenovo
```

Full run:

```powershell
conda run -n py12 python scripts/journal/build_crc_followup_experiments.py --stage1-output-dir journal_results/shenzhen_multihorizon/warmstart_raw --stage2-output-dir journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw --stage4-output-dir journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw --reinforcement-dir journal_results/shenzhen_multihorizon/reinforcement --paper-assets-dir journal_results/shenzhen_multihorizon/paper_assets --crc-v1-dir journal_results/shenzhen_multihorizon/crc_risk_control --output-dir journal_results/shenzhen_multihorizon/crc_risk_control_followup --alphas-violation 0.05,0.10,0.20 --h1-alphas 0.03,0.04,0.05,0.06,0.08,0.10 --cost-ratios 3:1,5:1,9:1,19:1 --bootstrap-samples 300 --bootstrap-seed 20260521 --enable-block-crc true --machine Lenovo
```

No leakage checks recorded in `crc_followup_metadata.json`:

- calibration source: reconstructed from Stage 1 checkpoint and calibration split
- test source: saved Stage 1 test arrays
- selection uses test labels: `false`
- test prediction shape: `[829, 1682, 5, 13]`
- test label shape: `[829, 1682, 5]`

## Main Output Files

Core tables:

- `crc_safety_margin_test_summary.csv`
- `crc_empirical_vs_corrected_test_summary.csv`
- `crc_h1_alpha_sensitivity.csv`
- `crc_blocked_test_summary.csv`
- `crc_followup_policy_comparison.csv`
- `crc_followup_risk_efficiency_table.csv`
- `crc_followup_bootstrap_ci.csv`
- `crc_followup_metadata.json`
- `crc_followup_report.md`

Figures:

- `crc_safety_margin_vs_crc.png`
- `crc_empirical_vs_crc_selector.png`
- `crc_h1_alpha_sensitivity.png`
- `crc_blocked_vs_station_window.png`
- `crc_followup_risk_efficiency_frontier.png`
- `crc_followup_expected_cost_by_policy.png`
- `crc_followup_shortage_cvar_by_policy.png`
- `crc_followup_violation_vs_overage.png`

## Result 1: Safety Margin Is Usually More Conservative

Representative 5:1 cost results:

| Policy | Violation rate | Mean overage | Expected cost 5:1 |
|---|---:|---:|---:|
| safety margin CRC global alpha 0.05 | 0.020790 | 2.397393 | 2.694715 |
| safety margin CRC horizon alpha 0.05 | 0.022286 | 2.289075 | 2.565703 |
| safety margin empirical global alpha 0.10 | 0.090367 | 0.655280 | 1.345925 |
| safety margin CRC horizon alpha 0.20 | 0.171317 | 0.348376 | 1.228429 |
| raw quantile CRC global alpha 0.10 | 0.102679 | 0.562110 | 0.923726 |
| raw quantile CRC horizon alpha 0.10 | 0.090430 | 0.709945 | 1.003747 |

Interpretation:

- A deterministic safety margin can control shortage risk, but strict settings are very conservative.
- At alpha 0.05, safety-margin CRC pushes violation well below the target but pays a large overage cost.
- For the practical 5:1 cost setting, raw quantile CRC global alpha 0.10 has lower expected cost than comparable safety-margin policies.
- This supports keeping CRC as a quantile-policy selection layer rather than presenting it as only an additive margin.

## Result 2: Empirical Risk and CRC-Corrected Selection Coincide Here

For the raw quantile policy family, empirical-risk and CRC-corrected selectors chose the same policies in the tested settings:

| Policy | Violation rate | Mean overage | Expected cost 5:1 |
|---|---:|---:|---:|
| raw quantile CRC global alpha 0.05 | 0.035669 | 1.275474 | 1.390813 |
| raw quantile empirical global alpha 0.05 | 0.035669 | 1.275474 | 1.390813 |
| raw quantile CRC global alpha 0.10 | 0.102679 | 0.562110 | 0.923726 |
| raw quantile empirical global alpha 0.10 | 0.102679 | 0.562110 | 0.923726 |
| raw quantile CRC horizon alpha 0.10 | 0.090430 | 0.709945 | 1.003747 |
| raw quantile empirical horizon alpha 0.10 | 0.090430 | 0.709945 | 1.003747 |

Interpretation:

- The calibration set is large enough that the CRC correction is numerically small for this policy class.
- The value of CRC in the paper should therefore be framed as a principled finite-sample deployment protocol, not as a large empirical performance boost over the empirical selector on this dataset.
- This is still useful: the same selected policy gets a defensible risk-control interpretation.

## Result 3: H1 Needs Stricter Alpha for Robust Control

H1-only alpha sensitivity:

| Alpha | Selected quantile | CRC upper | H1 test violation | H1 mean overage | Expected cost 5:1 |
|---:|---:|---:|---:|---:|---:|
| 0.03 | 0.95 | 0.027661 | 0.043102 | 0.477604 | 0.539516 |
| 0.04 | 0.95 | 0.027661 | 0.043102 | 0.477604 | 0.539516 |
| 0.05 | 0.90 | 0.042469 | 0.066595 | 0.340217 | 0.443328 |
| 0.06 | 0.90 | 0.042469 | 0.066595 | 0.340217 | 0.443328 |
| 0.08 | 0.833333 | 0.069523 | 0.098348 | 0.251299 | 0.405643 |
| 0.10 | 0.80 | 0.084311 | 0.112161 | 0.231822 | 0.402077 |

Interpretation:

- If H1 must stay below 5% shortage violation on test, alpha 0.05 is not conservative enough; it selects q0.90 and reaches 0.066595 test violation.
- Alpha 0.03 or 0.04 selects q0.95 and keeps H1 test violation at 0.043102.
- This gives a clean explanation for the H1 concern: the available quantile grid creates discrete jumps, and H1 is sensitive to calibration-test shift near the 5% target.
- For the paper, H1 can be discussed as a stricter service-level case where the target risk should be set below the desired deployment threshold.

## Result 4: Block CRC Makes Only a Modest Difference

Block CRC used calibration window blocks and averaged station-level violations within each block.

| Policy | Violation rate | Block violation rate | Mean overage | Expected cost 5:1 |
|---|---:|---:|---:|---:|
| block CRC horizon alpha 0.05 | 0.040367 | 0.040367 | 1.247997 | 1.371575 |
| block CRC horizon alpha 0.10 | 0.088721 | 0.088721 | 0.720954 | 1.005236 |
| block CRC horizon alpha 0.20 | 0.165967 | 0.165967 | 0.378273 | 0.942606 |

Interpretation:

- Block CRC is a useful robustness check, but it does not overturn the CRC v1 conclusion.
- At alpha 0.10 it becomes slightly more conservative than station-window horizon CRC because H6 moves to q0.833333.
- The main paper can mention block CRC as a sensitivity analysis rather than the primary method.

## Paper Direction

The follow-up results strengthen the CRC story, but they also narrow how it should be claimed.

Recommended framing:

- CRC is a deployment-time risk-control layer for choosing quantile policies under explicit shortage-risk targets.
- On this dataset, the finite-sample correction does not visibly outperform empirical risk selection because calibration is large; its role is methodological validity and leakage-safe deployment.
- The main empirical benefit comes from choosing a risk/cost-appropriate quantile policy, especially compared with median or overly conservative safety-margin rules.
- Safety margins are a fair baseline, but strict safety-margin policies are inefficient at alpha 0.05.
- H1 should be reported as a stricter sensitivity case; if the paper needs H1 under 5% violation, alpha 0.03 or 0.04 is the defensible setting.

Current go/no-go assessment:

- Keep CRC in the paper as a service-risk control contribution.
- Do not claim CRC correction itself creates a large empirical gain over empirical selection on this dataset.
- Use safety-margin results to show the proposed quantile-policy CRC is more risk-efficient than simple margin inflation.
- Use block CRC and H1 alpha sensitivity as reviewer-defense experiments.

## Sync Notes

To reproduce or inspect this follow-up on Dell, copy the new result directory:

```text
journal_results/shenzhen_multihorizon/crc_risk_control_followup/
```

The smoke directory is optional and can be skipped unless debugging is needed:

```text
journal_results/shenzhen_multihorizon/crc_risk_control_followup_smoke/
```

The calibration arrays themselves were not saved. The run records hashes and a manifest instead:

- `crc_calib_hashes.json`
- `crc_calib_manifest.json`

