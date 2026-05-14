# CRC Risk-Control Experiment Results

## Metadata
- Date: 2026-05-14
- Machine: Lenovo
- Branch at execution: `codex/CRC_risk_control`
- Commit at execution: `e8d0c9c`
- Output directory: `journal_results/shenzhen_multihorizon/crc_risk_control/`

## Code Changes
This run added the CRC risk-control experiment implementation:

- `scripts/journal/build_crc_risk_control_experiments.py`
- `scripts/journal/run_crc_risk_control_warmstart.sh`
- `tests/test_journal_crc_risk_control.py`

The implementation follows the no-leakage rule:

- calibration predictions are reconstructed from the Stage 1 checkpoint and calibration split
- CRC quantile selections are made on calibration predictions and labels only
- final policy evaluation uses saved Stage 1 test predictions and labels

Version-1 scope:

- losses: violation and normalized shortage
- candidates: raw quantile grid only
- baselines: median, symmetric 90 upper, raw cost-aligned quantiles, one-sided refined cost-aligned quantiles, global CRC, horizon-wise CRC
- weighted loss: not implemented in v1
- safety-margin baseline: not implemented in v1
- calibration arrays: not saved; hashes and manifest were saved instead

## Verification
Unit tests passed:

```powershell
conda run -n py12 python -m pytest tests/test_journal_crc_risk_control.py -q -p no:cacheprovider
```

Result:

- `7 passed`
- one existing `torch_geometric` deprecation warning

Smoke run also passed with:

- `--max-calib-batches 1`
- `--max-test-windows 8`
- `--bootstrap-samples 20`

## Official Run Command
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

## No-Leakage Reproducibility
Calibration reconstruction:

- `calib_quantiles`: `(828, 1682, 5, 13)`
- `calib_labels`: `(828, 1682, 5)`
- `test_quantiles`: `(829, 1682, 5, 13)`
- `test_labels`: `(829, 1682, 5)`
- calibration source: `reconstructed_from_stage1_checkpoint_and_calibration_split`
- test source: saved Stage 1 test arrays
- `selection_uses_test_labels`: `false`

Saved reproducibility files:

- `crc_calibration_manifest.json`
- `crc_calibration_hashes.json`

## Generated Files
- `crc_selection_table.csv`
- `crc_test_risk_summary.csv`
- `crc_by_horizon.csv`
- `crc_baseline_comparison.csv`
- `crc_risk_efficiency_frontier.csv`
- `crc_cost_ratio_evaluation.csv`
- `crc_bootstrap_ci_day_level.csv`
- `crc_calibration_manifest.json`
- `crc_calibration_hashes.json`
- `crc_metadata.json`
- `crc_experiment_report.md`
- `crc_test_risk_vs_alpha.png`
- `crc_selected_quantiles_by_alpha.png`
- `crc_risk_efficiency_frontier.png`
- `crc_expected_cost_by_policy.png`
- `crc_shortage_cvar_by_policy.png`

## CRC Selection Summary
Violation-loss CRC selects intuitive quantiles:

- global alpha `0.05`: `q0.95`
- global alpha `0.10`: `q0.8333`
- global alpha `0.20`: `q0.70`
- horizon-wise alpha `0.05`: H1 selects `q0.90`, other horizons select `q0.95`
- horizon-wise alpha `0.10`: H1/H3/H6 select `q0.80`, H12/H24 select `q0.90`

Normalized-shortage CRC selects much lower quantiles:

- global alpha `0.03`: `q0.30`
- global alpha `0.05`: `q0.20`
- global alpha `0.10`: `q0.05`

This is scientifically useful: normalized shortage risk is not the same as violation risk. It can be satisfied with low quantiles because the shortage is normalized by a large calibration-side positive-demand scale.

## Test Risk Results
Key pooled test metrics:

| policy | violation rate | normalized shortage risk | mean decision | mean overage | CVaR95 shortage |
|---|---:|---:|---:|---:|---:|
| median_decision | 0.445107 | 0.013471 | 2.665455 | 0.202002 | 3.096228 |
| symmetric_90_upper | 0.035665 | 0.001400 | 3.937920 | 1.275521 | 0.023066 |
| horizon_crc_violation_alpha0.05 | 0.040367 | 0.001501 | 3.908747 | 1.247997 | 0.024716 |
| horizon_crc_violation_alpha0.10 | 0.090430 | 0.003572 | 3.336651 | 0.709945 | 1.103513 |
| horizon_crc_violation_alpha0.20 | 0.165967 | 0.006868 | 2.950872 | 0.378273 | 1.880872 |
| global_crc_normalized_shortage_alpha0.03 | 0.566853 | 0.024852 | 2.382343 | 0.110565 | 4.881278 |
| horizon_crc_normalized_shortage_alpha0.05 | 0.609446 | 0.037458 | 2.135555 | 0.074188 | 6.405873 |

Violation-loss CRC has the cleanest interpretation:

- alpha `0.05`: pooled violation `0.040367`, below target
- alpha `0.10`: pooled violation `0.090430`, below target
- alpha `0.20`: pooled violation `0.165967`, below target

By-horizon nuance:

- horizon-wise violation alpha `0.05` has pooled control, but H1 test violation is `0.066595`
- this means the marginal pooled result is strong, but strict per-horizon finite-sample test control is mixed at H1

Normalized-shortage CRC controls normalized shortage risk but can allow high violation rates:

- alpha `0.03`: normalized shortage risk around `0.024-0.025`, but violation around `0.57`
- alpha `0.05`: normalized shortage risk around `0.037`, but violation around `0.61`

This should be framed as a different risk target, not a service-violation policy.

## Resource Efficiency Versus Symmetric 90 Upper
Compared with `symmetric_90_upper`:

- symmetric 90 upper mean overage: `1.275521`
- horizon CRC violation alpha `0.05` mean overage: `1.247997`
- horizon CRC violation alpha `0.10` mean overage: `0.709945`
- horizon CRC violation alpha `0.20` mean overage: `0.378273`

Day-level bootstrap overage reduction versus `symmetric_90_upper`:

| policy | point | 95% CI |
|---|---:|---:|
| horizon_crc_violation_alpha0.05 | 0.027615 | [0.025266, 0.029751] |
| horizon_crc_violation_alpha0.10 | 0.567334 | [0.556668, 0.578603] |
| horizon_crc_violation_alpha0.20 | 0.900383 | [0.873838, 0.932877] |

This supports the claim that violation-risk CRC can reduce conservativeness relative to interval-upper-bound deployment while maintaining a service-risk target.

## Expected Cost Comparison
Under evaluation cost ratio `5:1`:

- one-sided refined `5:1`: `0.923485`
- raw cost-aligned `5:1`: `0.923726`
- global CRC violation alpha `0.10`: `0.923726`
- horizon CRC violation alpha `0.10`: `1.003747`
- symmetric 90 upper: `1.390851`
- horizon CRC violation alpha `0.05`: `1.371575`

Interpretation:

- CRC is not better than cost-aligned quantiles when the monetary cost ratio is known and reliable
- CRC is clearly better than symmetric 90 upper for some service-risk targets
- global violation alpha `0.10` effectively recovers the `5:1` raw cost-aligned policy in this grid

## Scientific Interpretation
The CRC result is strong enough to keep as a decision-layer comparison, but not as a replacement for the current main narrative.

What it proves well:

- no-leakage CRC selection is executable on the current branch
- violation-risk CRC can hit pooled empirical test violation targets
- CRC can reduce overage/resource use relative to conservative interval-upper-bound deployment
- CRC gives a credible alternative when service-risk targets are easier to specify than exact monetary cost ratios

What it does not prove:

- CRC does not dominate cost-aligned quantile decisions when the cost ratio is known
- normalized-shortage CRC should not be sold as violation control
- alpha `0.05` horizon-wise violation control has an H1 finite-sample miss despite pooled control

Recommended paper positioning:

- use CRC as an alternative deployment layer for service-risk targets
- keep cost-aligned quantiles as the expected-cost tool when cost ratios are reliable
- keep boundary-aware diagnostics as the reliability narrative
- do not reopen broad Stage 3 calibration because of this result
