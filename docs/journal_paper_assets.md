# Journal Paper Evidence Assets

## Metadata
- Date: 2026-05-06
- Machine: Lenovo, Windows GPU experiment machine
- Branch: `multi_horizon_journal`
- Commit at export time: `bb41b2b`
- Output directory: `journal_results/shenzhen_multihorizon/paper_assets/`

## Purpose
This export turns the current journal evidence into paper-facing tables and figures.
It covers:

- Stage 2.75 boundary-aware reliability decomposition
- Stage 4.5 decision-value attribution
- bootstrap confidence intervals for the main decision and reliability claims

No Stage 3 localized calibration was implemented in this step.

## Code Added
- `scripts/journal/build_paper_evidence_assets.py`
- `scripts/journal/run_paper_evidence_assets_warmstart.sh`
- `tests/test_journal_paper_assets.py`

## Verification
Test command:

```powershell
conda run -n py12 python -m pytest tests/test_journal_paper_assets.py tests/test_journal_stage4_decision.py tests/test_journal_stage2_diagnostics.py -q -p no:cacheprovider
```

Result:

- `14 passed`
- `1 warning`

## Full Lenovo Command
```powershell
conda run -n py12 python scripts/journal/build_paper_evidence_assets.py --stage1-output-dir journal_results/shenzhen_multihorizon/warmstart_raw --stage2-output-dir journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw --stage2-diagnostics-dir journal_results/shenzhen_multihorizon/stage2_diagnostics/warmstart_raw --stage4-output-dir journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw --output-dir journal_results/shenzhen_multihorizon/paper_assets --deltas 0.1 --primary-delta 0.1 --bootstrap-samples 300 --bootstrap-seed 20260506 --machine Lenovo
```

## Generated Files
- `boundary_aware_reliability_table.csv`
- `coverage_gain_decomposition.csv`
- `positive_hard_cell_summary.csv`
- `decision_value_attribution.csv`
- `decision_value_attribution_by_horizon.csv`
- `decision_cost_by_quantile_curve.csv`
- `bootstrap_ci_summary.csv`
- `coverage_gain_decomposition.png`
- `positive_demand_ace_heatmap_global_cqr.png`
- `decision_value_attribution.png`
- `decision_cost_by_quantile_curve.png`
- `paper_assets_metadata.json`

## Boundary-Aware Reliability Result
Primary method: `global_cqr`

Primary interval: 90 percent nominal coverage, `delta = 0.1`

| Horizon | PICP all | PICP zero | PICP positive | PICP top10 | PICP top5 | ACE positive | ZBR share |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.892134 | 0.920105 | 0.879434 | 0.944916 | 0.951522 | 0.020566 | 0.999940 |
| 3 | 0.918925 | 0.918626 | 0.919061 | 0.940386 | 0.941589 | 0.019061 | 0.999967 |
| 6 | 0.924940 | 0.897854 | 0.937198 | 0.954146 | 0.957066 | 0.037198 | 0.999974 |
| 12 | 0.917449 | 0.843494 | 0.950817 | 0.951363 | 0.954250 | 0.050817 | 0.999992 |
| 24 | 0.913492 | 0.827990 | 0.951853 | 0.962760 | 0.970046 | 0.051853 | 0.999989 |

Interpretation:

- Tail-demand coverage is not the main failure mode at 90 percent.
- The remaining positive-demand issue is mainly localized, especially H1 hard target-hour cells.
- `ZBR_share` is essentially 1 across horizons, supporting the zero-boundary rescue explanation.

## Coverage-Gain Decomposition
For global CQR at 90 percent:

| Horizon | Raw PICP | Zero-boundary rescue gain | Positive rescue gain | Reconstructed calibrated PICP | Zero share of rescues |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.604806 | 0.287311 | 0.000017 | 0.892134 | 0.999940 |
| 3 | 0.632331 | 0.286585 | 0.000009 | 0.918925 | 0.999967 |
| 6 | 0.645196 | 0.279737 | 0.000007 | 0.924940 | 0.999974 |
| 12 | 0.655198 | 0.262250 | 0.000002 | 0.917449 | 0.999992 |
| 24 | 0.657056 | 0.256433 | 0.000003 | 0.913492 | 0.999989 |

This is now the clearest formal evidence for the paper's boundary-aware reliability narrative.

## Positive-Demand Hard Cells
The worst positive-demand cells remain concentrated at H1:

- H1, hour 18: PICP about `0.78037`, ACE about `0.11963`
- H1, hour 10: PICP about `0.78195`, ACE about `0.11805`
- H1, hour 14: PICP about `0.80173-0.80175`, ACE about `0.09825`

This supports a narrow localized Stage 3 only if the paper needs an extra targeted reliability experiment.

## Decision-Value Attribution
The main decision-value claim is strongly supported.

| Cost ratio | Median cost | Raw target cost | Horizon CQR cost | Quantile-choice gain | Horizon calibration gain | Quantile-choice share |
|---|---:|---:|---:|---:|---:|---:|
| `3:1` | 0.868041 | 0.720889 | 0.720752 | 0.147152 | 0.000136 | 0.999076 |
| `5:1` | 1.312067 | 0.923726 | 0.923485 | 0.388340 | 0.000241 | 0.999379 |
| `9:1` | 2.200118 | 1.222980 | 1.222530 | 0.977138 | 0.000450 | 0.999540 |
| `19:1` | 4.420247 | 1.713762 | 1.712642 | 2.706485 | 0.001120 | 0.999586 |

Interpretation:

- The expected-cost improvement is overwhelmingly from using the cost-aligned target quantile.
- One-sided conformal calibration gives a consistent but very small extra gain.
- Calibration should be framed as a lightweight safeguard/refiner, not the main decision-value generator.

## Bootstrap Confidence Intervals
Bootstrap type:

- test-window-level bootstrap
- 300 samples
- seed `20260506`

Reason:

- the saved Stage 1/4 result arrays do not currently expose a reliable day or calendar index
- day-level or station-day cluster bootstrap can be added later if the date mapping is exported

Selected decision CIs:

| Metric | Cost ratio | Point | 95% CI |
|---|---|---:|---:|
| raw reduction vs median | `3:1` | 0.147152 | [0.141970, 0.152030] |
| raw reduction vs median | `5:1` | 0.388340 | [0.377349, 0.398624] |
| raw reduction vs median | `9:1` | 0.977138 | [0.950152, 1.002247] |
| raw reduction vs median | `19:1` | 2.706485 | [2.655120, 2.771543] |
| horizon calibration gain vs raw | `3:1` | 0.000136 | [0.000134, 0.000138] |
| horizon calibration gain vs raw | `19:1` | 0.001120 | [0.001062, 0.001185] |
| quantile-choice share | `3:1` | 0.999076 | [0.999032, 0.999119] |
| quantile-choice share | `19:1` | 0.999586 | [0.999559, 0.999614] |

Selected reliability CIs for global CQR:

| Horizon | PICP positive | 95% CI | ZBR share | 95% CI |
|---:|---:|---:|---:|---:|
| 1 | 0.879434 | [0.874151, 0.884089] | 0.999940 | [0.999915, 0.999962] |
| 3 | 0.919061 | [0.915136, 0.922957] | 0.999967 | [0.999950, 0.999983] |
| 6 | 0.937198 | [0.935040, 0.939236] | 0.999974 | [0.999959, 0.999990] |
| 12 | 0.950817 | [0.948869, 0.952738] | 0.999992 | [0.999981, 1.000000] |
| 24 | 0.951853 | [0.949827, 0.953760] | 0.999989 | [0.999976, 0.999997] |

## Decision
The next paper-facing step should be figure/table polishing and optional day-level bootstrap support if date indices are exported.

Do not start broad Stage 3 by default.

If Stage 3 is later pursued, it should be narrow:

- H1 only
- positive-demand only
- hard hour cells only
- judged by worst-cell reliability and possibly hard-cell decision cost

