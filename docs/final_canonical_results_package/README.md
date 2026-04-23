# Final Canonical Results Package

This package is the paper-facing summary of the final canonical experiments.

Use only:
- `canonical_main_results/`
- `canonical_baseline_results/`

Do not use older non-canonical result folders for tables, figures, or writing.

## Main Comparison Table

See:
- [main_comparison_table.csv](/C:/Users/100064422/PhD_projects/quntile/mcs-mcs_quantile/mcs-mcs_quantile/docs/final_canonical_results_package/main_comparison_table.csv)

Interpretation:
- `Historical Quantiles` is the lightweight probabilistic baseline on the canonical split.
- `Main Model (Raw Quantiles)` is the uncalibrated joint-quantile model.
- `Main Model (+ CQR)` is the final calibrated method for the paper.

## Interval Calibration Table

See:
- [interval_calibration_table.csv](/C:/Users/100064422/PhD_projects/quntile/mcs-mcs_quantile/mcs-mcs_quantile/docs/final_canonical_results_package/interval_calibration_table.csv)

Interpretation:
- This table isolates the effect of conformal calibration across the three interval settings.
- The key story is that CQR substantially improves coverage at all deltas while leaving interval width nearly unchanged.

## Final Figures

- Economic evaluation figure:
  [risk_aware_economic_evaluation.png](/C:/Users/100064422/PhD_projects/quntile/mcs-mcs_quantile/mcs-mcs_quantile/canonical_main_results/risk_aware_economic_evaluation.png)
- Hourly raw-vs-calibrated reliability figure:
  [time_of_day_diagnostic_delta0.1.png](/C:/Users/100064422/PhD_projects/quntile/mcs-mcs_quantile/mcs-mcs_quantile/canonical_main_results/time_of_day_diagnostic_delta0.1.png)

## Final Writing Takeaways

- The canonical joint quantile model achieves strong point accuracy on the canonical split (`MAPE 5.44`, `MAE 0.065`).
- Raw quantile intervals are under-covered, especially at the target 90% interval (`PICP 0.622`).
- Conformal calibration improves 90% interval coverage to `0.885` with almost unchanged width.
- The historical quantiles baseline is a credible low-cost probabilistic baseline, but it is much less accurate and dramatically less sharp than the main model.
- In risk-aware evaluation, the preferred decision quantile shifts upward as shortage cost increases, matching the intended newsvendor story.
- Hourly diagnostics show that calibration helps substantially at difficult hours, but a small set of hours remains harder than the rest.
