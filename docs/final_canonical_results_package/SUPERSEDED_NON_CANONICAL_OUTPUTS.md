# Superseded Non-Canonical Outputs

The following outputs are superseded and should not be used for final paper writing, tables, or figure selection:

- `quantile_model/dura_pag_informer_quantile_on_pretrain_results/`
- `quantile_model/historical_quantiles_hourly_baseline_results/`
- any earlier risk-aware evaluation outputs generated before the canonical dataset switch
- any earlier hourly coverage or hourly diagnostic outputs generated before the canonical dataset switch

Use these folders instead:

- `canonical_main_results/`
- `canonical_baseline_results/`

Reason:
- the earlier outputs were generated on a different dataset version
- the canonical folders are the only apples-to-apples final experiment set
