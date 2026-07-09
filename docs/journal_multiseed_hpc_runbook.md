# Journal Multi-Seed HPC Runbook

This runbook keeps the multi-seed forecasting runs isolated from canonical single-run results.

## Output Layout

All new Stage 1 runs are written under:

```text
journal_results/shenzhen_multihorizon/multiseed/<model_key>/seed_<seed>/
```

Each run writes `run_metadata.json`, `point_metrics_by_horizon.csv`,
`raw_interval_metrics_by_horizon.csv`, and `quantile_crossing_metrics_by_horizon.csv`.

## Seeds

The default seed set is:

```text
2023, 2024, 2025, 2026, 2027
```

## Submit Proposed Model First

```bash
sbatch jobs/train_hg_repatchtst_multiseed.slurm
```

The proposed model job uses:

```text
horizon_gated_repatchtst_p8s4_from_repatch_identity_wd0_lr5e5_epoch300
```

By default it warm-starts from:

```text
journal_results/shenzhen_multihorizon/graph_patchtst_ablate_g2_identity_weightdecay0_lr1e4_epoch200
```

If the checkpoint lives elsewhere on HPC, submit with:

```bash
sbatch --export=ALL,SOURCE_CKPT=/absolute/path/to/checkpoint.pt jobs/train_hg_repatchtst_multiseed.slurm
```

## Submit Baselines

Run independent baselines first:

```bash
sbatch --array=0-34%5 jobs/train_multiseed_baselines.slurm
```

Array mapping:

```text
0-4    pag_informer_warmstart
5-9    lstm_quantile
10-14  tft_quantile
15-19  patchtst_quantile_hidden256_seq48_p8s4_lr2e4_epoch500
20-24  nlinear_quantile
25-29  dlinear_quantile
30-34  multi_scale_temporal_graph_quantile
```

After PatchTST seed checkpoints finish, run GraphPatchTST:

```bash
sbatch --array=35-39%2 jobs/train_multiseed_baselines.slurm
```

GraphPatchTST uses the matching PatchTST seed checkpoint from:

```text
journal_results/shenzhen_multihorizon/multiseed/patchtst_quantile_hidden256_seq48_p8s4_lr2e4_epoch500/seed_<seed>/
```

If your HPC paths differ, override the repo and Python executable:

```bash
sbatch --export=ALL,REPO_ROOT=/path/to/quantile-MCS,PYTHON=/path/to/python jobs/train_multiseed_baselines.slurm
```

## Deterministic Baseline

The historical quantile baseline is deterministic, so it should be reported once rather than across seeds:

```bash
python scripts/inference/historical_quantiles_baseline.py \
  --data_path data/datasets/ST_EVCDP_v2_canonical \
  --out_dir journal_results/shenzhen_multihorizon/historical_quantiles_baseline
```

## Summarize Results

After runs finish:

```bash
python scripts/journal/summarize_multiseed_forecasting.py \
  --root journal_results/shenzhen_multihorizon/multiseed \
  --output-dir journal_results/shenzhen_multihorizon/multiseed_summary
```

Main summary files:

```text
point_metrics_overall_summary.csv
point_metrics_by_horizon_summary.csv
raw_interval_metrics_overall_summary.csv
raw_interval_metrics_by_horizon_summary.csv
quantile_crossing_metrics_by_horizon_summary.csv
multiseed_run_manifest.csv
```
