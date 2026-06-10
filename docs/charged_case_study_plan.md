# CHARGED Case Study Plan

This case study is an external generalization check. It does not replace the main UrbanEV station-level experiments.

## Scope

- Main experiments remain on `data/datasets/ST_EVCDP_v2_canonical/` with station-level UrbanEV data.
- CHARGED is used only through `--dataset-family charged_site`.
- Case-study outputs are isolated under `journal_results/charged_case_study/`.
- The first two cities are `LOA_remove_zero` and `MEL_remove_zero`.

## Data Choice

- Use `volume` as the CHARGED target because the CHARGED paper reports demand primarily as charging volume.
- Normalize the target by `charger_num` during training, then de-normalize by the same `charger_num` for metrics and decision evaluation.
- Use `_remove_zero` city folders for the official case study. This removes inactive sites but does not remove all zero-demand hours.
- CHARGED distance matrices are physical distance, so the loader converts them to a kNN Gaussian similarity adjacency before GraphPatchTST receives them.

## Reproducible Audit

Run:

```bash
python scripts/journal/audit_charged_case_study.py \
  --charged-root CHARGED/data \
  --cities AMS,JHB,LOA,MEL,SPO,SZH \
  --variants remove_zero,full \
  --output-dir journal_results/charged_case_study/audit
```

Outputs:

- `charged_city_manifest.csv`
- `charged_data_issues.csv`
- `audit_metadata.json`

## Training Plan

1. Train city-specific PatchTST source checkpoints:

```bash
sbatch jobs/train_charged_patchtst_source_case_study.slurm
```

2. Train the selected GraphPatchTST configuration with same-city warm start:

```bash
sbatch jobs/train_charged_graph_patchtst_case_study.slurm
```

The GraphPatchTST run mirrors the selected main configuration:

- `seq_len=48`
- `patch_len=8`
- `patch_stride=4`
- `hidden_dim=256`
- `transformer_layers=3`
- `attention_heads=4`
- `graph_layers=2`
- `learning_rate=1e-4`
- `weight_decay=0`
- `epochs=200`
- horizons `1,3,6,12,24`
- quantiles `0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.833333333333,0.9,0.95`

## Post-Training Experiments

After a city Stage 1 run finishes:

```bash
bash scripts/journal/run_charged_case_post_training.sh LOA
bash scripts/journal/run_charged_case_post_training.sh MEL
```

This runs:

- Stage 2 CQR calibration
- Stage 4 asymmetric decision evaluation

These stages reuse `run_metadata.json`, so they preserve the CHARGED `volume` target and site-level data contract automatically.

## Reporting

Report the case study as a compact external validation section:

- Show LOA and MEL point/interval metrics by horizon.
- Show CQR coverage correction compared with raw intervals.
- Show decision regret under the same cost ratios as the main experiment.
- Avoid mixing CHARGED metrics into the main UrbanEV benchmark table; keep them in a separate case-study table.
