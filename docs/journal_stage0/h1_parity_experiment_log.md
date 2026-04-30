# Stage 0 H=[1] Parity Experiment Log

## Record Metadata
- Record version: `stage0-h1-parity-gpu-2026-04-30`
- Record date: `2026-04-30 13:46:42 +04:00`
- Time zone: `Asia/Dubai`
- Machine: `Lenovo`
- Machine role: Windows GPU experiment machine
- Git branch: `multi_horizon_journal`
- Base commit before local fixes: `663a85a`
- Experiment stage: Journal Stage 0, H=[1] parity gate

## Purpose
This record documents the first successful Stage 0 H=[1] parity run on Lenovo.

The goal was to verify that the journal branch can reproduce the existing single-step
conference forecasting path before any direct multi-horizon implementation begins.

## Environment
- Conda environment: `py12`
- Python: `3.12.13`
- PyTorch: `2.6.0+cu124`
- CUDA reported by PyTorch: `12.4`
- GPU: `NVIDIA GeForce RTX 3060 Laptop GPU`
- Key graph packages:
  - `torch-geometric==2.7.0`
  - `torch-scatter==2.1.2+pt26cu124`
  - `torch-sparse==0.6.18+pt26cu124`
  - `torch-cluster==1.6.3+pt26cu124`
  - `torch-spline-conv==1.2.2+pt26cu124`

## Inputs
- Dataset: `data/datasets/ST_EVCDP_v2_canonical/`
- Model name: `dura_pag_informer_quantile_on_pretrain`
- Legacy checkpoint source:
  - `quantile_model/dura_pag_informer_quantile_on_pretrain_results/dura_pag_informer_quantile_on_pretrain_1_bs8_completed.pt`
- Runtime checkpoint location expected by `train.py`:
  - `data/results/ST_EVCDP_v2_canonical/dura_pag_informer_quantile_on_pretrain_results/dura_pag_informer_quantile_on_pretrain_1_bs8_completed.pt`
- Reference output directory:
  - `canonical_main_results/`
- Candidate output directory:
  - `journal_results/stage0_h1_parity/`
- Parity report:
  - `docs/journal_stage0/h1_parity_report.json`

## Run Command
```bash
bash scripts/journal/run_journal_h1_parity.sh
```

After the fixes recorded below, the script:
1. regenerates the Stage 0 run specs,
2. copies the legacy checkpoint into the runtime result directory if needed,
3. runs the H=[1] inference/evaluation path,
4. copies candidate outputs into `journal_results/stage0_h1_parity/`,
5. writes `docs/journal_stage0/h1_parity_report.json`.

## Code Changes Made For This Run
- `scripts/journal/run_journal_h1_parity.sh`
  - Added checkpoint bootstrap from `quantile_model/...` to `data/results/...`.
  - Changed H=[1] parity quantiles to the legacy 7-quantile grid.
  - Added automatic candidate result copy to `journal_results/stage0_h1_parity/`.
  - Added automatic parity report generation.
- `scripts/journal/stage0_runtime.py`
  - Added `LEGACY_H1_QUANTILES`.
  - Kept future journal multi-horizon planning on the 13-quantile grid.
  - Updated parity target metadata to distinguish reference and candidate point metric CSV names.
- `scripts/journal/check_h1_parity.py`
  - Added separate `--reference-point-metrics-file` and `--candidate-point-metrics-file` arguments.
  - This allows comparison between `canonical_point_q50.csv` and the runtime training-output metric filename.
- `train.py`
  - Changed checkpoint loading to `torch.load(..., weights_only=False)`.
  - Reason: PyTorch 2.6 defaults `weights_only=True`, but this repository stores full model objects in `.pt` checkpoints.
- `utils/model_training/training_utils.py`
  - Changed console metric label from `R²` to `R2`.
  - Reason: the Windows console raised a GBK encoding error when printing the superscript character.
- `docs/journal_stage0/stage0_runbook.md`
  - Updated the runbook so the H=[1] script is the single operational entry point.
- `docs/journal_stage0/journal_h1_run_spec.json`
  - Regenerated with the legacy 7-quantile H=[1] parity configuration.

## Why These Changes Were Needed
The initial H=[1] run exposed Lenovo migration issues rather than modeling issues:

1. `train.py` expected the checkpoint under `data/results/...`, while the synced legacy checkpoint was under `quantile_model/...`.
2. The existing checkpoint and canonical outputs use 7 quantiles, but the initial journal H=[1] spec used the future 13-quantile grid.
3. PyTorch 2.6 changed the default behavior of `torch.load`, so old full-object checkpoints need `weights_only=False`.
4. The Windows console failed on a non-ASCII metric label.
5. The canonical point metric file and newly generated runtime point metric file use different names.

## Results
The successful parity report is stored in:

```text
docs/journal_stage0/h1_parity_report.json
```

Main parity summary:

| Item | Result |
| --- | --- |
| `predict_quantiles` shape | `(852, 1682, 7)` |
| `predict_quantiles` MAE | `6.999567535129148e-08` |
| `predict_quantiles` max abs error | `7.510185241699219e-05` |
| `predict_quantiles` RMSE | `4.2009105587758263e-07` |
| `label_list` shape | `(852, 1682)` |
| `label_list` MAE | `0.0` |
| `label_list` max abs error | `0.0` |

Point metric parity summary:

| Metric | Reference | Candidate | Abs diff |
| --- | ---: | ---: | ---: |
| MSE | `0.1022227760141641` | `0.1022227778378138` | `1.8236497034695986e-09` |
| RMSE | `0.3197229676050255` | `0.3197229704569471` | `2.8519215877764736e-09` |
| MAPE | `5.44109338942325` | `5.441093372570927` | `1.6852322204385928e-08` |
| RAE | `0.0216437433729183` | `0.0216437433347159` | `3.8202399577080826e-11` |
| MAE | `0.0649845622951791` | `0.0649845621804777` | `1.1470140115488192e-10` |
| R2 | `0.9965995109431128` | `0.9965995108824482` | `6.066458446696288e-11` |
| MedAE | `0.0054545179009437` | `0.0054545179009437` | `0.0` |
| EVS | `0.9966008659406674` | `0.9966008658800874` | `6.057998547248644e-11` |

Conclusion:

The H=[1] journal parity gate passes. The differences are at floating-point tolerance level,
and labels match exactly.

## Verification Commands
```bash
conda run -n py12 python -m pytest tests/test_stage0_runtime.py tests/test_journal_contracts.py -q
```

Result:

```text
10 passed
```

```bash
conda run -n py12 python scripts/journal/check_h1_parity.py \
  --reference-dir canonical_main_results \
  --candidate-dir journal_results/stage0_h1_parity \
  --reference-point-metrics-file canonical_point_q50.csv \
  --candidate-point-metrics-file dura_pag_informer_quantile_on_pretrain_1bs8_point_q50.csv \
  --report-file docs/journal_stage0/h1_parity_report.json
```

Result:

```text
docs/journal_stage0/h1_parity_report.json generated successfully
```

## Sync Guidance
Commit through Git:
- code and run-script fixes,
- Stage 0 documentation,
- `docs/journal_stage0/h1_parity_report.json`,
- lightweight CSV/Markdown/JSON metadata.

Do not commit through normal Git unless Git LFS policy is expanded:
- `.pt` checkpoints,
- large `.npy` arrays,
- bulky generated result folders.

For this run, the most important result artifact to commit is:

```text
docs/journal_stage0/h1_parity_report.json
```

The full candidate output directory can be transferred separately or moved to Git LFS:

```text
journal_results/stage0_h1_parity/
```

Current `.gitignore` ignores `journal_results/` to avoid accidental large-array commits.
Current `.gitattributes` only routes `*.csv` through Git LFS. It does not route `.npy`, `.pt`,
or image outputs through LFS.

## Next Step
Before Stage 1 training starts, commit and push the code/documentation fixes from this Stage 0 run.
Then implement the direct multi-horizon path behind explicit journal-specific entry points.
