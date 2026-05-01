# Windows GPU Runbook

## Purpose
This document explains how to hand work off between Dell and Lenovo.

## Machine Names
- `Dell`: original/editing machine for broad code editing, repository cleanup, and paper-facing packaging.
- `Lenovo`: Windows GPU experiment machine for CUDA runs, heavy evaluation, and generated experiment outputs.

## Assumptions
- Both machines are Windows machines.
- Experiments should still be launched from bash scripts for reproducibility.
- Git is the primary synchronization mechanism.

## Recommended Machine Roles
- Dell: code changes, documentation updates, cleanup, lightweight checks.
- Lenovo: training runs, heavy evaluation, and result generation that depends on CUDA.

## Standard Handoff Flow
1. Commit or stash local work before switching machines.
2. Push the current branch after updating scripts and docs.
3. On Lenovo, pull the same branch.
4. Activate the experiment environment.
5. Run the intended bash script from the repository root.
6. Commit back the script changes, useful metadata, and only the outputs worth preserving.

## What To Sync Before Running
- The target branch.
- Updated docs if the experiment purpose changed.
- The exact bash script used to launch the run.
- Any new plotting or evaluation scripts needed after training.

## What To Record After Running
- The branch name used for the run.
- The exact script used.
- Output directory names.
- Any important notes about runtime, CUDA behavior, or failed attempts.

## Current Stage 0 Handoff Record
- Version: `stage0-h1-parity-gpu-2026-04-30`
- Date: `2026-04-30`
- Branch: `multi_horizon_journal`
- Detailed experiment log: `docs/journal_stage0/h1_parity_experiment_log.md`
- Parity report: `docs/journal_stage0/h1_parity_report.json`
- Candidate output directory on Lenovo: `journal_results/stage0_h1_parity/`
- Status: H=[1] journal parity passed; labels match exactly and prediction differences are floating-point scale.

## Current Stage 1 Handoff Record
- Version: `stage1-multihorizon-raw-scaffold-2026-04-30`
- Date: `2026-04-30`
- Detailed implementation log: `docs/journal_stage1_multihorizon_raw.md`
- Lenovo launcher: `scripts/journal/run_stage1_multihorizon_raw.sh`
- Python entry: `scripts/journal/train_multihorizon_raw.py`
- Smoke status: passed on Lenovo with `predict_quantiles = (2, 1682, 5, 13)`.
- Full run command:
```bash
bash scripts/journal/run_stage1_multihorizon_raw.sh
```
- Full run output directory:
```text
journal_results/shenzhen_multihorizon/raw/
```

## Sync Guidance For Stage 0
Use normal Git for:
- code changes under `train.py`, `utils/`, and `scripts/`
- documentation under `docs/`
- lightweight JSON/Markdown reports, especially `docs/journal_stage0/h1_parity_report.json`

Avoid normal Git for large experiment artifacts unless Git LFS rules are expanded:
- `.pt` checkpoints
- large `.npy` arrays
- full generated result folders such as `journal_results/stage0_h1_parity/`

Current `.gitignore` ignores `journal_results/` to avoid accidental large result commits.
Current `.gitattributes` only sends `*.csv` to Git LFS. It does not cover `.npy`, `.pt`, or image files.
Therefore, the recommended Stage 0 transfer is:
1. commit and push code, docs, and `h1_parity_report.json`
2. transfer `journal_results/stage0_h1_parity/` separately if Dell needs the full arrays
3. add explicit Git LFS rules before tracking large future journal result arrays through Git

## Result Management Guidance
- Keep canonical, paper-facing outputs in clearly named tracked directories.
- Keep canonical ablation outputs under `canonical_main_results/ablation/` rather than separate unpacked transfer folders.
- Avoid using ad hoc zip archives as the main collaboration format.
- If a zip is needed for backup, also record where its contents belong in the repository structure.

## Bash On Windows
- Prefer running repository scripts from Git Bash or another bash-compatible shell on Windows.
- Keep `.sh` files in LF format so they behave consistently.
- Avoid machine-specific absolute paths inside scripts.

## Git Notes
- Push before moving to the other machine when you want Codex there to have the latest context.
- Use a dedicated branch for each experiment stream when the work may diverge.
- Avoid storing temporary logs and checkpoints in tracked canonical result folders.
