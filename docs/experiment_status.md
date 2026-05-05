# Experiment Status

## Objective
Keep a lightweight handoff record so Lenovo, Dell, and Codex can resume work quickly.

## Canonical Outputs
- `canonical_main_results/` contains the main paper-facing forecasting outputs.
- `canonical_main_results/ablation/` contains canonical ablation outputs.
- `canonical_main_results/ablation/dura_pag_informer_quantile_independent_on_pretrain_results/` is the current home of the ablation A result set.
- `canonical_baseline_results/` contains the baseline outputs used for comparison.
- `docs/final_canonical_results_package/` contains packaged tables intended for reporting.

## Result Organization Notes
- The former unpacked `dura_pag_informer__results/` content has been reorganized into `canonical_main_results/ablation/`.
- The ablation A result set should now be treated as part of the canonical tracked structure instead of a temporary unpacked folder.
- Future experiment exchange should prefer committed scripts, committed metadata, and tracked result folders instead of manual zip drops.

## Current Practical Constraints
- Lenovo is configured with the `py12` CUDA environment for experiment runs.
- Dell remains the preferred place for broad code editing and repository cleanup.
- New experiment coordination will happen by switching branches and syncing through git.

## Latest GPU-Machine Run
- Version: `stage0-h1-parity-gpu-2026-04-30`
- Date: `2026-04-30`
- Branch: `multi_horizon_journal`
- Base commit before local fixes: `663a85a`
- Status: Stage 0 H=[1] journal parity gate passed.
- Detailed log: `docs/journal_stage0/h1_parity_experiment_log.md`
- Parity report: `docs/journal_stage0/h1_parity_report.json`
- Candidate output directory on Lenovo: `journal_results/stage0_h1_parity/`
- Main result: `predict_quantiles` matched the canonical reference with MAE `6.999567535129148e-08`; labels matched exactly.

## Latest Stage 1 Implementation Status
- Version: `stage1-multihorizon-raw-scaffold-2026-04-30`
- Date: `2026-04-30`
- Branch: `multi_horizon_journal`
- Status: direct multi-horizon raw path implemented, smoke-tested, and full Lenovo run completed.
- Detailed log: `docs/journal_stage1_multihorizon_raw.md`
- New launcher: `scripts/journal/run_stage1_multihorizon_raw.sh`
- New Python entry: `scripts/journal/train_multihorizon_raw.py`
- Smoke output shape: `predict_quantiles = (2, 1682, 5, 13)`, `label_list = (2, 1682, 5)`.
- Full Stage 1 output directory: `journal_results/shenzhen_multihorizon/raw/`.

## Latest Stage 1B Warm-Start Status
- Version: `stage1B-warmstart-scaffold-2026-05-01`
- Date: `2026-05-01`
- Branch: `multi_horizon_journal`
- Status: warm-start path implemented and full Lenovo run completed.
- Detailed log: `docs/journal_stage1B_warmstart.md`
- New launcher: `scripts/journal/run_stage1_multihorizon_warmstart.sh`
- Warm-start source checkpoint: `quantile_model/dura_pag_informer_quantile_on_pretrain_results/dura_pag_informer_quantile_on_pretrain_1_bs8_completed.pt`
- Output directory: `journal_results/shenzhen_multihorizon/warmstart_raw/`
- Primary comparison target: Stage 1A from-scratch raw output in `journal_results/shenzhen_multihorizon/raw/`.

## Latest Stage 2 CQR Status
- Version: `stage2-cqr-warmstart-full-2026-05-01`
- Date: `2026-05-01`
- Branch: `multi_horizon_journal`
- Status: global and horizon-wise CQR post-processing implemented; smoke test passed; full Lenovo warm-start run completed.
- Detailed log: `docs/journal_stage2_cqr.md`
- New Python entry: `scripts/journal/calibrate_multihorizon_cqr.py`
- New launchers:
  - `scripts/journal/run_stage2_cqr_raw.sh`
  - `scripts/journal/run_stage2_cqr_warmstart.sh`
- Completed output directory:
  - `journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw/`
- Optional output directory not run yet:
  - `journal_results/shenzhen_multihorizon/stage2_cqr/raw/`
- Smoke output directory: `journal_results/shenzhen_multihorizon/stage2_cqr/smoke_raw/`
- Verification: `tests/test_journal_stage2_cqr.py` and `tests/test_journal_stage1_multihorizon.py` passed.
- Main Stage 2 result: warm-start global CQR improves mean 90 percent PICP from `0.638917` raw to `0.913388`, with mean MPIW changing only from `2.916468` to `2.916553`.
- Important interpretation: most of the coverage gain comes from correcting zero-demand lower-bound misses near the nonnegative boundary, not from large interval inflation.
- Adaptive calibration status: no longer needed to fix marginal 90 percent coverage, but still potentially useful as a localized reliability or worst-cell ACE diagnostic.

## Immediate Priorities
- Keep repository context accurate enough for Codex to resume work on either machine.
- Standardize bash-based experiment entry points.
- Preserve canonical outputs while keeping temporary experiment clutter out of the main story.
- Make each new experiment easy to identify by branch, script, and output directory.
- Keep the journal multi-horizon exploration isolated from the conference canonical result folders.

## Next Experiment Workflow
1. Create or switch to a dedicated experiment branch.
2. Update docs before leaving one machine if the experiment goal changed.
3. Add or update the bash script that defines the experiment.
4. Run the experiment on Lenovo.
5. Commit the script, metadata, and only the outputs worth preserving.

## Journal Exploration Notes
- The journal line is a separate paper track built from the current forecasting codebase.
- The journal line currently targets multi-horizon forecasting plus stratified conformal calibration.
- Conference canonical outputs should remain unchanged unless explicitly regenerated for the conference story.
- Journal-specific outputs should live in journal-specific directories rather than `canonical_main_results/`.
- The direct multi-horizon Stage 1A and Stage 1B runs are available; Stage 2 warm-start calibration is complete.
- The next journal decision is whether adaptive/stratified calibration improves localized worst-cell reliability enough to remain a central paper contribution.
