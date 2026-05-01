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
- Status: direct multi-horizon raw path implemented and smoke-tested on Lenovo.
- Detailed log: `docs/journal_stage1_multihorizon_raw.md`
- New launcher: `scripts/journal/run_stage1_multihorizon_raw.sh`
- New Python entry: `scripts/journal/train_multihorizon_raw.py`
- Smoke output shape: `predict_quantiles = (2, 1682, 5, 13)`, `label_list = (2, 1682, 5)`.
- Full Stage 1 training is pending and should write to `journal_results/shenzhen_multihorizon/raw/`.

## Latest Stage 1B Warm-Start Status
- Version: `stage1B-warmstart-scaffold-2026-05-01`
- Date: `2026-05-01`
- Branch: `multi_horizon_journal`
- Status: warm-start path implemented; full Lenovo run pending.
- Detailed log: `docs/journal_stage1B_warmstart.md`
- New launcher: `scripts/journal/run_stage1_multihorizon_warmstart.sh`
- Warm-start source checkpoint: `quantile_model/dura_pag_informer_quantile_on_pretrain_results/dura_pag_informer_quantile_on_pretrain_1_bs8_completed.pt`
- Planned output directory: `journal_results/shenzhen_multihorizon/warmstart_raw/`
- Primary comparison target: Stage 1A from-scratch raw output in `journal_results/shenzhen_multihorizon/raw/`.

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
- The direct multi-horizon Stage 1 scaffold is available; full raw training should be reviewed before Stage 2 calibration begins.
