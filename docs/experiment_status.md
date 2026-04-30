# Experiment Status

## Objective
Keep a lightweight handoff record so the GPU experiment machine and Codex can resume work quickly.

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
- This machine does not have a usable GPU for training experiments.
- The other Windows machine is the main experiment runner.
- New experiment coordination will happen by switching branches and syncing through git.

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
4. Run the experiment on the GPU machine.
5. Commit the script, metadata, and only the outputs worth preserving.

## Journal Exploration Notes
- The journal line is a separate paper track built from the current forecasting codebase.
- The journal line currently targets multi-horizon forecasting plus stratified conformal calibration.
- Conference canonical outputs should remain unchanged unless explicitly regenerated for the conference story.
- Journal-specific outputs should live in journal-specific directories rather than `canonical_main_results/`.
