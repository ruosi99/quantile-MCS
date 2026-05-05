# Current Task

## What This File Tracks
This file is the short operational handoff for the next work session.

## Current Situation
- `Lenovo` is the Windows GPU experiment machine and has a working `py12` CUDA environment.
- `Dell` is the original/editing machine and remains the preferred place for broad code editing, repository cleanup, and paper-facing packaging.
- The repository will be used across both machines through git and branch-based handoff.
- This branch now also hosts a separate journal exploration stream based on the forecasting codebase, distinct from the conference paper line.
- On `2026-04-30`, Lenovo passed the journal Stage 0 H=[1] parity gate. See `docs/journal_stage0/h1_parity_experiment_log.md`.
- On `2026-04-30`, the Stage 1 direct multi-horizon raw path was implemented and smoke-tested on Lenovo; the full Stage 1A run was completed on `2026-05-01`. See `docs/journal_stage1_multihorizon_raw.md`.
- On `2026-05-01`, the Stage 1B warm-start path was added and the full warm-start run was completed. See `docs/journal_stage1B_warmstart.md`.
- On `2026-05-01`, the Stage 2 global and horizon-wise CQR post-processing path was implemented and smoke-tested. See `docs/journal_stage2_cqr.md`.
- On `2026-05-01`, the full Stage 2 CQR run was completed for the Stage 1B warm-start model. Global CQR corrected 90 percent mean PICP from `0.638917` to `0.913388` with almost unchanged MPIW.

## Immediate Working Priorities
- Keep forecasting work as the main focus for the conference submission.
- Avoid large architecture changes unless explicitly needed.
- Prefer changes that improve reproducibility, evaluation, plotting, or lightweight baselines.
- Convert future experiment execution into explicit bash scripts that can be rerun on Lenovo.
- For the journal branch, define implementation guardrails before changing the multi-horizon forecasting pipeline.
- Commit and push the Stage 0 H=[1] parity fixes and report before beginning Stage 1 multi-horizon implementation.
- Review and commit the Stage 1 raw multi-horizon scaffold before launching the full Lenovo run.
- Run or review Stage 1B warm-start after preserving the Stage 1A raw result.
- Treat Stage 1B warm-start plus Stage 2 global CQR as the current representative multi-horizon calibration result.
- Reframe adaptive/stratified calibration as a localized reliability or worst-cell ACE experiment, not as a required fix for marginal 90 percent coverage.

## For The Next Codex Session
- Read `docs/project_goal.md`, `docs/experiment_status.md`, and `docs/windows_gpu_runbook.md` first.
- Treat `canonical_main_results/` and `canonical_baseline_results/` as the trusted outputs unless the branch explicitly replaces them.
- Treat `canonical_main_results/ablation/` as the canonical home for ablation outputs, including ablation A.
- Use forecasting-related code paths before touching legacy allocation code.
- For journal-only work, also read `docs/journal_multihorizon_context.md`, `docs/journal_multihorizon_guardrails.md`, and `docs/journal_multihorizon_risk_register.md`.

## Expected Near-Term Work
- Commit and push the Stage 0 H=[1] parity code/documentation changes.
- Decide whether the full `journal_results/stage0_h1_parity/` candidate output should be transferred separately or tracked through Git LFS.
- Run the full Stage 1 raw multi-horizon baseline on Lenovo with `bash scripts/journal/run_stage1_multihorizon_raw.sh`.
- Run the Stage 1B warm-start baseline on Lenovo with `bash scripts/journal/run_stage1_multihorizon_warmstart.sh`.
- Stage 2 warm-start CQR has been run with `bash scripts/journal/run_stage2_cqr_warmstart.sh`.
- `bash scripts/journal/run_stage2_cqr_raw.sh` remains optional because Stage 1A versus Stage 1B is unlikely to become a paper-facing ablation.
- Keep conference-facing and journal-facing outputs clearly separated.
