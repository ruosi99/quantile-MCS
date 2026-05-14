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
- On `2026-05-05`, the Stage 2.5 Diagnostic Gate was implemented, tested, and run on Lenovo. See `docs/journal_stage2_diagnostic_gate.md`.
- Diagnostic Gate result: Stage 2 global CQR's marginal 90 percent coverage gain is mostly a zero-demand boundary-rescue effect, while localized H1 positive-demand hour cells still under-cover.
- Stage 3 should not be started automatically. If pursued, it should be framed as a localized reliability experiment rather than a required fix for marginal 90 percent coverage.
- On `2026-05-05`, Stage 4 decision-first evaluation was implemented, tested, and run on Lenovo. See `docs/journal_stage4_decision_eval.md`.
- Stage 4 result: using cost-ratio-aligned raw target quantiles gives large expected-cost gains over median decisions under asymmetric costs; direct one-sided conformal calibration adds only tiny extra gains.
- The agreed paper direction is now boundary-aware reliability diagnostics plus decision-value attribution, not broad stratified calibration as the default main contribution.
- On `2026-05-06`, Stage 2.75/4.5 paper evidence assets were exported on Lenovo. See `docs/journal_paper_assets.md`.
- Paper-assets result: zero-boundary rescue explains nearly all Stage 2 coverage gain, while bootstrap decision attribution confirms that cost-aligned quantile choice accounts for almost all decision-value gain.
- On `2026-05-06`, the reinforcement package was implemented, tested, and run on Lenovo. See `docs/journal_reinforcement_experiments.md`.
- Reinforcement result: station-level sparsity explains where zero-boundary rescue matters; deployment strategy comparison confirms that cost-aligned quantile decisions dominate symmetric interval-upper-bound deployment except when the cost ratio already matches the upper quantile.
- On `2026-05-14`, the CRC risk-control deployment experiment was implemented, tested, and run on Lenovo. See `docs/journal_crc_risk_control_experiment_results.md`.
- CRC result: violation-risk CRC provides a credible service-risk deployment layer and reduces overage versus `symmetric_90_upper`; normalized-shortage CRC controls normalized shortage risk but can allow high violation rates, so it should not be interpreted as service-violation control.

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
- Use the completed diagnostic gate before committing to Stage 3 as a main contribution.
- Prioritize decision-oriented evaluation before treating broad stratified calibration as central.
- Treat the completed Stage 4 decision result as the next major decision checkpoint before any Stage 3 work.
- Use `docs/journal_paper_direction.md` and `docs/journal_stage4_decision_eval.md` as the narrative and execution references for the next journal step.
- Use `docs/journal_paper_assets.md` and `journal_results/shenzhen_multihorizon/paper_assets/` as the current paper-facing evidence package.
- The next journal reinforcement package should emphasize station-level zero-inflation sensitivity, deployment strategy comparison, and a paper-facing deployment diagnostic flowchart rather than reopening broad Stage 3.
- Use `docs/journal_reinforcement_experiments.md` and `journal_results/shenzhen_multihorizon/reinforcement/` as the current robustness and deployment-evidence package.
- Use `docs/journal_crc_risk_control_experiment_results.md` and `journal_results/shenzhen_multihorizon/crc_risk_control/` as the current service-risk deployment comparison package.

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
- Diagnostic analysis on raw/global/horizon-wise outputs has been completed for the Stage 1B warm-start result.
- The next concrete journal task is to review the Stage 4 decision outputs and decide the paper direction.
- The next concrete journal tasks are:
  - review and polish the paper-facing tables and figures in `journal_results/shenzhen_multihorizon/paper_assets/`
  - add station-level zero-inflation sensitivity to test whether ZBR dominance and quantile-choice dominance persist across different station sparsity regimes
  - add deployment strategy comparison so the decision-value result becomes a direct practitioner-facing cost comparison
  - optionally add day-level or station-day bootstrap if date indices are exported
  - package a deployment diagnostic flowchart once the above reinforcement results are stable
  - the above reinforcement tasks have now been completed at station-level and day-bootstrap level; remaining work is paper-facing polish
  - only then decide whether to run a narrowly scoped Stage 3 localized reliability experiment for H1 positive-demand hard cells
- Keep conference-facing and journal-facing outputs clearly separated.
