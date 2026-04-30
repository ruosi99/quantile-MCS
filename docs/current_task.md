# Current Task

## What This File Tracks
This file is the short operational handoff for the next work session.

## Current Situation
- `Lenovo` is the Windows GPU experiment machine and has a working `py12` CUDA environment.
- `Dell` is the original/editing machine and remains the preferred place for broad code editing, repository cleanup, and paper-facing packaging.
- The repository will be used across both machines through git and branch-based handoff.
- This branch now also hosts a separate journal exploration stream based on the forecasting codebase, distinct from the conference paper line.
- On `2026-04-30`, Lenovo passed the journal Stage 0 H=[1] parity gate. See `docs/journal_stage0/h1_parity_experiment_log.md`.

## Immediate Working Priorities
- Keep forecasting work as the main focus for the conference submission.
- Avoid large architecture changes unless explicitly needed.
- Prefer changes that improve reproducibility, evaluation, plotting, or lightweight baselines.
- Convert future experiment execution into explicit bash scripts that can be rerun on Lenovo.
- For the journal branch, define implementation guardrails before changing the multi-horizon forecasting pipeline.
- Commit and push the Stage 0 H=[1] parity fixes and report before beginning Stage 1 multi-horizon implementation.

## For The Next Codex Session
- Read `docs/project_goal.md`, `docs/experiment_status.md`, and `docs/windows_gpu_runbook.md` first.
- Treat `canonical_main_results/` and `canonical_baseline_results/` as the trusted outputs unless the branch explicitly replaces them.
- Treat `canonical_main_results/ablation/` as the canonical home for ablation outputs, including ablation A.
- Use forecasting-related code paths before touching legacy allocation code.
- For journal-only work, also read `docs/journal_multihorizon_context.md`, `docs/journal_multihorizon_guardrails.md`, and `docs/journal_multihorizon_risk_register.md`.

## Expected Near-Term Work
- Commit and push the Stage 0 H=[1] parity code/documentation changes.
- Decide whether the full `journal_results/stage0_h1_parity/` candidate output should be transferred separately or tracked through Git LFS.
- Begin Stage 1 only after the Stage 0 handoff is clean.
- Keep conference-facing and journal-facing outputs clearly separated.
