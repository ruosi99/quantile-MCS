# Current Task

## What This File Tracks
This file is the short operational handoff for the next work session.

## Current Situation
- This machine is for code editing, repository cleanup, and experiment preparation.
- The other Windows machine is for GPU-dependent experiment runs.
- The repository will be used across both machines through git and branch-based handoff.
- This branch now also hosts a separate journal exploration stream based on the forecasting codebase, distinct from the conference paper line.

## Immediate Working Priorities
- Keep forecasting work as the main focus for the conference submission.
- Avoid large architecture changes unless explicitly needed.
- Prefer changes that improve reproducibility, evaluation, plotting, or lightweight baselines.
- Convert future experiment execution into explicit bash scripts that can be rerun on the GPU machine.
- For the journal branch, define implementation guardrails before changing the multi-horizon forecasting pipeline.

## For The Next Codex Session
- Read `docs/project_goal.md`, `docs/experiment_status.md`, and `docs/windows_gpu_runbook.md` first.
- Treat `canonical_main_results/` and `canonical_baseline_results/` as the trusted outputs unless the branch explicitly replaces them.
- Treat `canonical_main_results/ablation/` as the canonical home for ablation outputs, including ablation A.
- Use forecasting-related code paths before touching legacy allocation code.
- For journal-only work, also read `docs/journal_multihorizon_context.md`, `docs/journal_multihorizon_guardrails.md`, and `docs/journal_multihorizon_risk_register.md`.

## Expected Near-Term Work
- Create new experiment branches as needed.
- Add or refine experiment bash scripts.
- Run experiments on the GPU machine.
- Sync back the useful outputs and keep the repository organized.
- Keep conference-facing and journal-facing outputs clearly separated.
