# Windows GPU Runbook

## Purpose
This document explains how to hand work off between the local editing machine and the Windows GPU experiment machine.

## Assumptions
- Both machines are Windows machines.
- Experiments should still be launched from bash scripts for reproducibility.
- Git is the primary synchronization mechanism.

## Recommended Machine Roles
- Editing machine: code changes, documentation updates, cleanup, lightweight checks.
- GPU machine: training runs, heavy evaluation, and result generation that depends on CUDA.

## Standard Handoff Flow
1. Commit or stash local work before switching machines.
2. Push the current branch after updating scripts and docs.
3. On the GPU machine, pull the same branch.
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
