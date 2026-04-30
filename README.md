# Probabilistic EV Charging Demand Forecasting

This repository supports a conference submission on risk-aware EV charging demand forecasting.

## Main Idea
The project extends a deterministic GAT-Informer forecasting model with:
- monotonic quantile prediction
- conformal calibration
- risk-aware evaluation

## Collaboration Goal
This repository is actively edited on one Windows machine and used for GPU experiments on another Windows machine.
The priority is to keep the project easy for both humans and Codex to resume without re-discovering context.

## Current Focus
The current goal is not large-scale model redesign.
The focus is to:
- understand the forecasting code path
- reuse existing predictions when possible
- add lightweight evaluation and plots for the paper
- keep experiment setup reproducible across both machines

## Read First
- `docs/project_goal.md`: project context and submission goal
- `docs/current_task.md`: active tasks and immediate priorities
- `docs/experiment_status.md`: what is finished, canonical, and still pending
- `docs/windows_gpu_runbook.md`: how to sync and run work on the experiment machine
- `docs/AGENTS.md`: collaboration rules for AI coding agents

## Main Code Paths
- `train.py`: main forecasting training and evaluation entry point
- `train.sh`: current bash entry for forecasting experiments
- `scripts/inference/`: scripts for rebuilding canonical forecasting outputs
- `scripts/visualization/`: plotting and downstream analysis scripts
- `canonical_main_results/`: canonical main paper-facing outputs
- `canonical_main_results/ablation/`: canonical ablation outputs, including ablation A
- `canonical_baseline_results/`: canonical baseline outputs

## Workflow Notes
- Use git to move context and scripts between the two machines.
- Push documentation and experiment scripts before switching machines.
- Keep canonical results separate from temporary experiment outputs.
- Prefer bash scripts for repeatable experiment execution, even on Windows.
