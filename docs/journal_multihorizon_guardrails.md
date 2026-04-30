# Journal Branch Guardrails

## Purpose
This document defines what this journal branch is allowed to change, what it should avoid, and how to keep the work auditable.

## Allowed Change Zones
- `train.py`
- `train.sh` or new journal-specific bash launchers under `scripts/` or repo root
- `utils/model_training/conformal.py`
- `utils/model_training/training_utils.py`
- `utils/model_training/loss_functions.py` if multi-horizon handling is required
- `utils/model_training/models.py` for multi-horizon quantile output support
- new evaluation scripts under `scripts/inference/` and `scripts/visualization/`
- new journal-specific documentation under `docs/`

## Preferred New Artifacts
- new journal-specific run scripts instead of overloading the conference launcher
- new evaluation scripts instead of mutating conference plotting code too early
- new result folders whose names clearly indicate journal and multi-horizon scope

## Avoid If Possible
- touching legacy allocation code
- changing existing conference canonical outputs
- large backbone redesigns unrelated to multi-horizon support
- mixing journal outputs into `canonical_main_results/` unless they are explicitly declared canonical for the journal line later
- irreversible refactors before single-step parity is preserved

## Boundary Rules
1. Single-step conference reproducibility must remain intact.
2. Multi-horizon support should be added as a new path, not as a silent behavior change.
3. Calibration wrappers must support explicit fallback behavior when strata are too small.
4. Every saved result should encode:
   - branch stream
   - horizon set
   - calibration mode
   - dataset or city
5. Every experiment script should be rerunnable from the repository root.

## Recommended Result Layout
- `journal_results/`
- `journal_results/shenzhen_multihorizon/`
- `journal_results/shenzhen_multihorizon/global_cqr/`
- `journal_results/shenzhen_multihorizon/horizon_wise_cqr/`
- `journal_results/shenzhen_multihorizon/stratified_conformal/`
- `journal_results/transfer_calibration/`

The exact names may evolve, but the journal branch should keep a clearly separate top-level result namespace.

## Required Regression Checks
Before trusting journal changes, keep at least these sanity checks:
1. Existing single-step inference still runs.
2. The quantile ordering constraint still holds for the journal multi-horizon head.
3. The calibration code can reproduce a global CQR baseline.
4. Saved tensor shapes are explicitly documented for each new stage.

## Minimal Shape Contract For Journal Work
- input history: `(B, N, seq_len)`
- multi-horizon point target: `(B, N, H)`
- multi-horizon quantile output: `(B, N, H, Q)` or another single documented convention
- calibration scores:
  - global or horizon-wise: scalar or `(H,)`
  - stratified: keyed by horizon and stratum

No new implementation should proceed without documenting the chosen tensor convention.

## Theorem / Claim Guardrail
The planned finite-sample coverage statement should be presented carefully:
- it is a standard split-conformal style guarantee inside each stratum
- it requires within-stratum exchangeability
- it does not by itself guarantee conditional coverage
- it becomes fragile when strata are sparse or chosen post hoc

This claim is useful, but it is not enough to carry the paper's novelty on its own.
