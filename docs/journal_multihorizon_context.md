# Journal Branch Context: Spatio-Temporal Stratified Conformal Multi-Horizon Forecasting

## Purpose
This document defines the research context for the journal-only branch built on top of the current conference forecasting codebase.

This branch is not a continuation of the conference paper narrative. It is a separate journal submission track that reuses the existing forecasting implementation as the starting point.

## Working Title
Spatio-Temporal Stratified Conformal Multi-Horizon Forecasting for City-Scale EV Charging Demand

## Target Outlet
- Target journal: Applied Energy
- Framing: city-scale uncertainty quantification and decision support for urban charging operations

## Core Research Claim
The branch should test whether the existing monotonic-quantile PAG/GAT-Informer pipeline can be extended from single-step forecasting to direct multi-horizon forecasting, and whether calibration quality can be improved by replacing a single global conformal threshold with spatio-temporal stratified calibration and a decision-oriented one-sided conformal layer.

## Proposed Technical Story
1. Keep the current forecasting backbone as intact as possible.
2. Extend the output path from single-step prediction to direct multi-horizon prediction for `H = {1, 3, 6, 12, 24}`.
3. Compare three calibration regimes:
   - global CQR
   - horizon-wise CQR
   - spatio-temporal stratified conformal calibration
4. Add a one-sided decision calibration layer for the critical fractile `tau_star = c_u / (c_u + c_o)`.
5. Evaluate both forecasting quality and downstream decision quality.

## Why This Branch Exists
The current conference code and results have two clear limitations that are acceptable for the conference paper but weak for a stronger journal submission:
- single-step forecasting only
- global calibration can hide systematic under-coverage at certain hours or subsets of stations

This branch explores whether those weaknesses can be turned into a stronger journal contribution without redesigning the full backbone.

## Non-Negotiable Separation From The Conference Line
- The conference paper remains a separate paper.
- Existing conference-facing canonical outputs remain valid and must not be overwritten or reinterpreted as journal outputs.
- New journal experiments must be stored under clearly journal-specific result directories.
- Any code changes for this branch must preserve the ability to reproduce the existing single-step conference pipeline.

## Current Code Reality
The current implementation is still fundamentally single-step:
- `train.py` hard-codes `pred_len = 1`
- `utils/model_training/conformal.py` assumes single-step tensors shaped like `(B, N, Q)` or `(T, N, Q)`
- `utils/model_training/training_utils.py` creates labels as single-step targets
- the current evaluation and plotting scripts are built around one-step outputs and interval tables

Because of this, the journal idea is a pipeline extension project, not just a calibration add-on.

## Immediate Branch Deliverables
1. Define clear code and experiment boundaries before implementation.
2. Preserve a stable single-step baseline path for regression checks.
3. Create a multi-horizon-capable data, training, calibration, and evaluation path.
4. Record all assumptions that matter for publication claims.

## Publication Positioning Constraint
The likely novelty is not "stratified conformal" by itself. The stronger publishable angle is the combination of:
- city-scale EV charging station network forecasting
- direct multi-horizon probabilistic forecasting
- spatio-temporal calibration diagnostics
- decision-oriented one-sided conformal calibration

Any future writing should avoid overselling the stratification piece as a fundamentally new conformal method.
