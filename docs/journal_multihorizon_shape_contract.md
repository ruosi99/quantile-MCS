# Journal Multi-Horizon Shape Contract

## Purpose
This document freezes the tensor contract for the journal-only multi-horizon pipeline before major implementation begins.

## Primary Convention
- multi-horizon quantile prediction: `(B, N, H, Q)`
- multi-horizon target tensor: `(B, N, H)`
- point prediction derived from a chosen quantile: `(B, N, H)`

## Axis Semantics
- `B`: batch size
- `N`: station count
- `H`: number of forecast horizons
- `Q`: number of quantiles

## Required Journal Assertions
1. All training, inference, calibration, and evaluation code must agree on the horizon axis position.
2. `H=[1]` must use the same code path as the multi-horizon implementation, not a hidden special case.
3. Quantiles must remain monotonic on the last axis:
   - `q[..., 1:] >= q[..., :-1]`
4. Any post-processing step that reduces the tensor must document which axis is reduced and why.

## H=[1] Parity Rule
Before launching the full horizon set `{1, 3, 6, 12, 24}`, the journal path must pass an `H=[1]` parity check against the legacy single-step path.

## Sentinel Rule
Any horizon-wise evaluation code should be tested on a toy tensor where each horizon contains a unique constant value, so axis-misalignment bugs are visible immediately.
