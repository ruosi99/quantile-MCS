# Journal Multi-Horizon Experiment Plan

## Purpose
This document defines the recommended experiment plan for the journal-only branch:
Spatio-Temporal Stratified Conformal Multi-Horizon Forecasting.

The plan is staged so that we first establish a correct and reproducible multi-horizon baseline, then add calibration variants, then add decision-oriented evaluation.

## Planning Assumptions
- Five-city transfer is out of scope for now and will be added later.
- Shenzhen is the only active dataset target in the current phase.
- The current repository is single-step, so the first milestone is pipeline correctness, not paper-scale benchmarking.
- The current backbone should be preserved as much as possible.
- Time estimates below are rough working estimates for one implementation cycle, not hard deadlines.

## Overall Goal
Build a paper-ready Shenzhen experiment stack that can support the following claims:
1. the current method extends from single-step to direct multi-horizon forecasting
2. horizon-aware and stratified calibration improve reliability over global calibration
3. the calibrated forecasts produce more useful downstream decisions under asymmetric costs

## Recommended Execution Order
1. Dataset and split audit
2. Infrastructure, contract validation, and H=[1] parity
3. Multi-horizon baseline
4. Global and horizon-wise calibration
5. Diagnostic gate on conditional reliability and boundary effects
6. Decision-oriented one-sided calibration and decision evaluation
7. Conditional stratified calibration only if diagnostics justify it
8. Robustness, ablations, and final figure/table packaging

## Stage 0: Pre-Implementation Audit And Infrastructure

### Experiment 0.0: Dataset / Split / Station Audit
- Purpose: freeze the exact Shenzhen data definition before journal-only experiments begin
- Required outputs:
  - `station_manifest.csv`
  - `split_manifest.json`
  - `adjacency_snapshot.npz`
  - `horizon_config.json`
- Expected time:
  - implementation and export: 0.5 to 1 day
  - verification: 0.5 day
- Success condition:
  - station count, station IDs, split boundaries, adjacency version, and horizon definitions are all fixed and reviewable
- Why it matters:
  - this closes a likely reviewer question around inconsistent station counts or silent dataset drift

### Experiment 0.1: Tensor Contract and Output Convention
- Purpose: lock down one consistent shape convention for multi-horizon outputs and labels before any heavy coding
- Deliverable:
  - written shape contract in docs and code comments
  - fixed result naming convention
  - journal-specific result root
- Expected time:
  - coding: 0.5 to 1 day
  - verification: 0.5 day
- Success condition:
  - all affected modules agree on one convention, preferably `(B, N, H, Q)` for quantiles and `(B, N, H)` for labels
- Why it matters:
  - this prevents silent horizon misalignment bugs later

### Experiment 0.1b: Contract Tests and Sentinel Tests
- Purpose: turn the tensor contract into executable checks rather than only documentation
- Required tests:
  - shape asserts for all major training, inference, calibration, and evaluation entry points
  - `H=[1]` parity test against the single-step path
  - horizon sentinel test with toy labels that differ by horizon
  - monotonicity assert for quantiles
- Expected time:
  - implementation: 0.5 to 1 day
  - verification: 0.5 day
- Success condition:
  - the test suite catches last-horizon-only bugs, shape drift, and monotonicity violations before long GPU runs
- Why it matters:
  - this is the main defense against silently wrong multi-horizon results

### Experiment 0.2: Journal Run Script Scaffold
- Purpose: create journal-only run entry points without breaking the conference path
- Deliverable:
  - one training script
  - one evaluation script
  - one calibration script or unified post-processing entry point
- Expected time:
  - coding: 0.5 day
  - smoke checks: 0.5 day
- Success condition:
  - conference single-step path still exists unchanged
  - journal path writes to a clearly separated output directory

### Experiment 0.3: H=[1] Parity / Regression Run
- Purpose: make the single-horizon journal path a hard gate before full multi-horizon training
- Required outputs:
  - one `H=[1]` journal-path run
  - comparison against the legacy single-step path
- Expected time:
  - code and run setup: 0.5 day
  - run and comparison: 0.5 to 1 day
- Success condition:
  - the new path reproduces the old single-step behavior closely enough to be trusted
- Why it matters:
  - this should be treated as a hard precondition, not an optional regression check

### Experiment 0.4: Quantile Grid Redesign
- Purpose: redesign the quantile grid early so the later decision-calibration stage does not depend on interpolation by default
- Recommended grid:
  - `{0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 5/6, 0.9, 0.95}`
- Expected time:
  - implementation: 0.5 day
  - smoke check: 0.5 day
- Success condition:
  - the chosen grid supports the target critical fractiles used in the decision study
- Why it matters:
  - it avoids relying on interpolation for `tau_star` in the main paper story

## Stage 1: Multi-Horizon Forecasting Baseline

### Experiment 1.1: Direct Multi-Horizon Forecasting Without Calibration
- Purpose: prove the model can produce direct forecasts for `H = {1, 3, 6, 12, 24}`
- Required outputs:
  - saved `predict_quantiles`
  - saved labels
  - point metrics by horizon
  - raw interval metrics by horizon
- Expected time:
  - implementation: 2 to 4 days
  - first GPU training cycle: 0.5 to 1.5 days
  - debugging and rerun: 1 to 2 days
- Success condition:
  - training runs end to end
  - output tensors are correct for all horizons
  - no systematic quantile crossing
  - long horizons are worse than short horizons, but not catastrophically unstable
- Minimum scientific purpose:
  - establish the non-calibrated direct multi-horizon baseline that all later calibration methods build on

## Stage 2: Baseline Calibration Comparisons

### Experiment 2.1: Global CQR for Multi-Horizon
- Purpose: port the current global CQR logic to multi-horizon outputs
- Required outputs:
  - one global threshold strategy
  - per-horizon PICP, ACE, MPIW, WIS
- Expected time:
  - implementation: 1 to 2 days
  - evaluation run: 0.5 day
- Success condition:
  - global CQR produces meaningful coverage improvement over raw intervals
  - horizon-level metrics reveal where global calibration still fails
- Scientific purpose:
  - this is the first reference baseline for the journal story

### Experiment 2.2: Horizon-Wise CQR
- Purpose: test whether separate calibration by horizon already fixes most reliability issues
- Required outputs:
  - threshold per horizon
  - coverage curves and reliability curves by horizon
- Expected time:
  - implementation: 1 day
  - evaluation run: 0.5 day
- Success condition:
  - horizon-wise CQR improves ACE or coverage stability relative to one global threshold
- Scientific purpose:
  - isolates the value of horizon awareness before adding more complex stratification

### Stage 2 Evidence Update: 2026-05-01
The full Stage 2 run on the Stage 1B warm-start model changed the calibration story.

For the primary 90 percent interval, global CQR already corrected mean PICP from `0.638917`
to `0.913388`, with mean MPIW changing only from `2.916468` to `2.916553`.

The large coverage gain came mostly from zero-demand lower-bound misses near the nonnegative
boundary. A very small CQR threshold moves the lower endpoint to zero after clipping, rescuing
many true-zero observations without materially widening intervals.

Implication:
- adaptive or stratified calibration is no longer needed as the main mechanism for fixing
  marginal 90 percent coverage
- adaptive or stratified calibration should now be evaluated as a localized reliability method:
  worst-cell ACE, hour-by-horizon under-coverage, lower nominal coverage settings, and
  station/time subgroups
- if it does not improve localized worst-cell reliability without unacceptable width inflation,
  it should be reported as exploratory rather than central

## Stage 2.5 / Stage 4.0: Diagnostic Gate

### Purpose
Before implementing any stratified or hierarchical conformal method, determine whether
global and horizon-wise CQR still leave a meaningful conditional reliability gap that
justifies a more complex calibration layer.

This stage should be treated as a gate, not as optional plotting.

### Experiment A1: Zero vs Positive-Demand Decomposition
- Purpose: separate average coverage gains caused by zero-demand boundary rescue from gains on positive-demand cases
- Required outputs for `raw`, `global_cqr`, and `horizon_cqr`:
  - `PICP_all`
  - `PICP_zero`
  - `PICP_positive`
  - `ACE_positive`
  - `MPIW_positive`
  - `WIS_positive`
- Scientific purpose:
  - determine whether the remaining problem is primarily a boundary phenomenon or a broader positive-demand calibration problem

### Experiment A2: Boundary-Rescue Decomposition
- Purpose: quantify how much of the CQR improvement comes from lower-bound clipping near zero
- Required outputs:
  - count of `raw miss -> calibrated cover`
  - count of rescues caused by lower bound crossing below zero and clipping back to zero
  - horizon-wise rescue summary
- Scientific purpose:
  - provide a defensible explanation for large PICP gains with near-zero MPIW change

### Experiment A3: Hour × Horizon Diagnostics
- Purpose: inspect whether important localized failure modes remain after global or horizon-wise calibration
- Required outputs:
  - PICP heatmap, all samples
  - ACE heatmap, all samples
  - PICP heatmap, positive-demand only
  - ACE heatmap, positive-demand only
- Scientific purpose:
  - identify whether peak-hour or long-horizon hard cells still under-cover in an operationally meaningful way

### Experiment A4: Demand-Bin Diagnostics
- Purpose: check whether calibration quality differs across demand magnitude regimes
- Minimum bins:
  - `y = 0`
  - `0 < y <= q50`
  - `q50 < y <= q90`
  - `y > q90`
- Required outputs:
  - coverage, ACE, and width summaries by demand bin and horizon
- Scientific purpose:
  - separate zero-demand boundary gains from tail-demand reliability behavior

### Experiment A5: Reliability Curves by Horizon
- Purpose: compare reliability behavior across horizons after Stage 2 methods
- Required outputs:
  - horizon-wise reliability curves for `raw`, `global_cqr`, and `horizon_cqr`
- Scientific purpose:
  - determine whether a more complex calibration method is still justified after baseline CQR

### Stage 2.5 Gate Rule
Continue to Stage 3 as a main methodological thread only if at least one of the following remains true after Stage 2:
- positive-demand `PICP@90` is clearly below `0.85` on any main horizon
- worst-cell `ACE` remains above roughly `0.08-0.10`
- peak-hour or long-horizon hard cells remain materially under-covered
- decision regret remains unstable under global or horizon-wise calibration

If these conditions are not met, Stage 3 should be narrowed and the paper should pivot toward:
- direct multi-horizon forecasting
- boundary-aware reliability diagnostics
- decision-calibrated quantiles

In that case, stratified calibration should be treated as supplementary, exploratory, or appendix material rather than as the central contribution.

## Stage 3: Conditional Stratified Conformal Calibration

### Experiment 3.1: Station Archetype Construction
- Purpose: build cheap and reproducible station archetypes from summary features
- Proposed features:
  - mean load
  - volatility
  - peak ratio
  - graph degree
- Required outputs:
  - archetype assignment file
  - archetype size summary
  - feature summary by cluster
- Expected time:
  - implementation: 1 day
  - inspection and refinement: 0.5 to 1 day
- Success condition:
  - clusters are interpretable enough to justify in the methods section
  - no tiny clusters dominate the partition
- Scientific purpose:
  - gives the stratified calibration a concrete operational definition

### Experiment 3.2: Stratified Calibration With Fallback
- Purpose: if Stage 2.5 shows a real conditional reliability gap, test whether hierarchical stratified calibration can improve localized reliability
- Default main design:
  - target-hour bins: `00-05`, `06-10`, `11-15`, `16-20`, `21-23`
  - archetypes: start with `K=3`
- Hierarchical fallback order:
  - `h × timebin × archetype`
  - fallback to `h × timebin`
  - fallback to `h`
- Required outputs:
  - stratified thresholds
  - effective sample size report per cell
  - `effective_cell_id`
  - `fallback_level`
  - `cell_size`
  - `threshold_used`
  - fallback usage statistics by horizon and by time bin
- Expected time:
  - implementation: 2 to 3 days
  - evaluation run: 0.5 to 1 day
- Success condition:
  - stratified calibration does not collapse because of sparse cells
  - compared with global and horizon-wise baselines, it improves positive-demand ACE, worst-cell ACE, tail-demand PICP, or hard-cell under-coverage without unacceptable interval inflation
- Scientific purpose:
  - this is only a core journal experiment if the Stage 2.5 gate shows that simpler calibration is still insufficient

### Experiment 3.3: Stratification Granularity Ablation
- Purpose: test how sensitive results are to bucket design
- Suggested settings:
  - 5 coarse target-hour bins as the main configuration
  - 4 coarse target-hour bins
  - `K=2` versus `K=3` archetypes
  - 24 hourly bins only as a negative-control ablation
- Expected time:
  - implementation reuse: 0.5 day
  - runs and analysis: 1 to 2 days
- Success condition:
  - there is at least one granularity setting that is both stable and beneficial
- Scientific purpose:
  - defends against the “arbitrary partition” reviewer criticism

## Stage 4: Decision-First Evaluation

### Experiment 4.1: One-Sided Critical-Fractile Calibration
- Purpose: calibrate the economically relevant target quantile `tau_star = c_u / (c_u + c_o)`
- Required outputs:
  - one-sided calibration routine
  - calibrated target-quantile predictions
- Expected time:
  - implementation: 1 to 2 days
  - evaluation run: 0.5 day
- Success condition:
  - the method handles target quantiles consistently for selected cost ratios
  - the procedure is simple enough to explain cleanly in the paper
- Scientific purpose:
  - connects uncertainty quantification to operational decisions

### Experiment 4.2: Decision Cost and Regret Under Multiple Cost Ratios
- Purpose: make probabilistic forecasting value visible through downstream decision quality
- Required methods:
  - median
  - raw target quantile
  - global CQR
  - horizon-wise CQR
  - stratified CQR if Stage 3 is activated
  - oracle quantile or oracle decision
- Required cost-ratio settings:
  - `1:1` -> `tau_star = 0.5`
  - `3:1` -> `tau_star = 0.75`
  - `5:1` -> `tau_star = 0.8333`
  - `9:1` -> `tau_star = 0.9`
  - `19:1` -> `tau_star = 0.95`
- Required outputs:
  - expected cost
  - regret versus oracle
  - shortage rate
  - overage rate
  - horizon-wise summaries
  - cost-ratio summaries
- Expected time:
  - implementation: 1 to 2 days
  - analysis: 0.5 to 1 day
- Success condition:
  - at least one calibrated strategy clearly improves cost or regret relative to median and raw target-quantile choices
- Scientific purpose:
  - this is now a main experiment, not a late optional extension

## Stage 5: Robustness, Diagnostics, And Conditional Extensions

### Experiment 5.1: Hour-by-Horizon Coverage Heatmap
- Purpose: visualize where calibration succeeds or fails
- Required outputs:
  - hour × horizon PICP heatmap
  - hour × horizon ACE heatmap
  - optionally width heatmap
- Expected time:
  - implementation: 0.5 to 1 day
  - analysis: 0.5 day
- Success condition:
  - figures clearly show whether global calibration hides structured failure modes
- Scientific purpose:
  - likely one of the most persuasive paper figures

### Experiment 5.2: Reliability Curves by Horizon
- Purpose: evaluate calibration quality across horizons in a reviewer-friendly way
- Required outputs:
  - reliability curves by horizon
  - calibration error summary
- Expected time:
  - implementation: 0.5 to 1 day
  - analysis: 0.5 day
- Success condition:
  - curves show consistent improvement after horizon-wise or stratified calibration
- Scientific purpose:
  - supports the reliability claim more clearly than only reporting PICP

### Experiment 5.3: Calibration-Set Size Ablation
- Purpose: test how much calibration data is required
- Suggested settings:
  - 25 percent
  - 50 percent
  - 75 percent
  - 100 percent of the planned calibration split
- Expected time:
  - implementation reuse: 0.5 day
  - runs and analysis: 1 day
- Success condition:
  - results are not extremely fragile to moderate calibration-set reductions
- Scientific purpose:
  - supports practical deployment claims

### Experiment 5.4: Sparse-Strata Stress Test
- Purpose: quantify how often fine stratification becomes unreliable
- Required outputs:
  - number of under-sized cells
  - performance with and without fallback
- Expected time:
  - implementation reuse: 0.5 day
  - analysis: 0.5 day
- Success condition:
  - fallback behavior can be justified empirically
- Scientific purpose:
  - directly addresses the most obvious reviewer concern

### Experiment 5.5: Nominal-Level Sensitivity
- Purpose: verify that the journal setup remains credible across more than one nominal coverage target
- Suggested settings:
  - `delta in {0.1, 0.2, 0.4}`
- Expected time:
  - implementation reuse: 0.5 day
  - runs and analysis: 0.5 to 1 day
- Success condition:
  - the relative behavior of calibration methods remains stable across nominal levels
- Scientific purpose:
  - preserves continuity with the conference evidence while strengthening the journal version

### Experiment 5.6: Worst-Cell Reliability Summary
- Purpose: summarize whether calibration reduces the most dangerous local failure modes rather than only improving averages
- Required outputs:
  - worst-cell PICP
  - worst-cell ACE
  - average ACE
  - interval width inflation versus raw
  - fallback usage percentage
- Expected time:
  - implementation reuse: 0.5 day
  - analysis: 0.5 day
- Success condition:
  - stratified or hierarchical calibration improves the worst operational cells without unacceptable width inflation
- Scientific purpose:
  - directly supports the paper's local-failure-mode narrative

### Experiment 5.7: Probabilistic-Component Ablation
- Purpose: separate the value of the monotonic quantile head from the value of the calibration layer
- Required comparisons:
  - independent quantile head versus monotonic incremental-softplus head
  - raw quantile versus global CQR versus horizon-wise CQR versus stratified CQR
- Expected time:
  - implementation reuse plus one added model path: 1 to 2 days
  - runs and analysis: 1 to 2 days
- Success condition:
  - the decomposition clarifies which gains come from non-crossing output design and which come from calibration
- Scientific purpose:
  - addresses a known ablation weakness in the conference line and strengthens the journal story

## Stage 7: Final Packaging

### Experiment 7.1: Final Table and Figure Export
- Purpose: convert experiment outputs into paper-ready assets
- Required outputs:
  - main comparison table
  - calibration table
  - decision table
  - heatmap figure
  - reliability figure
- Expected time:
  - scripting: 1 day
  - polishing: 1 day
- Success condition:
  - tables and figures can be inserted into the paper without manual reconstruction

## Suggested Total Timeline

### Minimal Viable Journal Stack
- Stage 0 to Stage 2.5
- Estimated duration: 8 to 12 working days
- Outcome:
  - multi-horizon forecasting
  - global and horizon-wise calibration
  - diagnostic gate evidence

### Strong Journal Core
- Stage 0 to Stage 4
- Estimated duration: 15 to 22 working days
- Outcome:
  - multi-horizon forecasting
  - calibrated uncertainty
  - decision-oriented evaluation on Shenzhen

### Stronger Submission Package
- Stage 0 to Stage 7
- Estimated duration: 20 to 30 working days
- Outcome:
  - main results, diagnostics, ablations, and paper-ready assets

## Priority Ranking

### Must-Have
1. direct multi-horizon baseline
2. global CQR
3. horizon-wise CQR
4. diagnostic gate with zero/positive-demand and hour-by-horizon decomposition
5. one decision-cost experiment

### Conditional After Stage 2
1. stratified calibration with hierarchical fallback, only if framed as a localized reliability or worst-cell improvement experiment
2. adaptive calibration ablations, only if they improve worst-cell ACE or localized under-coverage without unacceptable width inflation

### Strongly Recommended
1. granularity ablation
2. calibration-set size ablation
3. worst-cell reliability summary
4. probabilistic-component ablation

### Nice-To-Have
1. extra scoring rules beyond the main set
2. expanded cluster sensitivity study
3. five-city calibration-only transfer after Shenzhen is stable

## Pivot Rule
If stratified or hierarchical calibration does not clearly improve worst-cell ACE or under-coverage relative to horizon-wise CQR, or if it requires unacceptable interval-width inflation, then the paper should pivot.

The fallback paper framing should become:
- direct multi-horizon forecasting
- horizon-wise calibration
- decision-oriented quantile calibration

In that case, stratified calibration should be reported as an exploratory extension rather than the central contribution.

## Exit Criteria For The Current Phase
The Shenzhen-only phase can be considered successful when:
1. the multi-horizon pipeline is stable and reproducible
2. horizon-wise metrics are complete
3. at least one calibration strategy clearly improves reliability over raw intervals
4. the decision-oriented evaluation shows practical value
5. the stratified method is either validated on worst-cell reliability or honestly narrowed with evidence

## Recommended Immediate Next Step
Do not implement stratified calibration next by default.

The immediate next checkpoint should be the Diagnostic Gate:
- zero versus positive-demand decomposition
- boundary-rescue decomposition
- hour-by-horizon diagnostics for all samples and positive-demand-only samples
- demand-bin diagnostics
- reliability curves by horizon

Only after this gate should Stage 3 be confirmed as a main path or narrowed to an exploratory extension.
