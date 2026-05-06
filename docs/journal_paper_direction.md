# Journal Paper Direction

## Current Main Narrative
The journal paper should no longer be framed as "stratified conformal calibration is the main new method."

The current evidence supports a stronger and more honest main narrative:

> In zero-inflated, nonnegative urban EV charging demand forecasting, marginal coverage can be misleading. The real operational value of probabilistic forecasting comes mainly from cost-aligned quantile selection, while calibration acts as a lightweight reliability safeguard.

Working title direction:

> Boundary-Aware Reliability Diagnostics and Decision-Value Attribution for Multi-Horizon EV Charging Demand Forecasting

Alternative wording:

> Beyond Marginal Coverage: Boundary-Aware Reliability and Decision-Aligned Probabilistic Forecasting for Urban EV Charging Demand

## Why This Narrative Fits The Evidence

### Stage 1
- the single-step conference pipeline has been extended to direct multi-horizon forecasting
- the journal system now runs on `H = {1, 3, 6, 12, 24}` and `Q = 13`
- this is a real journal extension, not only post-processing

### Stage 2
- global CQR substantially improves marginal 90 percent coverage
- the improvement comes with almost no width inflation
- however, this turns out to be largely a zero-demand boundary-rescue phenomenon

### Stage 2.5
- positive-demand aggregate coverage is not catastrophically poor after global CQR
- the remaining issue is localized hard cells, especially H1 positive-demand hour cells
- this means broad Stage 3 stratified calibration is not automatically justified as the paper's main method

### Stage 4
- the major expected-cost reduction comes from choosing the cost-aligned target quantile instead of the median
- one-sided calibration provides only a small additional gain
- therefore decision value is mainly attributable to quantile choice, while calibration is a secondary safeguard

## Main Claim Structure
1. Extend the conference single-step forecaster to a direct multi-horizon probabilistic setting.
2. Show that marginal coverage alone can create a reliability illusion in zero-inflated, nonnegative EV demand.
3. Quantify this illusion through boundary-aware reliability decomposition.
4. Show that operational value comes primarily from cost-aligned quantile selection.
5. Position calibration as a lightweight reliability safeguard rather than as the main value generator.

## Role Of Stage 3
Stage 3 should now be treated as conditional and narrow.

If pursued, it should answer:

> Can a localized calibration design improve H1 positive-demand hard cells without becoming a broad, expensive, low-yield stratified framework?

This is no longer the default main contribution route.

## Immediate Experiment Blocks

### Block 1
Boundary-aware reliability decomposition:
- `PICP_all`
- `PICP_zero`
- `PICP_positive`
- `PICP_top10`
- `ACE_positive`
- `MPIW_positive`
- `WIS_positive`
- zero-boundary rescue share

### Block 2
Decision-value attribution:
- median
- raw target quantile
- global one-sided calibration
- horizon-wise one-sided calibration
- oracle decision

Report:
- expected cost
- regret versus oracle
- shortage and overage summaries
- quantile-choice share
- calibration share

### Block 3
Statistical robustness:
- bootstrap confidence intervals for the main cost and attribution claims

### Block 4
Conditional localized calibration:
- only if the previous blocks show that the remaining hard-cell gap is important enough to justify it

## Current Recommendation
Do not continue broad Stage 3 by default.

The most journal-efficient next step is:
1. formalize the boundary-aware reliability decomposition
2. formalize the decision-value attribution
3. add confidence intervals
4. only then decide whether a narrow localized calibration experiment is worth the extra time

## Evidence Assets Update: 2026-05-06
The first paper-facing evidence package has been exported on Lenovo. See:

- `docs/journal_paper_assets.md`
- `journal_results/shenzhen_multihorizon/paper_assets/`

The export includes:
- boundary-aware reliability table
- coverage-gain decomposition figure
- positive-demand hard-cell summary
- decision-value attribution table
- cost-by-quantile curve
- test-window bootstrap confidence intervals

Current evidence after this export:
- zero-boundary rescue explains almost all Stage 2 coverage gain
- positive-demand hard cells remain localized, mainly H1 target-hour cells
- decision-value gain is almost entirely attributable to cost-aligned quantile choice
- conformal calibration is consistent but tiny as a decision-cost refiner

This strengthens the current recommendation:
- do not start broad Stage 3 by default
- move next toward paper-table/figure polishing
- consider only a narrow H1 positive-demand localized calibration experiment if the paper needs an additional targeted reliability result

## Reinforcement Package After The First Evidence Export
The next high-yield additions should reinforce the current narrative rather than reopen the
full Stage 3 direction.

### Reinforcement 1: Station-Level Zero-Inflation Sensitivity
- group stations by station-level zero ratio
- report `ZBR_share`, positive-demand coverage gap, and `quantile_choice_share` by group
- use this to show whether the boundary-aware and decision-attribution findings remain strong
  even when zero inflation is not extreme

### Reinforcement 2: Deployment Strategy Comparison
- compare:
  - median decision
  - symmetric interval upper-bound decision
  - raw cost-aligned quantile decision
  - one-sided refined cost-aligned quantile decision
- this turns the attribution result into a direct deployment recommendation and shows that
  calibration alone is not enough if the deployed decision rule ignores the cost ratio

### Reinforcement 3: Deployment Diagnostic Flowchart
- convert the paper's conclusions into a simple four-step deployment protocol:
  - train the forecaster
  - run boundary-aware diagnostics
  - choose the cost-aligned quantile
  - use targeted calibration only for diagnostically identified hard cells
- this should be treated as a practical packaging contribution, not a new heavy experiment

## Updated Next-Step Priority
1. polish the current paper-facing figures and tables
2. add station-level zero-inflation sensitivity
3. add deployment strategy comparison
4. optionally upgrade bootstrap to day-level or station-day if date indices are exported
5. only then decide whether a narrow H1 positive-demand localized calibration experiment is still needed
