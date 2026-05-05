# Journal Stage 4 Decision-First Evaluation Spec

## Purpose
This document turns the decision-first journal experiment into an execution-ready specification.

The goal is to evaluate whether calibrated probabilistic forecasts improve downstream
risk-aware decisions, not just interval coverage.

This document is intended to be specific enough that Lenovo can implement and run the
experiment without needing to infer missing metric definitions.

## Why Stage 4 Comes Next
Stage 2 and Stage 2.5 changed the paper logic:
- global CQR already fixes marginal 90 percent coverage on average
- most of that gain is a zero-demand boundary-rescue effect
- the remaining gap is localized, mainly H1 positive-demand hard cells

Therefore the next paper-significant question is:

```text
Do calibrated quantiles improve decision quality under asymmetric costs?
```

This question should be answered before broad Stage 3 stratified calibration is treated as central.

## Stage 4 Scope
This stage should:
- use the current representative multi-horizon model result
- compare raw and calibrated decision rules
- quantify cost and regret by horizon and cost ratio

This stage should not:
- implement broad stratified calibration first
- require new multi-horizon training
- change the Stage 1B or Stage 2 result directories

## Representative Input Result
Use the Stage 1B warm-start model plus Stage 2 warm-start calibration outputs as the primary evaluation target.

Primary input directories:
- Stage 1 predictions:
  - `journal_results/shenzhen_multihorizon/warmstart_raw/`
- Stage 2 calibration summaries:
  - `journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw/`

Optional comparison inputs:
- Stage 1A raw:
  - `journal_results/shenzhen_multihorizon/raw/`
- Stage 2 raw-path CQR:
  - `journal_results/shenzhen_multihorizon/stage2_cqr/raw/`

The paper-facing Stage 4 run should use the warm-start path unless there is a strong reason to compare against Stage 1A.

## Methods To Compare
The minimum required method set is:
1. `median`
2. `raw_target_quantile`
3. `global_cqr`
4. `horizon_cqr`
5. `oracle_decision`

Method definitions:

### 1. `median`
- decision quantity is the raw `q0.5` prediction
- no calibration is applied

### 2. `raw_target_quantile`
- decision quantity is the raw model output at the target quantile `tau_star`
- must use an exact grid quantile from the journal 13-quantile grid
- no interpolation should be used in the main result tables

### 3. `global_cqr`
- first choose the raw target quantile at `tau_star`
- then apply the Stage 2 `global_cqr` threshold corresponding to the interval pair that contains `tau_star`
- for Stage 4 main results, keep the implementation simple and use the journal one-sided calibration routine rather than deriving decisions indirectly from two-sided intervals
- if a one-sided routine is not yet implemented, document the temporary approximation clearly

### 4. `horizon_cqr`
- same logic as `global_cqr`, but horizon-specific

### 5. `oracle_decision`
- use the realized demand `y` itself as the hindsight-optimal decision quantity
- this is the lower bound for achievable cost under perfect foresight
- all regret numbers should be defined against this oracle

## Cost Ratios And Target Quantiles
Use these exact cost ratios in the main result:

| Cost ratio | `c_u` | `c_o` | `tau_star = c_u / (c_u + c_o)` |
|---|---:|---:|---:|
| `1:1` | 1 | 1 | 0.5 |
| `3:1` | 3 | 1 | 0.75 |
| `5:1` | 5 | 1 | 0.8333333333333334 |
| `9:1` | 9 | 1 | 0.9 |
| `19:1` | 19 | 1 | 0.95 |

These all lie exactly on the current journal 13-quantile grid. Do not use interpolation in the main result.

## Decision Quantity Definition
For each sample, station, and horizon, let:
- `y` = realized demand
- `q` = decision quantity selected by the forecasting method
- `c_u` = underage cost
- `c_o` = overage cost

Decision quantity must be nonnegative.

If any method produces values below zero before clipping, apply:

```text
q = max(q, 0)
```

## Cost Function
Use the standard newsvendor-style asymmetric linear loss:

```text
cost(q, y) = c_u * max(y - q, 0) + c_o * max(q - y, 0)
```

This should be computed on the same scale as the saved Stage 1 predictions and labels.

Important:
- Stage 1 outputs are already de-normalized back to the capacity-scaled demand units used in the current pipeline
- do not re-normalize them during Stage 4

## Aggregation Rule
All primary decision metrics should be computed at the atomic level:

```text
sample × station × horizon
```

Then aggregate as follows:

### Per-horizon aggregation
For each horizon `h`, pool all samples and stations for that horizon and report:
- expected cost
- regret
- shortage rate
- overage rate

### Per-cost-ratio aggregation
For each cost ratio, average across all horizons after computing horizon-wise results.

### Overall aggregation
If an overall score is reported, use the simple arithmetic mean across horizons after the horizon-wise metrics are computed.

Do not weight horizons differently unless a later paper draft explicitly argues for it.

## Metric Definitions

### Expected Cost
For a chosen method and cost ratio:

```text
expected_cost(h) = mean(cost(q, y)) over all samples and stations at horizon h
```

### Regret vs Oracle

```text
regret(h) = expected_cost_method(h) - expected_cost_oracle(h)
```

Oracle cost should be zero under the linear formulation above if `q = y`.

If that happens, regret becomes numerically equal to expected cost, which is acceptable.
Still report both quantities for interpretability.

### Shortage Rate
Two shortage summaries should be recorded:

1. event shortage rate
```text
mean(1[q < y])
```

2. shortage amount
```text
mean(max(y - q, 0))
```

The main table should report event shortage rate.
The detailed CSV should include both.

### Overage Rate
Two overage summaries should be recorded:

1. event overage rate
```text
mean(1[q > y])
```

2. overage amount
```text
mean(max(q - y, 0))
```

The main table should report event overage rate.
The detailed CSV should include both.

## One-Sided Calibration Requirement
Stage 4 should ideally implement one-sided conformalized target-quantile calibration directly.

However, the main objective of the first Stage 4 run is comparative decision evidence.

Therefore:
- if direct one-sided calibration is ready, use it
- if not, first run Stage 4 with `median`, `raw_target_quantile`, `global_cqr`, and `horizon_cqr` using the simplest defensible mapping from Stage 2 outputs
- document the exact mapping in `decision_eval_metadata.json`

This prevents Stage 4 from being blocked by a perfect one-sided implementation.

## Input File Contract
At minimum, the Stage 4 script should read:

From Stage 1B:
- `predict_quantiles.npy`
- `label_list.npy`
- `run_metadata.json`

From Stage 2 warm-start:
- `cqr_thresholds.csv`
- `global_summary.csv`
- `raw_vs_cqr_comparison_by_horizon.csv`
- `calibrated_interval_metrics_by_horizon.csv`
- `stage2_metadata.json`

If the script only needs thresholds plus Stage 1 predictions, that is acceptable.

## Output Directory
Primary output directory:

```text
journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw/
```

## Required Output Files
The Stage 4 script must write:

1. `decision_summary_by_method_and_cost_ratio.csv`
   - one row per method and cost ratio
   - aggregated across horizons

2. `decision_metrics_by_horizon.csv`
   - one row per method, cost ratio, and horizon
   - includes expected cost, regret, shortage, overage

3. `decision_regret_vs_oracle.csv`
   - concise regret table for paper use

4. `decision_eval_metadata.json`
   - paths used
   - cost ratios
   - tau mapping
   - whether direct one-sided calibration or an approximation was used
   - branch, machine, date, and commit if available

5. `decision_cost_by_horizon.png`
   - optional but strongly recommended

6. `decision_regret_by_cost_ratio.png`
   - optional but strongly recommended

## Minimum Acceptance Criteria
The first Stage 4 run is successful if:
1. all required output files are generated
2. all methods are evaluated for all five cost ratios
3. horizon-wise and cost-ratio summaries are both available
4. the result is interpretable enough to tell whether calibrated methods improve decision quality over `median` and `raw_target_quantile`

## Preferred Paper-Facing Questions
The Stage 4 results should answer:
1. Does a calibrated decision rule reduce expected cost relative to median-only decisions?
2. Does calibration matter more at some horizons than others?
3. Does the benefit grow as underage cost becomes larger?
4. Are global and horizon-wise calibration already sufficient for decision quality, or is there still room for Stage 3 localized calibration?

## Recommended Lenovo Execution Order
1. Implement the Stage 4 script and tests.
2. Run a smoke test on a limited subset.
3. Run the full warm-start Stage 4 decision evaluation.
4. Review whether global or horizon-wise calibration already produces stable decision improvements.
5. Only then decide whether a localized Stage 3 experiment is still worth the extra complexity.

## Recommended Immediate Script Names
Suggested new files:
- `scripts/journal/evaluate_stage4_decision.py`
- `scripts/journal/run_stage4_decision_warmstart.sh`
- `tests/test_journal_stage4_decision.py`

These are recommendations only, but using this naming scheme will keep the journal pipeline easy to follow.
