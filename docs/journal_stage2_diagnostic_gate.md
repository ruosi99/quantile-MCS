# Journal Stage 2.5 Diagnostic Gate Log

## Metadata
- Date: 2026-05-05
- Machine: Lenovo, Windows GPU experiment machine
- Dell meaning: original/editing machine; this document records Lenovo outputs that can be copied back to Dell
- Branch: `multi_horizon_journal`
- Base commit at diagnosis time: `7044c2c`
- Environment: `py12`, Python 3.12, CUDA-capable Lenovo environment
- Representative Stage 1 model: Stage 1B warm-start raw multi-horizon model

## Purpose
Stage 2 global CQR already fixed the marginal 90 percent coverage problem on the
warm-start multi-horizon run. The diagnostic gate checks whether this improvement
reflects broader positive-demand calibration or mainly a zero-demand boundary effect.

This gate is meant to decide whether Stage 3 stratified/adaptive calibration should
become a main paper path, a localized reliability experiment, or an exploratory appendix.

## Code Added
The following diagnostic code was added on Lenovo:

- `scripts/journal/analyze_stage2_diagnostics.py`
- `tests/test_journal_stage2_diagnostics.py`

The diagnostic script reads Stage 1 prediction tensors and Stage 2 CQR summaries,
then writes:

- zero-vs-positive demand coverage summaries
- boundary-rescue decomposition
- hour-by-horizon coverage summaries for all samples
- hour-by-horizon coverage summaries for positive-demand samples
- demand-bin coverage summaries
- reliability summaries by horizon
- a machine-readable diagnostic gate JSON

No Stage 3 calibration code was implemented in this step.

## Verification
Test command:

```powershell
conda run -n py12 python -m pytest tests/test_journal_stage2_diagnostics.py tests/test_journal_stage2_cqr.py -q -p no:cacheprovider
```

Result:

- `8 passed`
- `1 warning`

## Diagnostic Run
Command:

```powershell
conda run -n py12 python scripts/journal/analyze_stage2_diagnostics.py --stage1-output-dir journal_results/shenzhen_multihorizon/warmstart_raw --stage2-output-dir journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw --output-dir journal_results/shenzhen_multihorizon/stage2_diagnostics/warmstart_raw --deltas 0.1,0.2,0.4 --gate-method global_cqr --gate-delta 0.1
```

Input directories:

- Stage 1B predictions: `journal_results/shenzhen_multihorizon/warmstart_raw/`
- Stage 2 CQR outputs: `journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw/`

Output directory:

- `journal_results/shenzhen_multihorizon/stage2_diagnostics/warmstart_raw/`

Generated files:

- `zero_vs_positive_summary.csv`
- `boundary_rescue_summary.csv`
- `hour_horizon_coverage_all.csv`
- `hour_horizon_coverage_positive_only.csv`
- `demand_bin_coverage_by_horizon.csv`
- `reliability_by_horizon.csv`
- `diagnostic_gate_summary.json`

## Key Results At 90 Percent Nominal Coverage

### Marginal Coverage
Stage 2 global CQR remains strong on average:

- Raw mean PICP: `0.638917`
- Global CQR mean PICP: `0.913388`
- Raw mean MPIW: `2.916468`
- Global CQR mean MPIW: `2.916553`

This explains why Stage 2 looked surprisingly good in the aggregate.

### Positive-Demand Coverage
Positive-demand aggregate coverage is not catastrophically low under global CQR.
The worst horizon is H1:

- Minimum positive-demand PICP: `0.879434`
- Worst positive-demand horizon: H1
- H1 positive-demand ACE: `0.020566`
- H1 positive-demand undercoverage: `0.020566`

This does not trigger the planned `PICP@90 < 0.85` gate condition.

### Boundary Rescue
The main Stage 2 gain is a boundary effect. The global CQR threshold at 90 percent is
very small:

- `s_hat = 4.817547960556112e-05`

The improvement comes almost entirely from true-zero observations that raw intervals
missed at the lower bound. After adding the tiny threshold and applying nonnegative
clipping, many zero-demand labels become covered.

Global CQR rescue summary by horizon:

| Horizon | Raw Misses | Rescued | Zero Share Of Rescues | Clipping Share Of Rescues |
|---:|---:|---:|---:|---:|
| 1 | 551050 | 400644 | 0.999940 | 0.999940 |
| 3 | 512670 | 399621 | 0.999967 | 0.999967 |
| 6 | 494731 | 390069 | 0.999974 | 0.999974 |
| 12 | 480785 | 365678 | 0.999992 | 0.999992 |
| 24 | 478194 | 357569 | 0.999989 | 0.999989 |

Mean zero share of rescues:

- `0.9999725`

Mean clipping share of rescues:

- `0.9999725`

Interpretation:

- CQR did not materially widen the intervals.
- Most rescued points were zero-demand lower-bound misses.
- The aggregate PICP gain should be explained as boundary-aware correction, not as a
  broad positive-demand reliability improvement.

### Local Hour-By-Horizon Reliability
Global CQR still leaves localized hard cells.

Worst all-sample hour-by-horizon cell:

- Horizon: H1
- Target hour: 10
- PICP: `0.819822`
- ACE: `0.080178`
- Undercoverage: `0.080178`

Worst positive-demand-only hour-by-horizon cell:

- Horizon: H1
- Target hour: 18
- PICP: `0.780394`
- ACE: `0.119606`
- Undercoverage: `0.119606`

This triggers the planned worst-cell ACE gate condition.

### Demand-Bin Diagnostics
Tail demand is not the main failure mode at 90 percent.

- Worst tail-bin PICP: `0.940317`
- Worst tail-bin horizon: H3
- Tail-bin undercoverage: `0.0`

The more important remaining issue is localized positive-demand H1 undercoverage,
especially in specific target-hour cells.

## Stage 2.5 Gate Decision
Gate method:

- `global_cqr`

Gate delta:

- `0.1`

Nominal coverage:

- `0.9`

Gate criteria:

| Criterion | Result |
|---|---|
| Positive-demand PICP below 0.85 | Not triggered |
| Worst-cell ACE above 0.08 | Triggered |
| Tail-demand PICP below 0.85 | Not triggered |
| Decision regret evaluated | Not yet evaluated |

Decision:

Stage 3 remains scientifically justified only as a conditional localized-reliability
experiment. It should not be framed as necessary for marginal 90 percent coverage,
because global CQR already fixes the average coverage target.

Recommended framing:

- Main finding so far: Stage 2 global CQR fixes marginal coverage through boundary rescue.
- Remaining reliability gap: H1 positive-demand and selected hour-by-horizon cells.
- Stage 3, if run later, should target localized worst-cell reliability and positive-demand
  undercoverage.
- The next paper-significant direction should be decision-oriented evaluation before a
  broad stratified calibration implementation is treated as central.

## What Not To Do Automatically
Do not start Stage 3 just because the diagnostic gate triggered one criterion.

A better next decision checkpoint is:

1. review the diagnostic evidence with the advisor/group
2. decide whether localized H1 positive-demand undercoverage is important enough for the paper story
3. run decision-cost evaluation before making stratified calibration the main contribution

## Transfer Notes From Lenovo To Dell
If copying only result artifacts back to Dell, copy:

- `journal_results/shenzhen_multihorizon/stage2_diagnostics/warmstart_raw/`

For full reproducibility on Dell, also commit or transfer:

- `scripts/journal/analyze_stage2_diagnostics.py`
- `tests/test_journal_stage2_diagnostics.py`
- `docs/journal_stage2_diagnostic_gate.md`
- any updated task or experiment-plan documents

The large Stage 1B prediction tensors are still required if Dell needs to rerun the
diagnostics from scratch:

- `journal_results/shenzhen_multihorizon/warmstart_raw/predict_quantiles.npy`
- `journal_results/shenzhen_multihorizon/warmstart_raw/label_list.npy`

