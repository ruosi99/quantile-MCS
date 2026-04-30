# Journal Stage 0 Runbook

## Purpose
This runbook defines the remaining operational steps needed to close Stage 0 on Lenovo.

## What Stage 0 Already Locks
- dataset and split manifests
- horizon and quantile configuration
- tensor contract
- contract and audit tests on Dell

## Remaining Stage 0 Gate
The last hard gate before Stage 1 is the `H=[1]` journal parity check.

## Files To Use
- [journal_h1_run_spec.json](/C:/Users/100064422/PhD_projects/quntile/mcs-mcs_quantile/mcs-mcs_quantile/docs/journal_stage0/journal_h1_run_spec.json)
- [journal_multihorizon_stub_run_spec.json](/C:/Users/100064422/PhD_projects/quntile/mcs-mcs_quantile/mcs-mcs_quantile/docs/journal_stage0/journal_multihorizon_stub_run_spec.json)
- [run_journal_h1_parity.sh](/C:/Users/100064422/PhD_projects/quntile/mcs-mcs_quantile/mcs-mcs_quantile/scripts/journal/run_journal_h1_parity.sh)
- [check_h1_parity.py](/C:/Users/100064422/PhD_projects/quntile/mcs-mcs_quantile/mcs-mcs_quantile/scripts/journal/check_h1_parity.py)

## Lenovo Sequence
1. Pull the current branch.
2. Run:
```bash
bash scripts/journal/run_journal_h1_parity.sh
```
3. The script copies the journal candidate output into `journal_results/stage0_h1_parity/`.
4. The script compares it against the legacy reference and writes:
```text
docs/journal_stage0/h1_parity_report.json
```

## Expected Output
- `docs/journal_stage0/h1_parity_report.json`

## Acceptance Rule
Stage 0 can be treated as operationally closed when:
- the journal `H=[1]` path runs successfully
- the parity report is generated
- any differences versus the legacy path are reviewed before Stage 1 starts

## Notes
- The multi-horizon run spec is intentionally still a stub at Stage 0.
- Full `{1,3,6,12,24}` training should not begin until the parity gate is reviewed.
