from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compare_arrays(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape:
        raise AssertionError(
            f"Shape mismatch between reference {reference.shape} and candidate {candidate.shape}"
        )

    diff = candidate - reference
    return {
        "shape": list(reference.shape),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs_error": float(np.max(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
    }


def compare_metrics(reference_csv: Path, candidate_csv: Path) -> dict:
    ref_df = pd.read_csv(reference_csv)
    cand_df = pd.read_csv(candidate_csv)
    if list(ref_df.columns) != list(cand_df.columns):
        raise AssertionError("Metric CSV column mismatch")

    summary = {}
    for column in ref_df.columns:
        ref_value = float(ref_df.iloc[0][column])
        cand_value = float(cand_df.iloc[0][column])
        summary[column] = {
            "reference": ref_value,
            "candidate": cand_value,
            "abs_diff": abs(cand_value - ref_value),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare legacy and journal H=[1] outputs for Stage 0 parity.")
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--point-metrics-file", default="dura_pag_informer_quantile_on_pretrain_1bs8_point_q50.csv")
    parser.add_argument("--predict-file", default="predict_quantiles.npy")
    parser.add_argument("--label-file", default="label_list.npy")
    parser.add_argument("--report-file", default="")
    args = parser.parse_args()

    reference_dir = Path(args.reference_dir)
    candidate_dir = Path(args.candidate_dir)

    array_report = compare_arrays(
        np.load(reference_dir / args.predict_file),
        np.load(candidate_dir / args.predict_file),
    )
    label_report = compare_arrays(
        np.load(reference_dir / args.label_file),
        np.load(candidate_dir / args.label_file),
    )
    metrics_report = compare_metrics(
        reference_dir / args.point_metrics_file,
        candidate_dir / args.point_metrics_file,
    )

    report = {
        "predict_quantiles": array_report,
        "labels": label_report,
        "point_metrics": metrics_report,
    }

    report_text = json.dumps(report, indent=2)
    print(report_text)

    if args.report_file:
        Path(args.report_file).write_text(report_text, encoding="utf-8")


if __name__ == "__main__":
    main()
