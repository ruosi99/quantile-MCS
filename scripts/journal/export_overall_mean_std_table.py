from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_LABELS = {
    "dlinear_quantile": "DLinear",
    "nlinear_quantile": "NLinear",
    "lstm_quantile": "LSTM",
    "tft_quantile": "TFT",
    "patchtst_quantile_hidden256_seq48_p8s4_lr2e4_epoch500": "PatchTST",
    "multi_scale_temporal_graph_quantile": "TGQ",
    "horizon_gated_repatchtst_p8s4_from_repatch_identity_wd0_lr5e5_epoch300":
        "HG-Re-PatchTST (Ours)",
}

METRICS = [
    ("RMSE", "RMSE", 4, "min"),
    ("MAE", "MAE", 4, "min"),
    ("MAPE", "MAPE (\\%)", 3, "min"),
    ("R2", "$R^2$", 4, "max"),
    ("WIS", "WIS-90", 4, "min"),
]


def load_overall_results(summary_dir: Path, delta: float) -> pd.DataFrame:
    point = pd.read_csv(summary_dir / "point_metrics_overall_summary.csv")
    interval = pd.read_csv(summary_dir / "raw_interval_metrics_overall_summary.csv")
    interval = interval[np.isclose(interval["delta"].astype(float), delta)].copy()

    point_columns = ["model_key"]
    for metric in ["RMSE", "MAE", "MAPE", "R2"]:
        point_columns.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_count"])
    wis_columns = ["model_key", "WIS_mean", "WIS_std", "WIS_count"]

    results = point[point_columns].merge(interval[wis_columns], on="model_key", how="inner")
    results = results[results["model_key"].isin(MODEL_LABELS)].copy()
    results["Model"] = results["model_key"].map(MODEL_LABELS)
    results["order"] = results["model_key"].map({key: i for i, key in enumerate(MODEL_LABELS)})
    results = results.sort_values("order").reset_index(drop=True)

    if len(results) != len(MODEL_LABELS):
        missing = sorted(set(MODEL_LABELS) - set(results["model_key"]))
        raise ValueError(f"Missing paper models from overall summaries: {missing}")

    count_columns = [column for column in results.columns if column.endswith("_count")]
    bad_counts = results[(results[count_columns] != 5).any(axis=1)]
    if not bad_counts.empty:
        raise ValueError(
            "Expected five seeds for every metric:\n"
            + bad_counts[["model_key", *count_columns]].to_string(index=False)
        )
    return results


def metric_cell(mean: float, std: float, decimals: int, bold: bool) -> str:
    body = f"{mean:.{decimals}f} \\pm {std:.{decimals}f}"
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def render_latex(results: pd.DataFrame) -> str:
    best_by_metric: dict[str, float] = {}
    for metric, _, _, direction in METRICS:
        values = results[f"{metric}_mean"]
        best_by_metric[metric] = float(values.min() if direction == "min" else values.max())

    rows: list[str] = []
    for row in results.itertuples(index=False):
        cells = [row.Model]
        for metric, _, decimals, _ in METRICS:
            mean = float(getattr(row, f"{metric}_mean"))
            std = float(getattr(row, f"{metric}_std"))
            cells.append(metric_cell(mean, std, decimals, np.isclose(mean, best_by_metric[metric])))
        rows.append(" & ".join(cells) + r" \\")

    headers = " & ".join(["Model", *[label for _, label, _, _ in METRICS]]) + r" \\"
    return "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Overall forecasting performance reported as mean $\pm$ standard deviation over five random seeds. For each seed, metrics are first averaged across all prediction horizons.}",
        r"\label{tab:overall_mean_std}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        headers,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the five-seed overall paper table.")
    parser.add_argument(
        "--summary-dir",
        default="journal_results/shenzhen_multihorizon/multiseed_standard_wis_summary",
    )
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir)
    output = Path(args.output) if args.output else summary_dir / "overall_mean_std_table.tex"
    results = load_overall_results(summary_dir, delta=args.delta)
    latex = render_latex(results)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(latex, encoding="utf-8")
    print(latex)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
