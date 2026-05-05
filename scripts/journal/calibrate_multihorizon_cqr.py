from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import utils.model_training.training_utils as fn
from scripts.journal.train_multihorizon_raw import parse_bool, parse_float_list, quantile_index
from utils.model_training.conformal import interval_metrics
from utils.model_training.journal_contracts import (
    assert_monotonic_quantiles,
    assert_quantile_tensor_shape,
    assert_target_tensor_shape,
)


DEFAULT_DELTAS = [0.1, 0.2, 0.4]
METRIC_COLUMNS = [
    "method",
    "horizon",
    "delta",
    "nominal_coverage",
    "lower_quantile",
    "upper_quantile",
    "s_hat",
    "PICP",
    "coverage_gap",
    "ACE",
    "undercoverage",
    "MPIW",
    "WIS",
]


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def resolve_path(path_value: str | Path, base: Path = REPO_ROOT) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return base / path


def dataset_path_string(path: Path) -> str:
    value = str(path)
    if not value.endswith(("/", "\\")):
        value += "/"
    return value


def conformal_quantile(scores: np.ndarray, delta: float) -> float:
    flat_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if flat_scores.size == 0:
        raise ValueError("CQR score array is empty.")

    sorted_scores = np.sort(flat_scores)
    k = int(np.ceil((sorted_scores.size + 1) * (1.0 - delta))) - 1
    k = min(max(k, 0), sorted_scores.size - 1)
    return float(sorted_scores[k])


def compute_cqr_scores(
    predict_quantiles: np.ndarray,
    labels: np.ndarray,
    quantiles: list[float],
    delta: float,
) -> np.ndarray:
    assert_quantile_tensor_shape(
        predict_quantiles,
        n_horizons=labels.shape[2],
        n_quantiles=len(quantiles),
        context="calibration predict_quantiles",
    )
    assert_target_tensor_shape(labels, n_horizons=predict_quantiles.shape[2], context="calibration labels")

    lower_idx = quantile_index(quantiles, delta / 2.0)
    upper_idx = quantile_index(quantiles, 1.0 - delta / 2.0)
    lower = predict_quantiles[..., lower_idx]
    upper = predict_quantiles[..., upper_idx]
    return np.maximum(lower - labels, labels - upper)


def compute_global_threshold(scores: np.ndarray, delta: float) -> float:
    return conformal_quantile(scores, delta=delta)


def compute_horizon_thresholds(scores: np.ndarray, horizons: list[int], delta: float) -> dict[int, float]:
    if scores.ndim != 3 or scores.shape[2] != len(horizons):
        raise AssertionError(
            f"scores must have shape (T, N, H) with H={len(horizons)}, got shape={scores.shape}"
        )
    return {
        int(horizon): conformal_quantile(scores[:, :, h_idx], delta=delta)
        for h_idx, horizon in enumerate(horizons)
    }


def apply_cqr_threshold(
    lower: np.ndarray,
    upper: np.ndarray,
    s_hat: float | np.ndarray,
    nonnegative: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    calibrated_lower = lower - s_hat
    calibrated_upper = upper + s_hat
    if nonnegative:
        calibrated_lower = np.maximum(calibrated_lower, 0.0)
        calibrated_upper = np.maximum(calibrated_upper, 0.0)
    return calibrated_lower, calibrated_upper


def add_interval_row(
    rows: list[dict[str, float | int | str]],
    method: str,
    horizon: int,
    delta: float,
    lower_quantile: float,
    upper_quantile: float,
    s_hat: float,
    labels: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    point_pred: np.ndarray,
) -> None:
    metrics = interval_metrics(labels, lower, upper, point_pred=point_pred, delta=delta)
    nominal = 1.0 - delta
    picp = float(metrics["PICP"])
    coverage_gap = picp - nominal
    rows.append({
        "method": method,
        "horizon": int(horizon),
        "delta": float(delta),
        "nominal_coverage": float(nominal),
        "lower_quantile": float(lower_quantile),
        "upper_quantile": float(upper_quantile),
        "s_hat": float(s_hat),
        "PICP": picp,
        "coverage_gap": float(coverage_gap),
        "ACE": float(abs(coverage_gap)),
        "undercoverage": float(max(nominal - picp, 0.0)),
        "MPIW": float(metrics["MPIW"]),
        "WIS": float(metrics.get("WIS", np.nan)),
    })


def evaluate_interval_methods(
    test_quantiles: np.ndarray,
    test_labels: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    deltas: list[float],
    global_thresholds: dict[float, float],
    horizon_thresholds: dict[float, dict[int, float]],
) -> pd.DataFrame:
    q50_idx = quantile_index(quantiles, 0.5)
    point_pred = test_quantiles[..., q50_idx]
    rows: list[dict[str, float | int | str]] = []

    for delta in deltas:
        lower_idx = quantile_index(quantiles, delta / 2.0)
        upper_idx = quantile_index(quantiles, 1.0 - delta / 2.0)
        lower_q = quantiles[lower_idx]
        upper_q = quantiles[upper_idx]

        for h_idx, horizon in enumerate(horizons):
            labels_h = test_labels[:, :, h_idx]
            point_h = point_pred[:, :, h_idx]
            raw_lower = test_quantiles[:, :, h_idx, lower_idx]
            raw_upper = test_quantiles[:, :, h_idx, upper_idx]

            add_interval_row(
                rows=rows,
                method="raw",
                horizon=horizon,
                delta=delta,
                lower_quantile=lower_q,
                upper_quantile=upper_q,
                s_hat=0.0,
                labels=labels_h,
                lower=raw_lower,
                upper=raw_upper,
                point_pred=point_h,
            )

            global_s = global_thresholds[delta]
            global_lower, global_upper = apply_cqr_threshold(raw_lower, raw_upper, global_s)
            add_interval_row(
                rows=rows,
                method="global_cqr",
                horizon=horizon,
                delta=delta,
                lower_quantile=lower_q,
                upper_quantile=upper_q,
                s_hat=global_s,
                labels=labels_h,
                lower=global_lower,
                upper=global_upper,
                point_pred=point_h,
            )

            horizon_s = horizon_thresholds[delta][int(horizon)]
            horizon_lower, horizon_upper = apply_cqr_threshold(raw_lower, raw_upper, horizon_s)
            add_interval_row(
                rows=rows,
                method="horizon_cqr",
                horizon=horizon,
                delta=delta,
                lower_quantile=lower_q,
                upper_quantile=upper_q,
                s_hat=horizon_s,
                labels=labels_h,
                lower=horizon_lower,
                upper=horizon_upper,
                point_pred=point_h,
            )

    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def build_threshold_table(
    deltas: list[float],
    global_thresholds: dict[float, float],
    horizon_thresholds: dict[float, dict[int, float]],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for delta in deltas:
        rows.append({
            "method": "global_cqr",
            "delta": float(delta),
            "horizon": "all",
            "s_hat": float(global_thresholds[delta]),
        })
        for horizon, s_hat in horizon_thresholds[delta].items():
            rows.append({
                "method": "horizon_cqr",
                "delta": float(delta),
                "horizon": int(horizon),
                "s_hat": float(s_hat),
            })
    return pd.DataFrame(rows)


def build_comparison_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    raw = metrics_df[metrics_df["method"] == "raw"].copy()
    calibrated = metrics_df[metrics_df["method"].isin(["global_cqr", "horizon_cqr"])].copy()
    comparison = calibrated.merge(
        raw,
        on=["horizon", "delta", "nominal_coverage", "lower_quantile", "upper_quantile"],
        suffixes=("", "_raw"),
    )
    comparison["PICP_gain_vs_raw"] = comparison["PICP"] - comparison["PICP_raw"]
    comparison["ACE_reduction_vs_raw"] = comparison["ACE_raw"] - comparison["ACE"]
    comparison["MPIW_change_vs_raw"] = comparison["MPIW"] - comparison["MPIW_raw"]
    comparison["MPIW_ratio_vs_raw"] = comparison["MPIW"] / comparison["MPIW_raw"]
    comparison["WIS_change_vs_raw"] = comparison["WIS"] - comparison["WIS_raw"]
    return comparison[
        [
            "method",
            "horizon",
            "delta",
            "nominal_coverage",
            "PICP_raw",
            "PICP",
            "PICP_gain_vs_raw",
            "ACE_raw",
            "ACE",
            "ACE_reduction_vs_raw",
            "MPIW_raw",
            "MPIW",
            "MPIW_change_vs_raw",
            "MPIW_ratio_vs_raw",
            "WIS_raw",
            "WIS",
            "WIS_change_vs_raw",
            "s_hat",
        ]
    ]


def build_summary_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics_df
        .groupby(["method", "delta", "nominal_coverage"], as_index=False)
        .agg(
            mean_PICP=("PICP", "mean"),
            min_PICP=("PICP", "min"),
            max_PICP=("PICP", "max"),
            mean_ACE=("ACE", "mean"),
            max_ACE=("ACE", "max"),
            mean_undercoverage=("undercoverage", "mean"),
            max_undercoverage=("undercoverage", "max"),
            mean_MPIW=("MPIW", "mean"),
            mean_WIS=("WIS", "mean"),
        )
    )
    summary["mean_coverage_gap"] = summary["mean_PICP"] - summary["nominal_coverage"]
    return summary


def load_stage1_metadata(stage1_output_dir: Path) -> dict[str, object]:
    metadata_path = stage1_output_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Stage 1 metadata not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def checkpoint_from_metadata(stage1_output_dir: Path, metadata: dict[str, object]) -> Path:
    model_name = str(metadata["model_name"])
    expected = stage1_output_dir / f"{model_name}_checkpoint.pt"
    if expected.exists():
        return expected

    matches = sorted(stage1_output_dir.glob("*_checkpoint.pt"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Could not identify Stage 1 checkpoint in {stage1_output_dir}; expected {expected}"
    )


def build_calib_test_loaders(
    input_series: np.ndarray,
    price_raw: np.ndarray,
    seq_len: int,
    horizons: list[int],
    batch_size: int,
    split_rates: dict[str, float],
    device: torch.device,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    _, _, calib_demand, test_demand = fn.division(
        input_series,
        train_rate=split_rates["train"],
        valid_rate=split_rates["valid"],
        calib_rate=split_rates["calib"],
        test_rate=split_rates["test"],
    )
    _, _, calib_price, test_price = fn.division(
        price_raw,
        train_rate=split_rates["train"],
        valid_rate=split_rates["valid"],
        calib_rate=split_rates["calib"],
        test_rate=split_rates["test"],
    )

    calib_dataset = fn.CreateMultiHorizonDataset(calib_demand, calib_price, seq_len, horizons, device)
    test_dataset = fn.CreateMultiHorizonDataset(test_demand, test_price, seq_len, horizons, device)
    loaders = (
        DataLoader(calib_dataset, batch_size=batch_size, shuffle=False, drop_last=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False),
    )
    lengths = {
        "calib": int(len(calib_demand)),
        "test": int(len(test_demand)),
        "calib_windows": int(len(calib_dataset)),
        "test_windows": int(len(test_dataset)),
    }
    return (*loaders, lengths)


def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cap: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    max_batches: int,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predict_list = []
    label_list = []
    with torch.no_grad():
        for batch_idx, (demand, price, label) in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            demand, price = demand.to(device), price.to(device)
            pred_q = model(demand, price)
            if pred_q.ndim == 3 and len(horizons) == 1:
                pred_q = pred_q.unsqueeze(2)
            assert_quantile_tensor_shape(
                pred_q,
                n_horizons=len(horizons),
                n_quantiles=len(quantiles),
                context=f"{context} predict_quantiles",
            )
            assert_target_tensor_shape(label, n_horizons=len(horizons), context=f"{context} labels")
            predict_list.append(pred_q.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

    if not predict_list:
        raise RuntimeError(f"No batches were evaluated for {context}.")

    predict_q_normalized = np.concatenate(predict_list, axis=0)
    label_normalized = np.concatenate(label_list, axis=0)
    predict_q = predict_q_normalized * cap.reshape(1, -1, 1, 1)
    labels = label_normalized * cap.reshape(1, -1, 1)
    assert_monotonic_quantiles(predict_q, context=f"{context} saved predict_quantiles")
    return predict_q, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 multi-horizon CQR calibration.")
    parser.add_argument("--stage1-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--use-cuda", type=parse_bool, default=True)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--deltas", default=",".join(str(delta) for delta in DEFAULT_DELTAS))
    parser.add_argument("--max-calib-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    args = parser.parse_args()

    stage1_output_dir = resolve_path(args.stage1_output_dir)
    output_dir = resolve_path(args.output_dir)
    metadata = load_stage1_metadata(stage1_output_dir)
    checkpoint_path = checkpoint_from_metadata(stage1_output_dir, metadata)

    horizons = [int(h) for h in metadata["horizons"]]
    quantiles = [float(q) for q in metadata["quantiles"]]
    deltas = parse_float_list(args.deltas)
    seq_len = int(metadata["seq_len"])
    batch_size = int(args.batch_size or metadata["batch_size"])
    split_rates = {key: float(value) for key, value in metadata["split_rates"].items()}
    model_name = str(metadata["model_name"])
    data_dir = resolve_path(str(metadata["data_dir"]))

    device = torch.device("cuda:0" if args.use_cuda and torch.cuda.is_available() else "cpu")
    occ, duration, price_raw, distance, cap = fn.read_dataset_v2(dataset_path_string(data_dir))
    input_series = duration if "dura" in model_name else occ

    calib_loader, test_loader, split_lengths = build_calib_test_loaders(
        input_series=input_series,
        price_raw=price_raw,
        seq_len=seq_len,
        horizons=horizons,
        batch_size=batch_size,
        split_rates=split_rates,
        device=device,
    )

    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model.to(device)
    calib_quantiles, calib_labels = predict_loader(
        model=model,
        loader=calib_loader,
        device=device,
        cap=cap,
        quantiles=quantiles,
        horizons=horizons,
        max_batches=args.max_calib_batches,
        context="stage2 calibration",
    )
    test_quantiles, test_labels = predict_loader(
        model=model,
        loader=test_loader,
        device=device,
        cap=cap,
        quantiles=quantiles,
        horizons=horizons,
        max_batches=args.max_test_batches,
        context="stage2 test",
    )

    global_thresholds: dict[float, float] = {}
    horizon_thresholds: dict[float, dict[int, float]] = {}
    for delta in deltas:
        scores = compute_cqr_scores(calib_quantiles, calib_labels, quantiles, delta)
        global_thresholds[delta] = compute_global_threshold(scores, delta)
        horizon_thresholds[delta] = compute_horizon_thresholds(scores, horizons, delta)

    metrics_df = evaluate_interval_methods(
        test_quantiles=test_quantiles,
        test_labels=test_labels,
        quantiles=quantiles,
        horizons=horizons,
        deltas=deltas,
        global_thresholds=global_thresholds,
        horizon_thresholds=horizon_thresholds,
    )
    thresholds_df = build_threshold_table(deltas, global_thresholds, horizon_thresholds)
    comparison_df = build_comparison_table(metrics_df)
    summary_df = build_summary_table(metrics_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "calibrated_interval_metrics_by_horizon.csv", index=False)
    thresholds_df.to_csv(output_dir / "cqr_thresholds.csv", index=False)
    comparison_df.to_csv(output_dir / "raw_vs_cqr_comparison_by_horizon.csv", index=False)
    summary_df.to_csv(output_dir / "global_summary.csv", index=False)

    stage2_metadata = {
        "stage": "2.1-2.2",
        "goal": "Global and horizon-wise CQR calibration for direct multi-horizon forecasts",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "branch": git_value(["branch", "--show-current"]),
        "commit": git_value(["rev-parse", "--short", "HEAD"]),
        "stage1_output_dir": str(stage1_output_dir),
        "stage1_checkpoint": str(checkpoint_path),
        "stage1_model_name": model_name,
        "output_dir": str(output_dir),
        "data_dir": str(data_dir),
        "seq_len": seq_len,
        "horizons": horizons,
        "quantiles": quantiles,
        "deltas": deltas,
        "batch_size": batch_size,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "split_rates": split_rates,
        "split_lengths": split_lengths,
        "prediction_shapes": {
            "calib_quantiles": list(calib_quantiles.shape),
            "calib_labels": list(calib_labels.shape),
            "test_quantiles": list(test_quantiles.shape),
            "test_labels": list(test_labels.shape),
        },
        "smoke_limits": {
            "max_calib_batches": args.max_calib_batches,
            "max_test_batches": args.max_test_batches,
        },
        "output_files": {
            "metrics": "calibrated_interval_metrics_by_horizon.csv",
            "thresholds": "cqr_thresholds.csv",
            "comparison": "raw_vs_cqr_comparison_by_horizon.csv",
            "summary": "global_summary.csv",
        },
    }
    (output_dir / "stage2_metadata.json").write_text(
        json.dumps(stage2_metadata, indent=2),
        encoding="utf-8",
    )

    print(summary_df.to_string(index=False))
    print(json.dumps(stage2_metadata, indent=2))


if __name__ == "__main__":
    main()
