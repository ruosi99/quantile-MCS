from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.journal.calibrate_multihorizon_cqr import apply_cqr_threshold, resolve_path
from scripts.journal.train_multihorizon_raw import parse_float_list, quantile_index
from utils.model_training.conformal import interval_metrics
from utils.model_training.journal_contracts import assert_quantile_tensor_shape, assert_target_tensor_shape


METHODS = ["raw", "global_cqr", "horizon_cqr"]
DEFAULT_DELTAS = [0.1, 0.2, 0.4]


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


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def metric_values(
    labels: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    point: np.ndarray,
    mask: np.ndarray,
    delta: float,
) -> dict[str, float | int]:
    if mask.shape != labels.shape:
        raise AssertionError(f"mask shape {mask.shape} does not match labels shape {labels.shape}")

    count = int(mask.sum())
    if count == 0:
        return {
            "count": 0,
            "PICP": float("nan"),
            "coverage_gap": float("nan"),
            "ACE": float("nan"),
            "undercoverage": float("nan"),
            "MPIW": float("nan"),
            "WIS": float("nan"),
        }

    metrics = interval_metrics(
        labels[mask],
        lower[mask],
        upper[mask],
        point_pred=point[mask],
        delta=delta,
    )
    nominal = 1.0 - delta
    picp = float(metrics["PICP"])
    coverage_gap = picp - nominal
    return {
        "count": count,
        "PICP": picp,
        "coverage_gap": float(coverage_gap),
        "ACE": float(abs(coverage_gap)),
        "undercoverage": float(max(nominal - picp, 0.0)),
        "MPIW": float(metrics["MPIW"]),
        "WIS": float(metrics.get("WIS", np.nan)),
    }


def threshold_lookup(thresholds_df: pd.DataFrame, method: str, delta: float, horizon: int) -> float:
    if method == "raw":
        return 0.0
    if method == "global_cqr":
        rows = thresholds_df[
            (thresholds_df["method"] == "global_cqr")
            & np.isclose(thresholds_df["delta"].astype(float), delta)
        ]
    elif method == "horizon_cqr":
        rows = thresholds_df[
            (thresholds_df["method"] == "horizon_cqr")
            & np.isclose(thresholds_df["delta"].astype(float), delta)
            & (thresholds_df["horizon"].astype(str) == str(horizon))
        ]
    else:
        raise ValueError(f"Unsupported method: {method}")

    if len(rows) != 1:
        raise ValueError(f"Expected one threshold row for method={method}, delta={delta}, horizon={horizon}; got {len(rows)}")
    return float(rows.iloc[0]["s_hat"])


def method_interval(
    raw_lower: np.ndarray,
    raw_upper: np.ndarray,
    s_hat: float,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "raw":
        return raw_lower, raw_upper
    return apply_cqr_threshold(raw_lower, raw_upper, s_hat, nonnegative=True)


def interval_bundle(
    predict_quantiles: np.ndarray,
    labels: np.ndarray,
    thresholds_df: pd.DataFrame,
    quantiles: list[float],
    horizon: int,
    horizon_idx: int,
    delta: float,
    method: str,
) -> dict[str, np.ndarray | float]:
    lower_idx = quantile_index(quantiles, delta / 2.0)
    upper_idx = quantile_index(quantiles, 1.0 - delta / 2.0)
    point_idx = quantile_index(quantiles, 0.5)

    raw_lower = np.asarray(predict_quantiles[:, :, horizon_idx, lower_idx])
    raw_upper = np.asarray(predict_quantiles[:, :, horizon_idx, upper_idx])
    point = np.asarray(predict_quantiles[:, :, horizon_idx, point_idx])
    y = np.asarray(labels[:, :, horizon_idx])
    s_hat = threshold_lookup(thresholds_df, method, delta, horizon)
    lower, upper = method_interval(raw_lower, raw_upper, s_hat, method)
    return {
        "labels": y,
        "raw_lower": raw_lower,
        "raw_upper": raw_upper,
        "lower": lower,
        "upper": upper,
        "point": point,
        "s_hat": s_hat,
    }


def build_zero_positive_summary(
    predict_quantiles: np.ndarray,
    labels: np.ndarray,
    thresholds_df: pd.DataFrame,
    quantiles: list[float],
    horizons: list[int],
    deltas: list[float],
    positive_eps: float,
) -> pd.DataFrame:
    rows = []
    for delta in deltas:
        nominal = 1.0 - delta
        for h_idx, horizon in enumerate(horizons):
            for method in METHODS:
                bundle = interval_bundle(
                    predict_quantiles, labels, thresholds_df, quantiles, horizon, h_idx, delta, method
                )
                y = bundle["labels"]
                lower = bundle["lower"]
                upper = bundle["upper"]
                point = bundle["point"]

                all_metrics = metric_values(y, lower, upper, point, np.ones_like(y, dtype=bool), delta)
                zero_metrics = metric_values(y, lower, upper, point, y <= positive_eps, delta)
                pos_metrics = metric_values(y, lower, upper, point, y > positive_eps, delta)
                rows.append({
                    "method": method,
                    "delta": float(delta),
                    "nominal_coverage": float(nominal),
                    "horizon": int(horizon),
                    "s_hat": float(bundle["s_hat"]),
                    "count_all": all_metrics["count"],
                    "PICP_all": all_metrics["PICP"],
                    "ACE_all": all_metrics["ACE"],
                    "MPIW_all": all_metrics["MPIW"],
                    "WIS_all": all_metrics["WIS"],
                    "count_zero": zero_metrics["count"],
                    "PICP_zero": zero_metrics["PICP"],
                    "ACE_zero": zero_metrics["ACE"],
                    "MPIW_zero": zero_metrics["MPIW"],
                    "WIS_zero": zero_metrics["WIS"],
                    "count_positive": pos_metrics["count"],
                    "PICP_positive": pos_metrics["PICP"],
                    "ACE_positive": pos_metrics["ACE"],
                    "undercoverage_positive": pos_metrics["undercoverage"],
                    "MPIW_positive": pos_metrics["MPIW"],
                    "WIS_positive": pos_metrics["WIS"],
                })
    return pd.DataFrame(rows)


def build_boundary_rescue_summary(
    predict_quantiles: np.ndarray,
    labels: np.ndarray,
    thresholds_df: pd.DataFrame,
    quantiles: list[float],
    horizons: list[int],
    deltas: list[float],
    positive_eps: float,
) -> pd.DataFrame:
    rows = []
    for delta in deltas:
        for h_idx, horizon in enumerate(horizons):
            raw_bundle = interval_bundle(
                predict_quantiles, labels, thresholds_df, quantiles, horizon, h_idx, delta, "raw"
            )
            y = raw_bundle["labels"]
            raw_lower = raw_bundle["raw_lower"]
            raw_upper = raw_bundle["raw_upper"]
            raw_covered = (y >= raw_lower) & (y <= raw_upper)
            raw_lower_miss = y < raw_lower
            raw_upper_miss = y > raw_upper
            total = int(y.size)

            for method in ["global_cqr", "horizon_cqr"]:
                bundle = interval_bundle(
                    predict_quantiles, labels, thresholds_df, quantiles, horizon, h_idx, delta, method
                )
                lower = bundle["lower"]
                upper = bundle["upper"]
                s_hat = float(bundle["s_hat"])
                calibrated_covered = (y >= lower) & (y <= upper)
                rescued = (~raw_covered) & calibrated_covered
                zero_rescued = rescued & (y <= positive_eps)
                positive_rescued = rescued & (y > positive_eps)
                lower_rescued = rescued & raw_lower_miss
                upper_rescued = rescued & raw_upper_miss
                clipping_rescued = (
                    rescued
                    & (y <= positive_eps)
                    & (raw_lower > 0.0)
                    & ((raw_lower - s_hat) <= 0.0)
                    & (lower == 0.0)
                )
                near_boundary_lower_miss = (y <= positive_eps) & raw_lower_miss

                rows.append({
                    "method": method,
                    "delta": float(delta),
                    "horizon": int(horizon),
                    "s_hat": s_hat,
                    "total_count": total,
                    "raw_miss_count": int((~raw_covered).sum()),
                    "raw_lower_miss_count": int(raw_lower_miss.sum()),
                    "raw_upper_miss_count": int(raw_upper_miss.sum()),
                    "zero_raw_lower_miss_count": int(near_boundary_lower_miss.sum()),
                    "rescued_count": int(rescued.sum()),
                    "zero_rescued_count": int(zero_rescued.sum()),
                    "positive_rescued_count": int(positive_rescued.sum()),
                    "lower_rescued_count": int(lower_rescued.sum()),
                    "upper_rescued_count": int(upper_rescued.sum()),
                    "clipping_rescued_count": int(clipping_rescued.sum()),
                    "raw_miss_rate": float((~raw_covered).mean()),
                    "rescued_share_all": float(rescued.mean()),
                    "rescued_share_of_raw_misses": float(rescued.sum() / max((~raw_covered).sum(), 1)),
                    "zero_share_of_rescues": float(zero_rescued.sum() / max(rescued.sum(), 1)),
                    "clipping_share_of_rescues": float(clipping_rescued.sum() / max(rescued.sum(), 1)),
                })
    return pd.DataFrame(rows)


def compute_target_hours(
    data_dir: Path,
    split_rates: dict[str, float],
    seq_len: int,
    horizons: list[int],
    n_windows: int,
) -> dict[int, np.ndarray]:
    duration_path = data_dir / "duration.csv"
    if not duration_path.exists():
        raise FileNotFoundError(f"Duration CSV not found for target-hour diagnostics: {duration_path}")
    time_index = pd.to_datetime(pd.read_csv(duration_path, usecols=["time"])["time"])
    total_len = len(time_index)
    test_start = int(total_len * (split_rates["train"] + split_rates["valid"] + split_rates["calib"]))
    target_hours = {}
    for horizon in horizons:
        target_indices = test_start + np.arange(n_windows) + seq_len + int(horizon) - 1
        if target_indices.max() >= total_len:
            raise IndexError(
                f"Target-hour index exceeds dataset length: max={target_indices.max()}, total={total_len}"
            )
        target_hours[int(horizon)] = time_index.iloc[target_indices].dt.hour.to_numpy()
    return target_hours


def build_hour_horizon_summary(
    predict_quantiles: np.ndarray,
    labels: np.ndarray,
    thresholds_df: pd.DataFrame,
    quantiles: list[float],
    horizons: list[int],
    deltas: list[float],
    target_hours: dict[int, np.ndarray],
    positive_eps: float,
    positive_only: bool,
) -> pd.DataFrame:
    rows = []
    for delta in deltas:
        nominal = 1.0 - delta
        for h_idx, horizon in enumerate(horizons):
            hours = target_hours[int(horizon)]
            for method in METHODS:
                bundle = interval_bundle(
                    predict_quantiles, labels, thresholds_df, quantiles, horizon, h_idx, delta, method
                )
                y = bundle["labels"]
                lower = bundle["lower"]
                upper = bundle["upper"]
                point = bundle["point"]
                base_mask = y > positive_eps if positive_only else np.ones_like(y, dtype=bool)
                for hour in range(24):
                    time_mask = (hours == hour).reshape(-1, 1)
                    mask = base_mask & time_mask
                    metrics = metric_values(y, lower, upper, point, mask, delta)
                    rows.append({
                        "method": method,
                        "delta": float(delta),
                        "nominal_coverage": float(nominal),
                        "horizon": int(horizon),
                        "hour": int(hour),
                        "positive_only": bool(positive_only),
                        "sample_steps": int((hours == hour).sum()),
                        **metrics,
                    })
    return pd.DataFrame(rows)


def demand_bin_masks(y: np.ndarray, positive_eps: float) -> tuple[list[tuple[str, np.ndarray]], dict[str, float]]:
    positive_values = y[y > positive_eps]
    if positive_values.size == 0:
        q50 = float("nan")
        q90 = float("nan")
        return [("zero", y <= positive_eps)], {"positive_q50": q50, "positive_q90": q90}

    q50 = float(np.quantile(positive_values, 0.5))
    q90 = float(np.quantile(positive_values, 0.9))
    masks = [
        ("zero", y <= positive_eps),
        ("positive_le_q50", (y > positive_eps) & (y <= q50)),
        ("positive_q50_q90", (y > q50) & (y <= q90)),
        ("positive_gt_q90", y > q90),
    ]
    return masks, {"positive_q50": q50, "positive_q90": q90}


def build_demand_bin_summary(
    predict_quantiles: np.ndarray,
    labels: np.ndarray,
    thresholds_df: pd.DataFrame,
    quantiles: list[float],
    horizons: list[int],
    deltas: list[float],
    positive_eps: float,
) -> pd.DataFrame:
    rows = []
    for delta in deltas:
        nominal = 1.0 - delta
        for h_idx, horizon in enumerate(horizons):
            y_h = np.asarray(labels[:, :, h_idx])
            masks, thresholds = demand_bin_masks(y_h, positive_eps)
            for method in METHODS:
                bundle = interval_bundle(
                    predict_quantiles, labels, thresholds_df, quantiles, horizon, h_idx, delta, method
                )
                y = bundle["labels"]
                lower = bundle["lower"]
                upper = bundle["upper"]
                point = bundle["point"]
                for bin_name, mask in masks:
                    metrics = metric_values(y, lower, upper, point, mask, delta)
                    rows.append({
                        "method": method,
                        "delta": float(delta),
                        "nominal_coverage": float(nominal),
                        "horizon": int(horizon),
                        "demand_bin": bin_name,
                        **thresholds,
                        **metrics,
                    })
    return pd.DataFrame(rows)


def build_reliability_summary(
    zero_positive_df: pd.DataFrame,
    hour_all_df: pd.DataFrame,
    hour_positive_df: pd.DataFrame,
) -> pd.DataFrame:
    base_rows = []
    for row in zero_positive_df.itertuples(index=False):
        base_rows.append({
            "method": row.method,
            "delta": float(row.delta),
            "nominal_coverage": float(row.nominal_coverage),
            "horizon": int(row.horizon),
            "PICP_all": float(row.PICP_all),
            "ACE_all": float(row.ACE_all),
            "PICP_positive": float(row.PICP_positive),
            "ACE_positive": float(row.ACE_positive),
            "undercoverage_positive": float(row.undercoverage_positive),
            "MPIW_all": float(row.MPIW_all),
            "MPIW_positive": float(row.MPIW_positive),
        })
    reliability = pd.DataFrame(base_rows)

    hour_cols = ["method", "delta", "horizon"]
    hour_all = (
        hour_all_df
        .groupby(hour_cols, as_index=False)
        .agg(
            max_hour_ACE_all=("ACE", "max"),
            max_hour_undercoverage_all=("undercoverage", "max"),
            min_hour_PICP_all=("PICP", "min"),
            max_hour_PICP_all=("PICP", "max"),
        )
    )
    hour_pos = (
        hour_positive_df
        .groupby(hour_cols, as_index=False)
        .agg(
            max_hour_ACE_positive=("ACE", "max"),
            max_hour_undercoverage_positive=("undercoverage", "max"),
            min_hour_PICP_positive=("PICP", "min"),
            max_hour_PICP_positive=("PICP", "max"),
        )
    )
    reliability = reliability.merge(hour_all, on=hour_cols, how="left")
    reliability = reliability.merge(hour_pos, on=hour_cols, how="left")
    return reliability


def scalar_or_none(value: float) -> float | None:
    if value is None or np.isnan(value):
        return None
    return float(value)


def build_gate_summary(
    zero_positive_df: pd.DataFrame,
    hour_all_df: pd.DataFrame,
    hour_positive_df: pd.DataFrame,
    demand_bin_df: pd.DataFrame,
    boundary_df: pd.DataFrame,
    method_for_gate: str,
    delta_for_gate: float,
) -> dict[str, object]:
    zp = zero_positive_df[
        (zero_positive_df["method"] == method_for_gate)
        & np.isclose(zero_positive_df["delta"], delta_for_gate)
    ]
    hour_all = hour_all_df[
        (hour_all_df["method"] == method_for_gate)
        & np.isclose(hour_all_df["delta"], delta_for_gate)
    ]
    hour_pos = hour_positive_df[
        (hour_positive_df["method"] == method_for_gate)
        & np.isclose(hour_positive_df["delta"], delta_for_gate)
    ]
    demand_tail = demand_bin_df[
        (demand_bin_df["method"] == method_for_gate)
        & np.isclose(demand_bin_df["delta"], delta_for_gate)
        & (demand_bin_df["demand_bin"] == "positive_gt_q90")
    ]
    boundary = boundary_df[
        (boundary_df["method"] == method_for_gate)
        & np.isclose(boundary_df["delta"], delta_for_gate)
    ]

    min_positive_picp = float(zp["PICP_positive"].min())
    worst_positive_horizon = zp.sort_values("PICP_positive").iloc[0]
    max_hour_ace_all = float(hour_all["ACE"].max())
    worst_hour_all = hour_all.sort_values("ACE", ascending=False).iloc[0]
    max_hour_ace_positive = float(hour_pos["ACE"].max())
    worst_hour_positive = hour_pos.sort_values("ACE", ascending=False).iloc[0]
    min_tail_picp = float(demand_tail["PICP"].min())
    worst_tail = demand_tail.sort_values("PICP").iloc[0]
    zero_share_of_rescues = float(boundary["zero_share_of_rescues"].mean())
    clipping_share_of_rescues = float(boundary["clipping_share_of_rescues"].mean())

    criteria = {
        "positive_picp_below_0_85": bool(min_positive_picp < 0.85),
        "worst_cell_ace_above_0_08": bool(max(max_hour_ace_all, max_hour_ace_positive) > 0.08),
        "tail_picp_below_0_85": bool(min_tail_picp < 0.85),
        "decision_regret_not_evaluated": True,
    }
    triggered = [name for name, active in criteria.items() if active and name != "decision_regret_not_evaluated"]

    if triggered:
        recommendation = (
            "Stage 3 remains justified as a conditional localized-reliability experiment, "
            "especially for positive-demand, tail-demand, or hard hour-by-horizon cells. "
            "Do not frame it as necessary for marginal 90% coverage."
        )
    else:
        recommendation = (
            "Do not treat Stage 3 as a main contribution by default. Pivot toward direct "
            "multi-horizon forecasting, boundary-aware diagnostics, and decision-oriented calibration."
        )

    return {
        "gate_version": "stage2-diagnostic-gate-2026-05-05",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "gate_method": method_for_gate,
        "gate_delta": float(delta_for_gate),
        "nominal_coverage": float(1.0 - delta_for_gate),
        "criteria": criteria,
        "triggered_criteria": triggered,
        "min_positive_picp": min_positive_picp,
        "worst_positive_horizon": {
            "horizon": int(worst_positive_horizon["horizon"]),
            "PICP_positive": float(worst_positive_horizon["PICP_positive"]),
            "ACE_positive": float(worst_positive_horizon["ACE_positive"]),
            "undercoverage_positive": float(worst_positive_horizon["undercoverage_positive"]),
        },
        "max_hour_ace_all": max_hour_ace_all,
        "worst_hour_all": {
            "horizon": int(worst_hour_all["horizon"]),
            "hour": int(worst_hour_all["hour"]),
            "PICP": float(worst_hour_all["PICP"]),
            "ACE": float(worst_hour_all["ACE"]),
            "undercoverage": float(worst_hour_all["undercoverage"]),
        },
        "max_hour_ace_positive": max_hour_ace_positive,
        "worst_hour_positive": {
            "horizon": int(worst_hour_positive["horizon"]),
            "hour": int(worst_hour_positive["hour"]),
            "PICP": scalar_or_none(float(worst_hour_positive["PICP"])),
            "ACE": scalar_or_none(float(worst_hour_positive["ACE"])),
            "undercoverage": scalar_or_none(float(worst_hour_positive["undercoverage"])),
        },
        "min_tail_picp": min_tail_picp,
        "worst_tail_bin": {
            "horizon": int(worst_tail["horizon"]),
            "PICP": float(worst_tail["PICP"]),
            "ACE": float(worst_tail["ACE"]),
            "undercoverage": float(worst_tail["undercoverage"]),
        },
        "mean_zero_share_of_rescues": zero_share_of_rescues,
        "mean_clipping_share_of_rescues": clipping_share_of_rescues,
        "recommendation": recommendation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2.5 diagnostic gate for journal multi-horizon CQR.")
    parser.add_argument("--stage1-output-dir", default="journal_results/shenzhen_multihorizon/warmstart_raw")
    parser.add_argument("--stage2-output-dir", default="journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw")
    parser.add_argument("--output-dir", default="journal_results/shenzhen_multihorizon/stage2_diagnostics/warmstart_raw")
    parser.add_argument("--deltas", default=",".join(str(delta) for delta in DEFAULT_DELTAS))
    parser.add_argument("--positive-eps", type=float, default=1e-12)
    parser.add_argument("--gate-method", default="global_cqr", choices=METHODS)
    parser.add_argument("--gate-delta", type=float, default=0.1)
    args = parser.parse_args()

    stage1_dir = resolve_path(args.stage1_output_dir)
    stage2_dir = resolve_path(args.stage2_output_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage1_metadata = load_json(stage1_dir / "run_metadata.json")
    stage2_metadata = load_json(stage2_dir / "stage2_metadata.json")
    thresholds_df = pd.read_csv(stage2_dir / "cqr_thresholds.csv")

    predict_quantiles = np.load(stage1_dir / "predict_quantiles.npy", mmap_mode="r")
    labels = np.load(stage1_dir / "label_list.npy", mmap_mode="r")
    horizons = [int(h) for h in stage2_metadata["horizons"]]
    quantiles = [float(q) for q in stage2_metadata["quantiles"]]
    deltas = parse_float_list(args.deltas)
    seq_len = int(stage2_metadata["seq_len"])
    split_rates = {key: float(value) for key, value in stage2_metadata["split_rates"].items()}
    data_dir = resolve_path(str(stage2_metadata["data_dir"]))

    assert_quantile_tensor_shape(
        predict_quantiles,
        n_horizons=len(horizons),
        n_quantiles=len(quantiles),
        context="diagnostic predict_quantiles",
    )
    assert_target_tensor_shape(labels, n_horizons=len(horizons), context="diagnostic labels")

    target_hours = compute_target_hours(
        data_dir=data_dir,
        split_rates=split_rates,
        seq_len=seq_len,
        horizons=horizons,
        n_windows=int(labels.shape[0]),
    )

    zero_positive_df = build_zero_positive_summary(
        predict_quantiles=predict_quantiles,
        labels=labels,
        thresholds_df=thresholds_df,
        quantiles=quantiles,
        horizons=horizons,
        deltas=deltas,
        positive_eps=args.positive_eps,
    )
    boundary_df = build_boundary_rescue_summary(
        predict_quantiles=predict_quantiles,
        labels=labels,
        thresholds_df=thresholds_df,
        quantiles=quantiles,
        horizons=horizons,
        deltas=deltas,
        positive_eps=args.positive_eps,
    )
    hour_all_df = build_hour_horizon_summary(
        predict_quantiles=predict_quantiles,
        labels=labels,
        thresholds_df=thresholds_df,
        quantiles=quantiles,
        horizons=horizons,
        deltas=deltas,
        target_hours=target_hours,
        positive_eps=args.positive_eps,
        positive_only=False,
    )
    hour_positive_df = build_hour_horizon_summary(
        predict_quantiles=predict_quantiles,
        labels=labels,
        thresholds_df=thresholds_df,
        quantiles=quantiles,
        horizons=horizons,
        deltas=deltas,
        target_hours=target_hours,
        positive_eps=args.positive_eps,
        positive_only=True,
    )
    demand_bin_df = build_demand_bin_summary(
        predict_quantiles=predict_quantiles,
        labels=labels,
        thresholds_df=thresholds_df,
        quantiles=quantiles,
        horizons=horizons,
        deltas=deltas,
        positive_eps=args.positive_eps,
    )
    reliability_df = build_reliability_summary(zero_positive_df, hour_all_df, hour_positive_df)
    gate_summary = build_gate_summary(
        zero_positive_df=zero_positive_df,
        hour_all_df=hour_all_df,
        hour_positive_df=hour_positive_df,
        demand_bin_df=demand_bin_df,
        boundary_df=boundary_df,
        method_for_gate=args.gate_method,
        delta_for_gate=args.gate_delta,
    )
    gate_summary.update({
        "branch": git_value(["branch", "--show-current"]),
        "commit": git_value(["rev-parse", "--short", "HEAD"]),
        "stage1_output_dir": str(stage1_dir),
        "stage2_output_dir": str(stage2_dir),
        "output_dir": str(output_dir),
        "stage1_model_name": stage1_metadata.get("model_name", ""),
        "input_shapes": {
            "predict_quantiles": list(predict_quantiles.shape),
            "labels": list(labels.shape),
        },
        "output_files": {
            "zero_vs_positive": "zero_vs_positive_summary.csv",
            "boundary_rescue": "boundary_rescue_summary.csv",
            "hour_horizon_all": "hour_horizon_coverage_all.csv",
            "hour_horizon_positive": "hour_horizon_coverage_positive_only.csv",
            "demand_bin": "demand_bin_coverage_by_horizon.csv",
            "reliability": "reliability_by_horizon.csv",
            "gate_summary": "diagnostic_gate_summary.json",
        },
    })

    zero_positive_df.to_csv(output_dir / "zero_vs_positive_summary.csv", index=False)
    boundary_df.to_csv(output_dir / "boundary_rescue_summary.csv", index=False)
    hour_all_df.to_csv(output_dir / "hour_horizon_coverage_all.csv", index=False)
    hour_positive_df.to_csv(output_dir / "hour_horizon_coverage_positive_only.csv", index=False)
    demand_bin_df.to_csv(output_dir / "demand_bin_coverage_by_horizon.csv", index=False)
    reliability_df.to_csv(output_dir / "reliability_by_horizon.csv", index=False)
    (output_dir / "diagnostic_gate_summary.json").write_text(
        json.dumps(gate_summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(gate_summary, indent=2))


if __name__ == "__main__":
    main()
