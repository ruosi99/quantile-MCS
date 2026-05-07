from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.journal.build_paper_evidence_assets import (
    bootstrap_mean_ci,
    bootstrap_ratio_ci,
    interval_for_method,
    load_decision_threshold_lookup,
    load_json,
    metrics_for_mask,
    subset_masks,
    threshold_value,
)
from scripts.journal.calibrate_multihorizon_cqr import resolve_path
from scripts.journal.evaluate_stage4_decision import newsvendor_cost, parse_cost_ratios
from scripts.journal.train_multihorizon_raw import parse_float_list, quantile_index
from utils.model_training.journal_contracts import (
    assert_monotonic_quantiles,
    assert_quantile_tensor_shape,
    assert_target_tensor_shape,
)


INTERVAL_METHODS = ["raw", "global_cqr", "horizon_cqr"]
DEPLOYMENT_STRATEGIES = [
    "median_decision",
    "symmetric_interval_upper_bound",
    "raw_cost_aligned_quantile",
    "one_sided_refined_cost_aligned_quantile",
]


@dataclass(frozen=True)
class TimeManifest:
    frame: pd.DataFrame
    date_keys: np.ndarray
    target_hours: np.ndarray
    bootstrap_level: str


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def station_zero_groups(
    labels: np.ndarray,
    station_ids: list[str] | None = None,
    n_groups: int = 4,
    positive_eps: float = 1e-12,
) -> pd.DataFrame:
    if labels.ndim != 3:
        raise AssertionError(f"labels must have shape (T, N, H), got {labels.shape}")
    station_count = labels.shape[1]
    if station_ids is None:
        station_ids = [str(idx) for idx in range(station_count)]
    if len(station_ids) != station_count:
        raise AssertionError(f"station_ids length {len(station_ids)} != station count {station_count}")

    zero_ratio = np.mean(labels <= positive_eps, axis=(0, 2))
    group_count = min(int(n_groups), station_count)
    ranks = pd.Series(zero_ratio).rank(method="first")
    group_idx = pd.qcut(ranks, q=group_count, labels=False).astype(int).to_numpy()
    rows = []
    for station_idx, ratio in enumerate(zero_ratio):
        group = int(group_idx[station_idx])
        if group == 0:
            name = "G1_low_zero"
        elif group == group_count - 1:
            name = f"G{group + 1}_high_zero"
        else:
            name = f"G{group + 1}_mid_zero"
        rows.append(
            {
                "station_index": station_idx,
                "station_id": str(station_ids[station_idx]),
                "zero_ratio": float(ratio),
                "zero_group_index": group,
                "zero_group": name,
            }
        )
    return pd.DataFrame(rows)


def group_summary(groups_df: pd.DataFrame) -> pd.DataFrame:
    return (
        groups_df.groupby(["zero_group_index", "zero_group"], as_index=False)
        .agg(
            station_count=("station_index", "count"),
            min_zero_ratio=("zero_ratio", "min"),
            mean_zero_ratio=("zero_ratio", "mean"),
            max_zero_ratio=("zero_ratio", "max"),
        )
        .sort_values("zero_group_index")
    )


def mask_station_group(shape: tuple[int, int], station_indices: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[:, station_indices] = True
    return mask


def build_station_zero_inflation_sensitivity(
    predict_quantiles: np.ndarray,
    labels: np.ndarray,
    thresholds_df: pd.DataFrame,
    quantiles: list[float],
    horizons: list[int],
    groups_df: pd.DataFrame,
    deltas: list[float],
) -> pd.DataFrame:
    q50_idx = quantile_index(quantiles, 0.5)
    rows: list[dict[str, float | int | str]] = []

    for delta in deltas:
        lower_idx = quantile_index(quantiles, delta / 2.0)
        upper_idx = quantile_index(quantiles, 1.0 - delta / 2.0)
        for group_idx, group in groups_df.groupby("zero_group_index"):
            station_indices = group["station_index"].to_numpy(dtype=int)
            for h_idx, horizon in enumerate(horizons):
                labels_h = labels[:, station_indices, h_idx]
                point_h = predict_quantiles[:, station_indices, h_idx, q50_idx]
                raw_lower = predict_quantiles[:, station_indices, h_idx, lower_idx]
                raw_upper = predict_quantiles[:, station_indices, h_idx, upper_idx]
                raw_cover = (raw_lower <= labels_h) & (labels_h <= raw_upper)
                masks, demand_thresholds = subset_masks(labels_h)

                for method in INTERVAL_METHODS:
                    s_hat = threshold_value(thresholds_df, method=method, delta=delta, horizon=int(horizon))
                    lower, upper = interval_for_method(raw_lower, raw_upper, method=method, s_hat=s_hat)
                    cover = (lower <= labels_h) & (labels_h <= upper)
                    row: dict[str, float | int | str] = {
                        "zero_group_index": int(group_idx),
                        "zero_group": str(group.iloc[0]["zero_group"]),
                        "station_count": int(len(station_indices)),
                        "mean_station_zero_ratio": float(group["zero_ratio"].mean()),
                        "method": method,
                        "delta": float(delta),
                        "nominal_coverage": float(1.0 - delta),
                        "horizon": int(horizon),
                        "s_hat": float(s_hat),
                        **demand_thresholds,
                    }
                    for subset_name, mask in masks:
                        values = metrics_for_mask(labels_h, lower, upper, point_h, mask=mask, delta=delta)
                        for key, value in values.items():
                            row[f"{subset_name}_{key}"] = value

                    rescued = (~raw_cover) & cover
                    zero_rescued = rescued & (labels_h <= 1e-12)
                    positive_rescued = rescued & (labels_h > 1e-12)
                    rescued_count = int(rescued.sum())
                    total_count = int(labels_h.size)
                    row["rescued_count"] = rescued_count
                    row["zero_rescued_count"] = int(zero_rescued.sum())
                    row["positive_rescued_count"] = int(positive_rescued.sum())
                    row["ZBR_share"] = float(zero_rescued.sum() / rescued_count) if rescued_count else float("nan")
                    row["zero_boundary_rescue_gain"] = float(zero_rescued.sum() / total_count)
                    row["positive_rescue_gain"] = float(positive_rescued.sum() / total_count)
                    row["PICP_zero_minus_positive"] = float(row["zero_PICP"] - row["positive_PICP"])
                    rows.append(row)

    return pd.DataFrame(rows)


def build_station_level_rescue_scatter(
    predict_quantiles: np.ndarray,
    labels: np.ndarray,
    thresholds_df: pd.DataFrame,
    quantiles: list[float],
    horizons: list[int],
    groups_df: pd.DataFrame,
    delta: float,
) -> pd.DataFrame:
    lower_idx = quantile_index(quantiles, delta / 2.0)
    upper_idx = quantile_index(quantiles, 1.0 - delta / 2.0)
    rows = []
    for station_idx in range(labels.shape[1]):
        rescued_total = 0
        zero_rescued_total = 0
        positive_rescued_total = 0
        raw_miss_total = 0
        for h_idx, horizon in enumerate(horizons):
            labels_h = labels[:, station_idx, h_idx]
            raw_lower = predict_quantiles[:, station_idx, h_idx, lower_idx]
            raw_upper = predict_quantiles[:, station_idx, h_idx, upper_idx]
            raw_cover = (raw_lower <= labels_h) & (labels_h <= raw_upper)
            s_hat = threshold_value(thresholds_df, method="global_cqr", delta=delta, horizon=int(horizon))
            lower, upper = interval_for_method(raw_lower, raw_upper, method="global_cqr", s_hat=s_hat)
            cover = (lower <= labels_h) & (labels_h <= upper)
            rescued = (~raw_cover) & cover
            raw_miss_total += int((~raw_cover).sum())
            rescued_total += int(rescued.sum())
            zero_rescued_total += int((rescued & (labels_h <= 1e-12)).sum())
            positive_rescued_total += int((rescued & (labels_h > 1e-12)).sum())

        station_row = groups_df[groups_df["station_index"] == station_idx].iloc[0]
        rows.append(
            {
                "station_index": station_idx,
                "station_id": station_row["station_id"],
                "zero_ratio": float(station_row["zero_ratio"]),
                "zero_group": station_row["zero_group"],
                "raw_miss_count": raw_miss_total,
                "rescued_count": rescued_total,
                "zero_rescued_count": zero_rescued_total,
                "positive_rescued_count": positive_rescued_total,
                "ZBR_share": float(zero_rescued_total / rescued_total) if rescued_total else float("nan"),
                "rescued_share_of_raw_misses": float(rescued_total / raw_miss_total) if raw_miss_total else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def decision_threshold_info(decision_thresholds_df: pd.DataFrame) -> dict[str, dict[str, object]]:
    return load_decision_threshold_lookup(decision_thresholds_df)


def decision_arrays_for_cost_ratio(
    predict_quantiles: np.ndarray,
    labels: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    thresholds_df: pd.DataFrame,
    decision_lookup: dict[str, dict[str, object]],
    cost_ratio: str,
    c_u: float,
    c_o: float,
    delta: float,
) -> dict[str, np.ndarray]:
    q50_idx = quantile_index(quantiles, 0.5)
    q95_idx = quantile_index(quantiles, 1.0 - delta / 2.0)
    target_q = float(decision_lookup[cost_ratio]["target_quantile"])
    target_idx = quantile_index(quantiles, target_q)

    median = np.maximum(predict_quantiles[..., q50_idx], 0.0)
    raw_target = np.maximum(predict_quantiles[..., target_idx], 0.0)
    symmetric_upper = np.empty_like(raw_target, dtype=np.float64)
    refined = np.empty_like(raw_target, dtype=np.float64)
    horizon_thresholds = {int(k): float(v) for k, v in dict(decision_lookup[cost_ratio]["horizon"]).items()}

    for h_idx, horizon in enumerate(horizons):
        global_s = threshold_value(thresholds_df, method="global_cqr", delta=delta, horizon=int(horizon))
        symmetric_upper[:, :, h_idx] = np.maximum(predict_quantiles[:, :, h_idx, q95_idx] + global_s, 0.0)
        refined[:, :, h_idx] = np.maximum(
            predict_quantiles[:, :, h_idx, target_idx] + horizon_thresholds[int(horizon)],
            0.0,
        )

    return {
        "median_decision": newsvendor_cost(median, labels, c_u=c_u, c_o=c_o),
        "symmetric_interval_upper_bound": newsvendor_cost(symmetric_upper, labels, c_u=c_u, c_o=c_o),
        "raw_cost_aligned_quantile": newsvendor_cost(raw_target, labels, c_u=c_u, c_o=c_o),
        "one_sided_refined_cost_aligned_quantile": newsvendor_cost(refined, labels, c_u=c_u, c_o=c_o),
    }


def decision_decisions_for_cost_ratio(
    predict_quantiles: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    thresholds_df: pd.DataFrame,
    decision_lookup: dict[str, dict[str, object]],
    cost_ratio: str,
    delta: float,
) -> dict[str, np.ndarray]:
    q50_idx = quantile_index(quantiles, 0.5)
    q95_idx = quantile_index(quantiles, 1.0 - delta / 2.0)
    target_q = float(decision_lookup[cost_ratio]["target_quantile"])
    target_idx = quantile_index(quantiles, target_q)
    median = np.maximum(predict_quantiles[..., q50_idx], 0.0)
    raw_target = np.maximum(predict_quantiles[..., target_idx], 0.0)
    symmetric_upper = np.empty_like(raw_target, dtype=np.float64)
    refined = np.empty_like(raw_target, dtype=np.float64)
    horizon_thresholds = {int(k): float(v) for k, v in dict(decision_lookup[cost_ratio]["horizon"]).items()}
    for h_idx, horizon in enumerate(horizons):
        global_s = threshold_value(thresholds_df, method="global_cqr", delta=delta, horizon=int(horizon))
        symmetric_upper[:, :, h_idx] = np.maximum(predict_quantiles[:, :, h_idx, q95_idx] + global_s, 0.0)
        refined[:, :, h_idx] = np.maximum(
            predict_quantiles[:, :, h_idx, target_idx] + horizon_thresholds[int(horizon)],
            0.0,
        )
    return {
        "median_decision": median,
        "symmetric_interval_upper_bound": symmetric_upper,
        "raw_cost_aligned_quantile": raw_target,
        "one_sided_refined_cost_aligned_quantile": refined,
    }


def strategy_metrics(decision: np.ndarray, labels: np.ndarray, costs: np.ndarray) -> dict[str, float]:
    shortage = np.maximum(labels - decision, 0.0)
    overage = np.maximum(decision - labels, 0.0)
    return {
        "expected_cost": float(np.mean(costs)),
        "event_shortage_rate": float(np.mean(decision < labels)),
        "shortage_amount": float(np.mean(shortage)),
        "event_overage_rate": float(np.mean(decision > labels)),
        "overage_amount": float(np.mean(overage)),
        "mean_decision": float(np.mean(decision)),
        "mean_label": float(np.mean(labels)),
    }


def build_deployment_strategy_comparison(
    predict_quantiles: np.ndarray,
    labels: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    thresholds_df: pd.DataFrame,
    decision_thresholds_df: pd.DataFrame,
    cost_ratios: list,
    delta: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    decision_lookup = decision_threshold_info(decision_thresholds_df)
    rows_horizon = []
    for ratio in cost_ratios:
        decisions = decision_decisions_for_cost_ratio(
            predict_quantiles=predict_quantiles,
            quantiles=quantiles,
            horizons=horizons,
            thresholds_df=thresholds_df,
            decision_lookup=decision_lookup,
            cost_ratio=ratio.label,
            delta=delta,
        )
        costs = {
            strategy: newsvendor_cost(decision, labels, c_u=ratio.c_u, c_o=ratio.c_o)
            for strategy, decision in decisions.items()
        }
        for h_idx, horizon in enumerate(horizons):
            labels_h = labels[:, :, h_idx]
            median_cost = float(np.mean(costs["median_decision"][:, :, h_idx]))
            symmetric_cost = float(np.mean(costs["symmetric_interval_upper_bound"][:, :, h_idx]))
            for strategy in DEPLOYMENT_STRATEGIES:
                values = strategy_metrics(
                    decision=decisions[strategy][:, :, h_idx],
                    labels=labels_h,
                    costs=costs[strategy][:, :, h_idx],
                )
                rows_horizon.append(
                    {
                        "strategy": strategy,
                        "cost_ratio": ratio.label,
                        "c_u": ratio.c_u,
                        "c_o": ratio.c_o,
                        "tau_star": ratio.tau_star,
                        "horizon": int(horizon),
                        **values,
                        "cost_reduction_vs_median": median_cost - values["expected_cost"],
                        "cost_reduction_vs_symmetric_upper": symmetric_cost - values["expected_cost"],
                    }
                )
    by_horizon = pd.DataFrame(rows_horizon)
    metric_cols = [
        "expected_cost",
        "event_shortage_rate",
        "shortage_amount",
        "event_overage_rate",
        "overage_amount",
        "mean_decision",
        "mean_label",
        "cost_reduction_vs_median",
        "cost_reduction_vs_symmetric_upper",
    ]
    summary = by_horizon.groupby(["strategy", "cost_ratio", "c_u", "c_o", "tau_star"], as_index=False)[metric_cols].mean()
    return summary, by_horizon


def build_station_group_decision_attribution(
    predict_quantiles: np.ndarray,
    labels: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    thresholds_df: pd.DataFrame,
    decision_thresholds_df: pd.DataFrame,
    cost_ratios: list,
    groups_df: pd.DataFrame,
    delta: float,
) -> pd.DataFrame:
    decision_lookup = decision_threshold_info(decision_thresholds_df)
    rows = []
    for group_idx, group in groups_df.groupby("zero_group_index"):
        station_indices = group["station_index"].to_numpy(dtype=int)
        labels_g = labels[:, station_indices, :]
        pred_g = predict_quantiles[:, station_indices, :, :]
        for ratio in cost_ratios:
            costs = decision_arrays_for_cost_ratio(
                predict_quantiles=pred_g,
                labels=labels_g,
                quantiles=quantiles,
                horizons=horizons,
                thresholds_df=thresholds_df,
                decision_lookup=decision_lookup,
                cost_ratio=ratio.label,
                c_u=ratio.c_u,
                c_o=ratio.c_o,
                delta=delta,
            )
            median = float(np.mean(costs["median_decision"]))
            raw = float(np.mean(costs["raw_cost_aligned_quantile"]))
            refined = float(np.mean(costs["one_sided_refined_cost_aligned_quantile"]))
            symmetric = float(np.mean(costs["symmetric_interval_upper_bound"]))
            q_gain = median - raw
            c_gain = raw - refined
            total_gain = median - refined
            rows.append(
                {
                    "zero_group_index": int(group_idx),
                    "zero_group": str(group.iloc[0]["zero_group"]),
                    "station_count": int(len(station_indices)),
                    "mean_station_zero_ratio": float(group["zero_ratio"].mean()),
                    "cost_ratio": ratio.label,
                    "c_u": ratio.c_u,
                    "c_o": ratio.c_o,
                    "tau_star": ratio.tau_star,
                    "median_cost": median,
                    "symmetric_upper_cost": symmetric,
                    "raw_cost_aligned_quantile_cost": raw,
                    "one_sided_refined_cost": refined,
                    "quantile_choice_gain": q_gain,
                    "calibration_gain": c_gain,
                    "total_gain_vs_median": total_gain,
                    "quantile_choice_share": q_gain / total_gain if total_gain else float("nan"),
                    "calibration_share": c_gain / total_gain if total_gain else float("nan"),
                    "raw_gain_vs_median_pct": q_gain / median * 100.0 if median else float("nan"),
                    "refined_gain_vs_symmetric_pct": (symmetric - refined) / symmetric * 100.0 if symmetric else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def build_test_time_manifest(
    data_dir: Path,
    metadata: dict,
    n_test_windows: int,
    horizons: list[int],
) -> TimeManifest:
    duration_path = data_dir / "duration.csv"
    duration_index = pd.read_csv(duration_path, usecols=[0]).iloc[:, 0]
    timestamps = pd.to_datetime(duration_index, errors="coerce")
    if timestamps.isna().any():
        sample_indices = np.arange(n_test_windows)
        date_keys = np.tile(sample_indices[:, None].astype(str), (1, len(horizons)))
        target_hours = np.zeros_like(date_keys, dtype=int)
        frame = pd.DataFrame(
            [
                {"sample_index": int(i), "horizon": int(h), "target_time": "", "target_date": str(i), "target_hour": 0}
                for i in range(n_test_windows)
                for h in horizons
            ]
        )
        return TimeManifest(frame=frame, date_keys=date_keys, target_hours=target_hours, bootstrap_level="test-window")

    split_rates = {key: float(value) for key, value in metadata["split_rates"].items()}
    seq_len = int(metadata["seq_len"])
    n_total = len(timestamps)
    calib_end = int(n_total * (split_rates["train"] + split_rates["valid"] + split_rates["calib"]))

    date_keys = np.empty((n_test_windows, len(horizons)), dtype=object)
    target_hours = np.empty((n_test_windows, len(horizons)), dtype=int)
    rows = []
    for i in range(n_test_windows):
        for h_idx, horizon in enumerate(horizons):
            target_idx = calib_end + i + seq_len + int(horizon) - 1
            if target_idx >= len(timestamps):
                target_ts = pd.NaT
                date_key = f"overflow_{i}_{horizon}"
                hour = -1
                target_str = ""
            else:
                target_ts = timestamps.iloc[target_idx]
                date_key = str(target_ts.date())
                hour = int(target_ts.hour)
                target_str = target_ts.isoformat()
            date_keys[i, h_idx] = date_key
            target_hours[i, h_idx] = hour
            rows.append(
                {
                    "sample_index": int(i),
                    "horizon": int(horizon),
                    "target_time": target_str,
                    "target_date": date_key,
                    "target_hour": hour,
                }
            )
    return TimeManifest(frame=pd.DataFrame(rows), date_keys=date_keys, target_hours=target_hours, bootstrap_level="day")


def aggregate_by_date(values: np.ndarray, date_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.shape[:2] != date_keys.shape:
        raise AssertionError(f"values first dims {values.shape[:2]} do not match date_keys {date_keys.shape}")
    unique_dates = np.array(sorted(pd.unique(date_keys.reshape(-1))))
    out = np.empty(len(unique_dates), dtype=np.float64)
    for idx, date in enumerate(unique_dates):
        mask = date_keys == date
        expanded = np.broadcast_to(mask[:, :, None], values.shape)
        out[idx] = float(np.mean(values[expanded]))
    return unique_dates, out


def build_day_level_bootstrap_ci(
    predict_quantiles: np.ndarray,
    labels: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    thresholds_df: pd.DataFrame,
    decision_thresholds_df: pd.DataFrame,
    cost_ratios: list,
    date_keys: np.ndarray,
    delta: float,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    decision_lookup = decision_threshold_info(decision_thresholds_df)
    rows = []
    for ratio in cost_ratios:
        costs = decision_arrays_for_cost_ratio(
            predict_quantiles=predict_quantiles,
            labels=labels,
            quantiles=quantiles,
            horizons=horizons,
            thresholds_df=thresholds_df,
            decision_lookup=decision_lookup,
            cost_ratio=ratio.label,
            c_u=ratio.c_u,
            c_o=ratio.c_o,
            delta=delta,
        )
        per_day = {}
        for strategy, cost in costs.items():
            _, values = aggregate_by_date(np.transpose(cost, (0, 2, 1)), date_keys)
            per_day[strategy] = values
            point, low, high = bootstrap_mean_ci(values, rng=rng, n_samples=n_samples)
            rows.append(
                {
                    "bootstrap_level": "day",
                    "metric": "expected_cost",
                    "strategy": strategy,
                    "cost_ratio": ratio.label,
                    "point": point,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_units": int(len(values)),
                    "bootstrap_samples": int(n_samples),
                }
            )

        median = per_day["median_decision"]
        symmetric = per_day["symmetric_interval_upper_bound"]
        raw = per_day["raw_cost_aligned_quantile"]
        refined = per_day["one_sided_refined_cost_aligned_quantile"]
        derived = {
            "raw_reduction_vs_median": median - raw,
            "refined_reduction_vs_median": median - refined,
            "refined_reduction_vs_symmetric": symmetric - refined,
            "one_sided_gain_vs_raw": raw - refined,
        }
        total_gain = median - refined
        with np.errstate(divide="ignore", invalid="ignore"):
            derived["quantile_choice_share"] = np.divide(median - raw, total_gain)
            derived["calibration_share"] = np.divide(raw - refined, total_gain)

        for metric, values in derived.items():
            finite = values[np.isfinite(values)]
            point, low, high = bootstrap_mean_ci(finite, rng=rng, n_samples=n_samples)
            rows.append(
                {
                    "bootstrap_level": "day",
                    "metric": metric,
                    "strategy": "one_sided_refined_cost_aligned_quantile",
                    "cost_ratio": ratio.label,
                    "point": point,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_units": int(len(finite)),
                    "bootstrap_samples": int(n_samples),
                }
            )
    return pd.DataFrame(rows)


def write_plots(
    output_dir: Path,
    station_scatter_df: pd.DataFrame,
    station_attr_df: pd.DataFrame,
    deployment_summary_df: pd.DataFrame,
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except Exception:
        return []

    written: list[str] = []

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = station_scatter_df[np.isfinite(station_scatter_df["ZBR_share"])].copy()
    ax.scatter(scatter["zero_ratio"], scatter["ZBR_share"], s=12, alpha=0.45, color="#4C78A8")
    ax.set_xlabel("Station zero ratio")
    ax.set_ylabel("ZBR share among rescued samples")
    ax.set_title("Station zero ratio vs zero-boundary rescue share")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = output_dir / "station_zero_ratio_vs_zbr_share.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    written.append(path.name)

    plot_attr = station_attr_df[station_attr_df["cost_ratio"].isin(["3:1", "5:1", "9:1", "19:1"])].copy()
    plot_attr = plot_attr.groupby(["zero_group", "zero_group_index"], as_index=False)["quantile_choice_share"].mean()
    plot_attr = plot_attr.sort_values("zero_group_index")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(plot_attr["zero_group"], plot_attr["quantile_choice_share"], color="#59A14F")
    ax.set_ylim(0.95, 1.005)
    ax.set_xlabel("Station zero-ratio group")
    ax.set_ylabel("Mean quantile-choice share")
    ax.set_title("Decision-value attribution by station sparsity group")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = output_dir / "station_group_decision_attribution.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    written.append(path.name)

    plot_dep = deployment_summary_df[deployment_summary_df["cost_ratio"].isin(["3:1", "5:1", "9:1", "19:1"])].copy()
    strategies = DEPLOYMENT_STRATEGIES
    ratios = ["3:1", "5:1", "9:1", "19:1"]
    x = np.arange(len(ratios))
    width = 0.8 / len(strategies)
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, strategy in enumerate(strategies):
        values = (
            plot_dep[plot_dep["strategy"] == strategy]
            .set_index("cost_ratio")
            .reindex(ratios)["expected_cost"]
        )
        ax.bar(x + (idx - (len(strategies) - 1) / 2) * width, values, width=width, label=strategy)
    ax.set_xticks(x)
    ax.set_xticklabels(ratios)
    ax.set_xlabel("Cost ratio")
    ax.set_ylabel("Expected cost")
    ax.set_title("Deployment strategy comparison")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "deployment_strategy_cost_bar.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    written.append(path.name)

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.axis("off")
    steps = [
        "Train multi-horizon\nprobabilistic forecaster",
        "Run boundary-aware\nreliability diagnostics",
        "Choose cost-aligned\n target quantile",
        "Apply lightweight\none-sided calibration",
        "Use targeted local\ncalibration only for hard cells",
    ]
    for idx, text in enumerate(steps):
        x0 = 0.02 + idx * 0.195
        box = FancyBboxPatch(
            (x0, 0.35),
            0.16,
            0.34,
            boxstyle="round,pad=0.02,rounding_size=0.015",
            linewidth=1.2,
            edgecolor="#333333",
            facecolor="#EAF2F8" if idx < 4 else "#FDEBD0",
        )
        ax.add_patch(box)
        ax.text(x0 + 0.08, 0.52, text, ha="center", va="center", fontsize=9)
        if idx < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x0 + 0.185, 0.52),
                xytext=(x0 + 0.162, 0.52),
                arrowprops={"arrowstyle": "->", "linewidth": 1.2, "color": "#333333"},
            )
    ax.set_title("Deployment diagnostic protocol", fontsize=12)
    fig.tight_layout()
    path = output_dir / "deployment_diagnostic_flowchart.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    written.append(path.name)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build journal reinforcement experiments after paper-assets export.")
    parser.add_argument("--stage1-output-dir", required=True)
    parser.add_argument("--stage2-output-dir", required=True)
    parser.add_argument("--stage4-output-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deltas", default="0.1")
    parser.add_argument("--primary-delta", type=float, default=0.1)
    parser.add_argument("--cost-ratios", default="1:1,3:1,5:1,9:1,19:1")
    parser.add_argument("--station-groups", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--bootstrap-seed", type=int, default=20260506)
    parser.add_argument("--max-test-windows", type=int, default=0)
    parser.add_argument("--machine", default="Lenovo")
    args = parser.parse_args()

    stage1_output_dir = resolve_path(args.stage1_output_dir)
    stage2_output_dir = resolve_path(args.stage2_output_dir)
    stage4_output_dir = resolve_path(args.stage4_output_dir)
    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)

    metadata = load_json(stage1_output_dir / "run_metadata.json")
    horizons = [int(h) for h in metadata["horizons"]]
    quantiles = [float(q) for q in metadata["quantiles"]]
    deltas = parse_float_list(args.deltas)
    cost_ratios = parse_cost_ratios(args.cost_ratios)

    predict_quantiles = np.load(stage1_output_dir / "predict_quantiles.npy")
    labels = np.load(stage1_output_dir / "label_list.npy")
    if args.max_test_windows > 0:
        predict_quantiles = predict_quantiles[: args.max_test_windows]
        labels = labels[: args.max_test_windows]

    assert_quantile_tensor_shape(
        predict_quantiles,
        n_horizons=len(horizons),
        n_quantiles=len(quantiles),
        context="reinforcement predict_quantiles",
    )
    assert_target_tensor_shape(labels, n_horizons=len(horizons), context="reinforcement labels")
    assert_monotonic_quantiles(predict_quantiles, context="reinforcement predict_quantiles")

    thresholds_df = pd.read_csv(stage2_output_dir / "cqr_thresholds.csv")
    decision_thresholds_df = pd.read_csv(stage4_output_dir / "decision_one_sided_thresholds.csv")
    station_ids = pd.read_csv(data_dir / "duration.csv", nrows=0).columns[1:].tolist()
    groups_df = station_zero_groups(labels, station_ids=station_ids, n_groups=args.station_groups)
    summary_groups = group_summary(groups_df)
    station_sensitivity = build_station_zero_inflation_sensitivity(
        predict_quantiles=predict_quantiles,
        labels=labels,
        thresholds_df=thresholds_df,
        quantiles=quantiles,
        horizons=horizons,
        groups_df=groups_df,
        deltas=deltas,
    )
    station_scatter = build_station_level_rescue_scatter(
        predict_quantiles=predict_quantiles,
        labels=labels,
        thresholds_df=thresholds_df,
        quantiles=quantiles,
        horizons=horizons,
        groups_df=groups_df,
        delta=args.primary_delta,
    )
    station_attr = build_station_group_decision_attribution(
        predict_quantiles=predict_quantiles,
        labels=labels,
        quantiles=quantiles,
        horizons=horizons,
        thresholds_df=thresholds_df,
        decision_thresholds_df=decision_thresholds_df,
        cost_ratios=cost_ratios,
        groups_df=groups_df,
        delta=args.primary_delta,
    )
    deployment_summary, deployment_by_horizon = build_deployment_strategy_comparison(
        predict_quantiles=predict_quantiles,
        labels=labels,
        quantiles=quantiles,
        horizons=horizons,
        thresholds_df=thresholds_df,
        decision_thresholds_df=decision_thresholds_df,
        cost_ratios=cost_ratios,
        delta=args.primary_delta,
    )
    manifest = build_test_time_manifest(
        data_dir=data_dir,
        metadata=metadata,
        n_test_windows=labels.shape[0],
        horizons=horizons,
    )
    day_bootstrap = build_day_level_bootstrap_ci(
        predict_quantiles=predict_quantiles,
        labels=labels,
        quantiles=quantiles,
        horizons=horizons,
        thresholds_df=thresholds_df,
        decision_thresholds_df=decision_thresholds_df,
        cost_ratios=cost_ratios,
        date_keys=manifest.date_keys,
        delta=args.primary_delta,
        n_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    groups_df.to_csv(output_dir / "station_zero_ratio_by_station.csv", index=False)
    summary_groups.to_csv(output_dir / "station_zero_group_summary.csv", index=False)
    station_sensitivity.to_csv(output_dir / "station_zero_inflation_sensitivity.csv", index=False)
    station_scatter.to_csv(output_dir / "station_zero_ratio_vs_zbr_share.csv", index=False)
    station_attr.to_csv(output_dir / "station_zero_group_decision_attribution.csv", index=False)
    deployment_summary.to_csv(output_dir / "deployment_strategy_comparison.csv", index=False)
    deployment_by_horizon.to_csv(output_dir / "deployment_strategy_by_horizon.csv", index=False)
    manifest.frame.to_csv(output_dir / "test_window_manifest.csv", index=False)
    day_bootstrap.to_csv(output_dir / "bootstrap_ci_day_level.csv", index=False)
    plot_files = write_plots(
        output_dir=output_dir,
        station_scatter_df=station_scatter,
        station_attr_df=station_attr,
        deployment_summary_df=deployment_summary,
    )

    metadata_out = {
        "stage": "reinforcement",
        "goal": "Station-level zero-inflation sensitivity, deployment strategy comparison, day-level bootstrap, and deployment flowchart",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "machine": str(args.machine),
        "branch": git_value(["branch", "--show-current"]),
        "commit": git_value(["rev-parse", "--short", "HEAD"]),
        "stage1_output_dir": str(stage1_output_dir),
        "stage2_output_dir": str(stage2_output_dir),
        "stage4_output_dir": str(stage4_output_dir),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "horizons": horizons,
        "quantiles": quantiles,
        "deltas": deltas,
        "primary_delta": args.primary_delta,
        "cost_ratios": [ratio.__dict__ for ratio in cost_ratios],
        "station_groups": int(args.station_groups),
        "bootstrap_level": manifest.bootstrap_level,
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "input_shapes": {
            "predict_quantiles": list(predict_quantiles.shape),
            "labels": list(labels.shape),
        },
        "output_files": {
            "station_zero_ratio_by_station": "station_zero_ratio_by_station.csv",
            "station_zero_group_summary": "station_zero_group_summary.csv",
            "station_zero_inflation_sensitivity": "station_zero_inflation_sensitivity.csv",
            "station_zero_ratio_vs_zbr_share": "station_zero_ratio_vs_zbr_share.csv",
            "station_zero_group_decision_attribution": "station_zero_group_decision_attribution.csv",
            "deployment_strategy_comparison": "deployment_strategy_comparison.csv",
            "deployment_strategy_by_horizon": "deployment_strategy_by_horizon.csv",
            "test_window_manifest": "test_window_manifest.csv",
            "bootstrap_ci_day_level": "bootstrap_ci_day_level.csv",
            "plots": plot_files,
        },
    }
    (output_dir / "reinforcement_metadata.json").write_text(json.dumps(metadata_out, indent=2), encoding="utf-8")

    print(summary_groups.to_string(index=False))
    print(
        station_sensitivity[
            (station_sensitivity["method"] == "global_cqr")
            & np.isclose(station_sensitivity["delta"].astype(float), args.primary_delta)
        ][
            [
                "zero_group",
                "horizon",
                "mean_station_zero_ratio",
                "all_PICP",
                "zero_PICP",
                "positive_PICP",
                "ZBR_share",
            ]
        ].to_string(index=False)
    )
    print(deployment_summary.to_string(index=False))
    print(json.dumps(metadata_out, indent=2))


if __name__ == "__main__":
    main()
