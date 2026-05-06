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
from scripts.journal.evaluate_stage4_decision import (
    METHODS as DECISION_METHODS,
    newsvendor_cost,
    parse_cost_ratios,
)
from scripts.journal.train_multihorizon_raw import parse_float_list, quantile_index
from utils.model_training.conformal import interval_metrics
from utils.model_training.journal_contracts import (
    assert_monotonic_quantiles,
    assert_quantile_tensor_shape,
    assert_target_tensor_shape,
)


INTERVAL_METHODS = ["raw", "global_cqr", "horizon_cqr"]
DEFAULT_DELTAS = [0.1]


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


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def threshold_value(thresholds_df: pd.DataFrame, method: str, delta: float, horizon: int) -> float:
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
        raise ValueError(f"Unsupported interval method: {method}")
    if len(rows) != 1:
        raise ValueError(f"Expected one threshold for method={method}, delta={delta}, horizon={horizon}; got {len(rows)}")
    return float(rows.iloc[0]["s_hat"])


def interval_for_method(
    raw_lower: np.ndarray,
    raw_upper: np.ndarray,
    method: str,
    s_hat: float,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "raw":
        return raw_lower, raw_upper
    return apply_cqr_threshold(raw_lower, raw_upper, s_hat=s_hat, nonnegative=True)


def subset_masks(labels: np.ndarray, positive_eps: float = 1e-12) -> tuple[list[tuple[str, np.ndarray]], dict[str, float]]:
    positive = labels > positive_eps
    zero = ~positive
    if np.any(positive):
        positive_values = labels[positive]
        q50 = float(np.quantile(positive_values, 0.50))
        q90 = float(np.quantile(positive_values, 0.90))
        q95 = float(np.quantile(positive_values, 0.95))
    else:
        q50 = q90 = q95 = float("nan")

    masks = [
        ("all", np.ones_like(labels, dtype=bool)),
        ("zero", zero),
        ("positive", positive),
        ("low_positive", positive & (labels <= q50)),
        ("top10", positive & (labels >= q90)),
        ("top5", positive & (labels >= q95)),
    ]
    thresholds = {"positive_q50": q50, "positive_q90": q90, "positive_q95": q95}
    return masks, thresholds


def metrics_for_mask(
    labels: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    point: np.ndarray,
    mask: np.ndarray,
    delta: float,
) -> dict[str, float | int]:
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
    metrics = interval_metrics(labels[mask], lower[mask], upper[mask], point_pred=point[mask], delta=delta)
    nominal = 1.0 - delta
    picp = float(metrics["PICP"])
    gap = picp - nominal
    return {
        "count": count,
        "PICP": picp,
        "coverage_gap": float(gap),
        "ACE": float(abs(gap)),
        "undercoverage": float(max(nominal - picp, 0.0)),
        "MPIW": float(metrics["MPIW"]),
        "WIS": float(metrics.get("WIS", np.nan)),
    }


def build_boundary_reliability_table(
    predict_quantiles: np.ndarray,
    labels: np.ndarray,
    thresholds_df: pd.DataFrame,
    quantiles: list[float],
    horizons: list[int],
    deltas: list[float],
    boundary_df: pd.DataFrame | None = None,
    positive_eps: float = 1e-12,
) -> pd.DataFrame:
    q50_idx = quantile_index(quantiles, 0.5)
    rows: list[dict[str, float | int | str]] = []

    for delta in deltas:
        lower_idx = quantile_index(quantiles, delta / 2.0)
        upper_idx = quantile_index(quantiles, 1.0 - delta / 2.0)
        for h_idx, horizon in enumerate(horizons):
            labels_h = labels[:, :, h_idx]
            point_h = predict_quantiles[:, :, h_idx, q50_idx]
            raw_lower = predict_quantiles[:, :, h_idx, lower_idx]
            raw_upper = predict_quantiles[:, :, h_idx, upper_idx]
            masks, demand_thresholds = subset_masks(labels_h, positive_eps=positive_eps)

            for method in INTERVAL_METHODS:
                s_hat = threshold_value(thresholds_df, method=method, delta=delta, horizon=int(horizon))
                lower, upper = interval_for_method(raw_lower, raw_upper, method=method, s_hat=s_hat)
                row: dict[str, float | int | str] = {
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

                row["PICP_zero_minus_positive"] = float(row["zero_PICP"] - row["positive_PICP"])
                row["ACE_positive"] = float(row["positive_ACE"])
                row["MPIW_positive"] = float(row["positive_MPIW"])
                row["WIS_positive"] = float(row["positive_WIS"])
                row["PICP_top10"] = float(row["top10_PICP"])
                row["PICP_top5"] = float(row["top5_PICP"])
                row["ZBR_share"] = float("nan")
                row["rescued_share_of_raw_misses"] = float("nan")

                if boundary_df is not None and method != "raw":
                    match = boundary_df[
                        (boundary_df["method"] == method)
                        & np.isclose(boundary_df["delta"].astype(float), delta)
                        & (boundary_df["horizon"].astype(str) == str(horizon))
                    ]
                    if len(match) == 1:
                        row["ZBR_share"] = float(match.iloc[0]["clipping_share_of_rescues"])
                        row["rescued_share_of_raw_misses"] = float(match.iloc[0]["rescued_share_of_raw_misses"])

                rows.append(row)

    return pd.DataFrame(rows)


def build_coverage_gain_decomposition(boundary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for _, row in boundary_df.iterrows():
        total = float(row["total_count"])
        zero_rescue = float(row["zero_rescued_count"]) / total
        positive_rescue = float(row["positive_rescued_count"]) / total
        raw_picp = 1.0 - float(row["raw_miss_rate"])
        calibrated_picp = raw_picp + zero_rescue + positive_rescue
        rows.append(
            {
                "method": row["method"],
                "delta": float(row["delta"]),
                "nominal_coverage": float(1.0 - float(row["delta"])),
                "horizon": int(row["horizon"]),
                "raw_PICP": raw_picp,
                "zero_boundary_rescue_gain": zero_rescue,
                "positive_rescue_gain": positive_rescue,
                "total_rescue_gain": zero_rescue + positive_rescue,
                "calibrated_PICP_from_rescues": calibrated_picp,
                "zero_share_of_rescues": float(row["zero_share_of_rescues"]),
                "clipping_share_of_rescues": float(row["clipping_share_of_rescues"]),
                "rescued_share_of_raw_misses": float(row["rescued_share_of_raw_misses"]),
            }
        )
    return pd.DataFrame(rows)


def build_hard_cell_summary(hour_positive_df: pd.DataFrame, delta: float, top_k: int) -> pd.DataFrame:
    filtered = hour_positive_df[
        np.isclose(hour_positive_df["delta"].astype(float), delta)
        & hour_positive_df["method"].isin(["global_cqr", "horizon_cqr"])
    ].copy()
    filtered["ACE"] = filtered["ACE"].astype(float)
    return filtered.sort_values("ACE", ascending=False).head(top_k).reset_index(drop=True)


def load_decision_threshold_lookup(thresholds_df: pd.DataFrame) -> dict[str, dict[str, object]]:
    lookup: dict[str, dict[str, object]] = {}
    for cost_ratio in sorted(thresholds_df["cost_ratio"].unique()):
        rows = thresholds_df[thresholds_df["cost_ratio"] == cost_ratio]
        global_row = rows[rows["method"] == "global_cqr"]
        if len(global_row) != 1:
            raise ValueError(f"Expected one global one-sided threshold for cost ratio {cost_ratio}")
        horizon_rows = rows[rows["method"] == "horizon_cqr"]
        lookup[str(cost_ratio)] = {
            "target_quantile": float(global_row.iloc[0]["target_quantile"]),
            "global": float(global_row.iloc[0]["s_hat"]),
            "horizon": {
                int(row["horizon"]): float(row["s_hat"])
                for _, row in horizon_rows.iterrows()
            },
        }
    return lookup


def build_decision_value_attribution(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for cost_ratio in sorted(summary_df["cost_ratio"].unique()):
        subset = summary_df[summary_df["cost_ratio"] == cost_ratio].set_index("method")
        median = float(subset.loc["median", "expected_cost"])
        raw = float(subset.loc["raw_target_quantile", "expected_cost"])
        global_cost = float(subset.loc["global_cqr", "expected_cost"])
        horizon = float(subset.loc["horizon_cqr", "expected_cost"])
        oracle = float(subset.loc["oracle_decision", "expected_cost"])
        raw_gain = median - raw
        global_gain = raw - global_cost
        horizon_gain = raw - horizon
        total_horizon_gain = median - horizon
        total_possible_gain = median - oracle
        rows.append(
            {
                "cost_ratio": str(cost_ratio),
                "c_u": float(subset.loc["median", "c_u"]),
                "c_o": float(subset.loc["median", "c_o"]),
                "tau_star": float(subset.loc["median", "tau_star"]),
                "median_cost": median,
                "raw_target_cost": raw,
                "global_cqr_cost": global_cost,
                "horizon_cqr_cost": horizon,
                "oracle_cost": oracle,
                "quantile_choice_gain": raw_gain,
                "global_calibration_gain": global_gain,
                "horizon_calibration_gain": horizon_gain,
                "total_horizon_gain_vs_median": total_horizon_gain,
                "total_possible_gain_vs_oracle": total_possible_gain,
                "quantile_choice_share_of_horizon_gain": raw_gain / total_horizon_gain if total_horizon_gain else float("nan"),
                "horizon_calibration_share_of_horizon_gain": horizon_gain / total_horizon_gain if total_horizon_gain else float("nan"),
                "horizon_gain_vs_raw_pct": horizon_gain / raw * 100.0 if raw else float("nan"),
                "raw_gain_vs_median_pct": raw_gain / median * 100.0 if median else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_decision_value_attribution_by_horizon(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (cost_ratio, horizon), group in metrics_df.groupby(["cost_ratio", "horizon"]):
        subset = group.set_index("method")
        median = float(subset.loc["median", "expected_cost"])
        raw = float(subset.loc["raw_target_quantile", "expected_cost"])
        global_cost = float(subset.loc["global_cqr", "expected_cost"])
        horizon_cost = float(subset.loc["horizon_cqr", "expected_cost"])
        oracle = float(subset.loc["oracle_decision", "expected_cost"])
        raw_gain = median - raw
        horizon_gain = raw - horizon_cost
        total_gain = median - horizon_cost
        rows.append(
            {
                "cost_ratio": str(cost_ratio),
                "horizon": int(horizon),
                "median_cost": median,
                "raw_target_cost": raw,
                "global_cqr_cost": global_cost,
                "horizon_cqr_cost": horizon_cost,
                "oracle_cost": oracle,
                "quantile_choice_gain": raw_gain,
                "horizon_calibration_gain": horizon_gain,
                "total_horizon_gain_vs_median": total_gain,
                "quantile_choice_share_of_horizon_gain": raw_gain / total_gain if total_gain else float("nan"),
                "horizon_calibration_share_of_horizon_gain": horizon_gain / total_gain if total_gain else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def decision_arrays_for_cost_ratio(
    predict_quantiles: np.ndarray,
    labels: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    cost_ratio: str,
    c_u: float,
    c_o: float,
    threshold_lookup: dict[str, dict[str, object]],
) -> dict[str, np.ndarray]:
    info = threshold_lookup[cost_ratio]
    target_q = float(info["target_quantile"])
    target_idx = quantile_index(quantiles, target_q)
    q50_idx = quantile_index(quantiles, 0.5)
    raw_target = np.maximum(predict_quantiles[..., target_idx], 0.0)
    median = np.maximum(predict_quantiles[..., q50_idx], 0.0)
    global_decision = np.maximum(predict_quantiles[..., target_idx] + float(info["global"]), 0.0)
    horizon_decision = np.empty_like(raw_target, dtype=np.float64)
    horizon_thresholds = {int(k): float(v) for k, v in dict(info["horizon"]).items()}
    for h_idx, horizon in enumerate(horizons):
        horizon_decision[:, :, h_idx] = np.maximum(
            predict_quantiles[:, :, h_idx, target_idx] + horizon_thresholds[int(horizon)],
            0.0,
        )
    oracle = np.maximum(labels, 0.0)
    return {
        "median": newsvendor_cost(median, labels, c_u=c_u, c_o=c_o),
        "raw_target_quantile": newsvendor_cost(raw_target, labels, c_u=c_u, c_o=c_o),
        "global_cqr": newsvendor_cost(global_decision, labels, c_u=c_u, c_o=c_o),
        "horizon_cqr": newsvendor_cost(horizon_decision, labels, c_u=c_u, c_o=c_o),
        "oracle_decision": newsvendor_cost(oracle, labels, c_u=c_u, c_o=c_o),
    }


def build_cost_by_quantile_curve(
    predict_quantiles: np.ndarray,
    labels: np.ndarray,
    quantiles: list[float],
    cost_ratios: list,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for ratio in cost_ratios:
        for q_idx, q_value in enumerate(quantiles):
            decision = np.maximum(predict_quantiles[..., q_idx], 0.0)
            cost = newsvendor_cost(decision, labels, c_u=ratio.c_u, c_o=ratio.c_o)
            rows.append(
                {
                    "cost_ratio": ratio.label,
                    "c_u": ratio.c_u,
                    "c_o": ratio.c_o,
                    "tau_star": ratio.tau_star,
                    "selected_quantile": float(q_value),
                    "expected_cost": float(np.mean(cost)),
                    "is_target_quantile": bool(abs(float(q_value) - ratio.tau_star) < 1e-9),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_samples: int, alpha: float = 0.05) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise AssertionError(f"bootstrap_mean_ci expects a 1D array, got {values.shape}")
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    draws = np.empty(n_samples, dtype=np.float64)
    for sample_idx in range(n_samples):
        indices = rng.integers(0, values.size, size=values.size)
        draws[sample_idx] = float(np.mean(values[indices]))
    return float(np.mean(values)), float(np.quantile(draws, alpha / 2.0)), float(np.quantile(draws, 1.0 - alpha / 2.0))


def bootstrap_ratio_ci(
    numerator: np.ndarray,
    denominator: np.ndarray,
    rng: np.random.Generator,
    n_samples: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    if numerator.shape != denominator.shape:
        raise AssertionError(f"numerator shape {numerator.shape} != denominator shape {denominator.shape}")
    denom_sum = float(np.sum(denominator))
    point = float(np.sum(numerator) / denom_sum) if denom_sum else float("nan")
    draws = np.empty(n_samples, dtype=np.float64)
    for sample_idx in range(n_samples):
        indices = rng.integers(0, numerator.size, size=numerator.size)
        denom = float(np.sum(denominator[indices]))
        draws[sample_idx] = float(np.sum(numerator[indices]) / denom) if denom else float("nan")
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(finite, alpha / 2.0)), float(np.quantile(finite, 1.0 - alpha / 2.0))


def build_bootstrap_ci(
    predict_quantiles: np.ndarray,
    labels: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    thresholds_df: pd.DataFrame,
    decision_summary_df: pd.DataFrame,
    decision_threshold_lookup: dict[str, dict[str, object]],
    cost_ratios: list,
    delta: float,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    q50_idx = quantile_index(quantiles, 0.5)
    lower_idx = quantile_index(quantiles, delta / 2.0)
    upper_idx = quantile_index(quantiles, 1.0 - delta / 2.0)

    for ratio in cost_ratios:
        costs = decision_arrays_for_cost_ratio(
            predict_quantiles=predict_quantiles,
            labels=labels,
            quantiles=quantiles,
            horizons=horizons,
            cost_ratio=ratio.label,
            c_u=ratio.c_u,
            c_o=ratio.c_o,
            threshold_lookup=decision_threshold_lookup,
        )
        per_window_cost = {
            method: np.mean(cost, axis=(1, 2))
            for method, cost in costs.items()
        }
        for method in DECISION_METHODS:
            point, low, high = bootstrap_mean_ci(per_window_cost[method], rng=rng, n_samples=n_samples)
            rows.append(
                {
                    "family": "decision",
                    "metric": "expected_cost",
                    "method": method,
                    "cost_ratio": ratio.label,
                    "horizon": "all",
                    "delta": float("nan"),
                    "point": point,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_samples": n_samples,
                }
            )

        median = per_window_cost["median"]
        raw = per_window_cost["raw_target_quantile"]
        horizon = per_window_cost["horizon_cqr"]
        raw_reduction = median - raw
        horizon_reduction = median - horizon
        calibration_gain = raw - horizon
        for metric_name, values in [
            ("raw_reduction_vs_median", raw_reduction),
            ("horizon_reduction_vs_median", horizon_reduction),
            ("horizon_calibration_gain_vs_raw", calibration_gain),
        ]:
            point, low, high = bootstrap_mean_ci(values, rng=rng, n_samples=n_samples)
            rows.append(
                {
                    "family": "decision",
                    "metric": metric_name,
                    "method": "horizon_cqr",
                    "cost_ratio": ratio.label,
                    "horizon": "all",
                    "delta": float("nan"),
                    "point": point,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_samples": n_samples,
                }
            )
        q_share, q_low, q_high = bootstrap_ratio_ci(raw_reduction, horizon_reduction, rng=rng, n_samples=n_samples)
        c_share, c_low, c_high = bootstrap_ratio_ci(calibration_gain, horizon_reduction, rng=rng, n_samples=n_samples)
        rows.extend(
            [
                {
                    "family": "decision",
                    "metric": "quantile_choice_share_of_horizon_gain",
                    "method": "horizon_cqr",
                    "cost_ratio": ratio.label,
                    "horizon": "all",
                    "delta": float("nan"),
                    "point": q_share,
                    "ci_low": q_low,
                    "ci_high": q_high,
                    "bootstrap_samples": n_samples,
                },
                {
                    "family": "decision",
                    "metric": "horizon_calibration_share_of_horizon_gain",
                    "method": "horizon_cqr",
                    "cost_ratio": ratio.label,
                    "horizon": "all",
                    "delta": float("nan"),
                    "point": c_share,
                    "ci_low": c_low,
                    "ci_high": c_high,
                    "bootstrap_samples": n_samples,
                },
            ]
        )

    for h_idx, horizon in enumerate(horizons):
        labels_h = labels[:, :, h_idx]
        point_h = predict_quantiles[:, :, h_idx, q50_idx]
        raw_lower = predict_quantiles[:, :, h_idx, lower_idx]
        raw_upper = predict_quantiles[:, :, h_idx, upper_idx]
        positive = labels_h > 1e-12
        raw_cover = (raw_lower <= labels_h) & (labels_h <= raw_upper)
        for method in INTERVAL_METHODS:
            s_hat = threshold_value(thresholds_df, method=method, delta=delta, horizon=int(horizon))
            lower, upper = interval_for_method(raw_lower, raw_upper, method=method, s_hat=s_hat)
            cover = (lower <= labels_h) & (labels_h <= upper)
            numerator = np.sum(cover & positive, axis=1)
            denominator = np.sum(positive, axis=1)
            point, low, high = bootstrap_ratio_ci(numerator, denominator, rng=rng, n_samples=n_samples)
            rows.append(
                {
                    "family": "reliability",
                    "metric": "PICP_positive",
                    "method": method,
                    "cost_ratio": "",
                    "horizon": int(horizon),
                    "delta": float(delta),
                    "point": point,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_samples": n_samples,
                }
            )
            if method != "raw":
                rescued = (~raw_cover) & cover
                zero_rescued = rescued & (labels_h <= 1e-12)
                numerator = np.sum(zero_rescued, axis=1)
                denominator = np.sum(rescued, axis=1)
                point, low, high = bootstrap_ratio_ci(numerator, denominator, rng=rng, n_samples=n_samples)
                rows.append(
                    {
                        "family": "reliability",
                        "metric": "ZBR_share",
                        "method": method,
                        "cost_ratio": "",
                        "horizon": int(horizon),
                        "delta": float(delta),
                        "point": point,
                        "ci_low": low,
                        "ci_high": high,
                        "bootstrap_samples": n_samples,
                    }
                )

    return pd.DataFrame(rows)


def write_plots(
    output_dir: Path,
    decomposition_df: pd.DataFrame,
    hard_cell_df: pd.DataFrame,
    decision_attr_df: pd.DataFrame,
    cost_curve_df: pd.DataFrame,
    delta: float,
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    written: list[str] = []

    plot_df = decomposition_df[
        (decomposition_df["method"] == "global_cqr")
        & np.isclose(decomposition_df["delta"].astype(float), delta)
    ].sort_values("horizon")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(plot_df))
    ax.bar(x, plot_df["raw_PICP"], label="raw PICP", color="#5B6770")
    ax.bar(
        x,
        plot_df["zero_boundary_rescue_gain"],
        bottom=plot_df["raw_PICP"],
        label="zero-boundary rescue",
        color="#1F77B4",
    )
    ax.bar(
        x,
        plot_df["positive_rescue_gain"],
        bottom=plot_df["raw_PICP"] + plot_df["zero_boundary_rescue_gain"],
        label="positive rescue",
        color="#FFB000",
    )
    ax.axhline(1.0 - delta, color="#C44E52", linestyle="--", linewidth=1.2, label="nominal")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(h)) for h in plot_df["horizon"]])
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Coverage")
    ax.set_title("Coverage gain decomposition")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "coverage_gain_decomposition.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    written.append(path.name)

    heat_df = hard_cell_df[
        (hard_cell_df["method"] == "global_cqr")
        & np.isclose(hard_cell_df["delta"].astype(float), delta)
    ]
    if not heat_df.empty:
        pivot = heat_df.pivot_table(index="horizon", columns="hour", values="ACE", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(10, 4))
        image = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([str(int(hour)) for hour in pivot.columns], fontsize=8)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([str(int(h)) for h in pivot.index])
        ax.set_xlabel("Target hour")
        ax.set_ylabel("Horizon")
        ax.set_title("Positive-demand ACE heatmap, global CQR")
        fig.colorbar(image, ax=ax, label="ACE")
        fig.tight_layout()
        path = output_dir / "positive_demand_ace_heatmap_global_cqr.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        written.append(path.name)

    attr_plot = decision_attr_df.sort_values("tau_star")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(attr_plot))
    ax.bar(x, attr_plot["quantile_choice_gain"], label="quantile choice gain", color="#4C78A8")
    ax.bar(
        x,
        attr_plot["horizon_calibration_gain"],
        bottom=attr_plot["quantile_choice_gain"],
        label="horizon calibration gain",
        color="#F58518",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(attr_plot["cost_ratio"])
    ax.set_xlabel("Cost ratio")
    ax.set_ylabel("Expected-cost reduction vs median")
    ax.set_title("Decision-value attribution")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "decision_value_attribution.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    written.append(path.name)

    fig, ax = plt.subplots(figsize=(8, 5))
    for cost_ratio in sorted(cost_curve_df["cost_ratio"].unique()):
        subset = cost_curve_df[cost_curve_df["cost_ratio"] == cost_ratio].sort_values("selected_quantile")
        ax.plot(subset["selected_quantile"], subset["expected_cost"], marker="o", label=cost_ratio)
    ax.set_xlabel("Selected quantile")
    ax.set_ylabel("Expected cost")
    ax.set_title("Decision cost by selected quantile")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Cost ratio")
    fig.tight_layout()
    path = output_dir / "decision_cost_by_quantile_curve.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    written.append(path.name)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-facing evidence assets for the journal branch.")
    parser.add_argument("--stage1-output-dir", required=True)
    parser.add_argument("--stage2-output-dir", required=True)
    parser.add_argument("--stage2-diagnostics-dir", required=True)
    parser.add_argument("--stage4-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deltas", default=",".join(str(delta) for delta in DEFAULT_DELTAS))
    parser.add_argument("--primary-delta", type=float, default=0.1)
    parser.add_argument("--cost-ratios", default="1:1,3:1,5:1,9:1,19:1")
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--bootstrap-seed", type=int, default=20260506)
    parser.add_argument("--max-test-windows", type=int, default=0)
    parser.add_argument("--hard-cell-top-k", type=int, default=20)
    parser.add_argument("--machine", default="Lenovo")
    args = parser.parse_args()

    stage1_output_dir = resolve_path(args.stage1_output_dir)
    stage2_output_dir = resolve_path(args.stage2_output_dir)
    diagnostics_dir = resolve_path(args.stage2_diagnostics_dir)
    stage4_output_dir = resolve_path(args.stage4_output_dir)
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
        context="paper assets predict_quantiles",
    )
    assert_target_tensor_shape(labels, n_horizons=len(horizons), context="paper assets labels")
    assert_monotonic_quantiles(predict_quantiles, context="paper assets predict_quantiles")

    thresholds_df = pd.read_csv(stage2_output_dir / "cqr_thresholds.csv")
    boundary_df = pd.read_csv(diagnostics_dir / "boundary_rescue_summary.csv")
    hour_positive_df = pd.read_csv(diagnostics_dir / "hour_horizon_coverage_positive_only.csv")
    decision_summary_df = pd.read_csv(stage4_output_dir / "decision_summary_by_method_and_cost_ratio.csv")
    decision_metrics_df = pd.read_csv(stage4_output_dir / "decision_metrics_by_horizon.csv")
    decision_thresholds_df = pd.read_csv(stage4_output_dir / "decision_one_sided_thresholds.csv")
    decision_threshold_lookup = load_decision_threshold_lookup(decision_thresholds_df)

    boundary_table = build_boundary_reliability_table(
        predict_quantiles=predict_quantiles,
        labels=labels,
        thresholds_df=thresholds_df,
        quantiles=quantiles,
        horizons=horizons,
        deltas=deltas,
        boundary_df=boundary_df,
    )
    decomposition_df = build_coverage_gain_decomposition(boundary_df)
    hard_cell_summary = build_hard_cell_summary(
        hour_positive_df,
        delta=args.primary_delta,
        top_k=args.hard_cell_top_k,
    )
    decision_attr = build_decision_value_attribution(decision_summary_df)
    decision_attr_by_horizon = build_decision_value_attribution_by_horizon(decision_metrics_df)
    cost_curve = build_cost_by_quantile_curve(
        predict_quantiles=predict_quantiles,
        labels=labels,
        quantiles=quantiles,
        cost_ratios=cost_ratios,
    )
    bootstrap_ci = build_bootstrap_ci(
        predict_quantiles=predict_quantiles,
        labels=labels,
        quantiles=quantiles,
        horizons=horizons,
        thresholds_df=thresholds_df,
        decision_summary_df=decision_summary_df,
        decision_threshold_lookup=decision_threshold_lookup,
        cost_ratios=cost_ratios,
        delta=args.primary_delta,
        n_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    boundary_table.to_csv(output_dir / "boundary_aware_reliability_table.csv", index=False)
    decomposition_df.to_csv(output_dir / "coverage_gain_decomposition.csv", index=False)
    hard_cell_summary.to_csv(output_dir / "positive_hard_cell_summary.csv", index=False)
    decision_attr.to_csv(output_dir / "decision_value_attribution.csv", index=False)
    decision_attr_by_horizon.to_csv(output_dir / "decision_value_attribution_by_horizon.csv", index=False)
    cost_curve.to_csv(output_dir / "decision_cost_by_quantile_curve.csv", index=False)
    bootstrap_ci.to_csv(output_dir / "bootstrap_ci_summary.csv", index=False)
    plot_files = write_plots(
        output_dir=output_dir,
        decomposition_df=decomposition_df,
        hard_cell_df=hour_positive_df,
        decision_attr_df=decision_attr,
        cost_curve_df=cost_curve,
        delta=args.primary_delta,
    )

    metadata_out = {
        "stage": "2.75-4.5",
        "goal": "Paper-facing boundary-aware reliability and decision-value attribution assets",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "machine": str(args.machine),
        "branch": git_value(["branch", "--show-current"]),
        "commit": git_value(["rev-parse", "--short", "HEAD"]),
        "stage1_output_dir": str(stage1_output_dir),
        "stage2_output_dir": str(stage2_output_dir),
        "stage2_diagnostics_dir": str(diagnostics_dir),
        "stage4_output_dir": str(stage4_output_dir),
        "output_dir": str(output_dir),
        "horizons": horizons,
        "quantiles": quantiles,
        "deltas": deltas,
        "primary_delta": args.primary_delta,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "max_test_windows": args.max_test_windows,
        "input_shapes": {
            "predict_quantiles": list(predict_quantiles.shape),
            "labels": list(labels.shape),
        },
        "output_files": {
            "boundary_reliability": "boundary_aware_reliability_table.csv",
            "coverage_decomposition": "coverage_gain_decomposition.csv",
            "hard_cells": "positive_hard_cell_summary.csv",
            "decision_attribution": "decision_value_attribution.csv",
            "decision_attribution_by_horizon": "decision_value_attribution_by_horizon.csv",
            "cost_curve": "decision_cost_by_quantile_curve.csv",
            "bootstrap_ci": "bootstrap_ci_summary.csv",
            "plots": plot_files,
        },
    }
    (output_dir / "paper_assets_metadata.json").write_text(
        json.dumps(metadata_out, indent=2),
        encoding="utf-8",
    )

    print(decision_attr.to_string(index=False))
    print(hard_cell_summary.head(8).to_string(index=False))
    print(json.dumps(metadata_out, indent=2))


if __name__ == "__main__":
    main()
