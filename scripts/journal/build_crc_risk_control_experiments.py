from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import utils.model_training.training_utils as fn
from scripts.journal.build_paper_evidence_assets import bootstrap_mean_ci, threshold_value
from scripts.journal.build_reinforcement_experiments import aggregate_by_date
from scripts.journal.calibrate_multihorizon_cqr import (
    build_calib_test_loaders,
    checkpoint_from_metadata,
    dataset_path_string,
    load_stage1_metadata,
    predict_loader,
    resolve_path,
)
from scripts.journal.evaluate_stage4_decision import newsvendor_cost, parse_cost_ratios
from scripts.journal.train_multihorizon_raw import parse_bool, parse_float_list, quantile_index
from utils.model_training.journal_contracts import (
    assert_monotonic_quantiles,
    assert_quantile_tensor_shape,
    assert_target_tensor_shape,
)


DEFAULT_COST_RATIOS = "3:1,5:1,9:1,19:1"
DEFAULT_ALPHA_VIOLATION = "0.05,0.10,0.20"
DEFAULT_ALPHA_SHORTAGE = "0.03,0.05,0.10"
CRC_LOSSES = ["violation", "normalized_shortage"]


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


def array_hash(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(values)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.view(np.uint8))
    return h.hexdigest()


def bounded_violation_loss(decision: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return (np.asarray(labels) > np.asarray(decision)).astype(np.float64)


def shortage_scales(labels: np.ndarray, horizons: list[int], eps: float = 1e-12) -> dict[int, float]:
    if labels.ndim != 3:
        raise AssertionError(f"labels must have shape (T, N, H), got {labels.shape}")
    scales: dict[int, float] = {}
    for h_idx, horizon in enumerate(horizons):
        values = np.asarray(labels[:, :, h_idx], dtype=np.float64)
        positive = values[values > eps]
        if positive.size == 0:
            scales[int(horizon)] = 1.0
        else:
            scales[int(horizon)] = float(max(np.quantile(positive, 0.95), eps))
    return scales


def normalized_shortage_loss(decision: np.ndarray, labels: np.ndarray, scale: float) -> np.ndarray:
    shortage = np.maximum(np.asarray(labels, dtype=np.float64) - np.asarray(decision, dtype=np.float64), 0.0)
    return np.minimum(shortage / float(scale), 1.0)


def loss_table_for_horizon(
    predict_quantiles: np.ndarray,
    labels: np.ndarray,
    candidate_indices: list[int],
    loss_name: str,
    horizon_idx: int,
    scale: float,
) -> np.ndarray:
    rows = []
    labels_h = labels[:, :, horizon_idx]
    for q_idx in candidate_indices:
        decision = np.maximum(predict_quantiles[:, :, horizon_idx, q_idx], 0.0)
        if loss_name == "violation":
            loss = bounded_violation_loss(decision, labels_h)
        elif loss_name == "normalized_shortage":
            loss = normalized_shortage_loss(decision, labels_h, scale=scale)
        else:
            raise ValueError(f"Unsupported CRC loss: {loss_name}")
        rows.append(loss.reshape(-1))
    return np.stack(rows, axis=1)


def loss_table_global(
    predict_quantiles: np.ndarray,
    labels: np.ndarray,
    candidate_indices: list[int],
    loss_name: str,
    horizons: list[int],
    scales: dict[int, float],
) -> np.ndarray:
    rows_by_horizon = []
    for h_idx, horizon in enumerate(horizons):
        rows_by_horizon.append(
            loss_table_for_horizon(
                predict_quantiles=predict_quantiles,
                labels=labels,
                candidate_indices=candidate_indices,
                loss_name=loss_name,
                horizon_idx=h_idx,
                scale=scales[int(horizon)],
            )
        )
    return np.concatenate(rows_by_horizon, axis=0)


def crc_select(loss_table: np.ndarray, alpha: float, bound: float = 1.0) -> dict[str, float | int | str]:
    if loss_table.ndim != 2:
        raise AssertionError(f"loss_table must have shape (n, candidates), got {loss_table.shape}")
    if loss_table.shape[0] == 0 or loss_table.shape[1] == 0:
        raise AssertionError("loss_table must be non-empty")
    if np.nanmin(loss_table) < -1e-12 or np.nanmax(loss_table) > bound + 1e-12:
        raise AssertionError("CRC losses must be bounded in [0, B]")

    n = int(loss_table.shape[0])
    empirical = np.mean(loss_table, axis=0)
    crc_upper = (n / (n + 1.0)) * empirical + bound / (n + 1.0)
    feasible = np.flatnonzero(crc_upper <= float(alpha) + 1e-12)
    if feasible.size == 0:
        selected = 0
        status = "no_candidate_satisfies_alpha"
    else:
        selected = int(feasible[-1])
        if feasible.size == loss_table.shape[1]:
            status = "all_candidates_satisfy_alpha"
        else:
            status = "interior_candidate_selected"
    return {
        "candidate_rank": int(selected),
        "empirical_risk": float(empirical[selected]),
        "crc_upper": float(crc_upper[selected]),
        "selection_status": status,
        "n_calibration_samples": n,
        "feasible_candidate_count": int(feasible.size),
        "min_crc_upper": float(np.min(crc_upper)),
        "max_crc_upper": float(np.max(crc_upper)),
    }


def candidate_quantile_indices(quantiles: list[float]) -> list[int]:
    return sorted(range(len(quantiles)), key=lambda idx: quantiles[idx], reverse=True)


def build_crc_selection_table(
    calib_quantiles: np.ndarray,
    calib_labels: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    alphas_violation: list[float],
    alphas_shortage: list[float],
    scales: dict[int, float],
) -> pd.DataFrame:
    candidate_indices = candidate_quantile_indices(quantiles)
    rows: list[dict[str, float | int | str]] = []
    alpha_by_loss = {"violation": alphas_violation, "normalized_shortage": alphas_shortage}

    for loss_name in CRC_LOSSES:
        for alpha in alpha_by_loss[loss_name]:
            global_table = loss_table_global(
                predict_quantiles=calib_quantiles,
                labels=calib_labels,
                candidate_indices=candidate_indices,
                loss_name=loss_name,
                horizons=horizons,
                scales=scales,
            )
            global_selected = crc_select(global_table, alpha=alpha)
            global_rank = int(global_selected["candidate_rank"])
            global_q_idx = candidate_indices[global_rank]
            rows.append(
                {
                    "crc_scope": "global",
                    "loss_name": loss_name,
                    "alpha": float(alpha),
                    "horizon": "all",
                    "selected_quantile": float(quantiles[global_q_idx]),
                    "selected_quantile_index": int(global_q_idx),
                    **global_selected,
                }
            )

            for h_idx, horizon in enumerate(horizons):
                loss_table = loss_table_for_horizon(
                    predict_quantiles=calib_quantiles,
                    labels=calib_labels,
                    candidate_indices=candidate_indices,
                    loss_name=loss_name,
                    horizon_idx=h_idx,
                    scale=scales[int(horizon)],
                )
                selected = crc_select(loss_table, alpha=alpha)
                rank = int(selected["candidate_rank"])
                q_idx = candidate_indices[rank]
                rows.append(
                    {
                        "crc_scope": "horizon",
                        "loss_name": loss_name,
                        "alpha": float(alpha),
                        "horizon": int(horizon),
                        "selected_quantile": float(quantiles[q_idx]),
                        "selected_quantile_index": int(q_idx),
                        **selected,
                    }
                )
    return pd.DataFrame(rows)


def selection_lookup(selection_df: pd.DataFrame, scope: str, loss_name: str, alpha: float) -> dict[int | str, int]:
    subset = selection_df[
        (selection_df["crc_scope"] == scope)
        & (selection_df["loss_name"] == loss_name)
        & np.isclose(selection_df["alpha"].astype(float), float(alpha))
    ]
    if subset.empty:
        raise ValueError(f"Missing CRC selection for scope={scope}, loss={loss_name}, alpha={alpha}")
    out: dict[int | str, int] = {}
    for _, row in subset.iterrows():
        horizon = row["horizon"]
        key: int | str = "all" if str(horizon) == "all" else int(horizon)
        out[key] = int(row["selected_quantile_index"])
    return out


def cqr_global_threshold(thresholds_df: pd.DataFrame, delta: float = 0.1) -> float:
    rows = thresholds_df[
        (thresholds_df["method"] == "global_cqr")
        & np.isclose(thresholds_df["delta"].astype(float), float(delta))
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one global CQR threshold for delta={delta}, got {len(rows)}")
    return float(rows.iloc[0]["s_hat"])


def one_sided_threshold(decision_thresholds_df: pd.DataFrame, cost_ratio: str, horizon: int) -> float:
    rows = decision_thresholds_df[
        (decision_thresholds_df["method"] == "horizon_cqr")
        & (decision_thresholds_df["cost_ratio"].astype(str) == str(cost_ratio))
        & (decision_thresholds_df["horizon"].astype(str) == str(horizon))
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one one-sided threshold for cost_ratio={cost_ratio}, horizon={horizon}; got {len(rows)}")
    return float(rows.iloc[0]["s_hat"])


def build_policy_decisions(
    test_quantiles: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    selection_df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
    decision_thresholds_df: pd.DataFrame,
    cost_ratios: list,
) -> dict[str, dict[str, object]]:
    policies: dict[str, dict[str, object]] = {}
    q50_idx = quantile_index(quantiles, 0.5)
    q95_idx = quantile_index(quantiles, 0.95)
    policies["median_decision"] = {
        "policy_type": "baseline",
        "decision": np.maximum(test_quantiles[..., q50_idx], 0.0),
    }

    global_s = cqr_global_threshold(thresholds_df, delta=0.1)
    policies["symmetric_90_upper"] = {
        "policy_type": "baseline",
        "decision": np.maximum(test_quantiles[..., q95_idx] + global_s, 0.0),
    }

    for ratio in cost_ratios:
        target_idx = quantile_index(quantiles, ratio.tau_star)
        policies[f"raw_cost_aligned_quantile_{ratio.label}"] = {
            "policy_type": "baseline",
            "cost_ratio_label": ratio.label,
            "decision": np.maximum(test_quantiles[..., target_idx], 0.0),
        }
        refined = np.empty(test_quantiles.shape[:3], dtype=np.float64)
        for h_idx, horizon in enumerate(horizons):
            s_hat = one_sided_threshold(decision_thresholds_df, cost_ratio=ratio.label, horizon=int(horizon))
            refined[:, :, h_idx] = np.maximum(test_quantiles[:, :, h_idx, target_idx] + s_hat, 0.0)
        policies[f"one_sided_refined_cost_aligned_quantile_{ratio.label}"] = {
            "policy_type": "baseline",
            "cost_ratio_label": ratio.label,
            "decision": refined,
        }

    for _, row in selection_df.iterrows():
        loss_name = str(row["loss_name"])
        alpha = float(row["alpha"])
        scope = str(row["crc_scope"])
        if scope != "global":
            continue

        global_idx = int(row["selected_quantile_index"])
        decision = np.maximum(test_quantiles[..., global_idx], 0.0)
        name = f"global_crc_{loss_name}_alpha{alpha:g}"
        policies[name] = {
            "policy_type": "crc",
            "crc_scope": "global",
            "loss_name": loss_name,
            "alpha": alpha,
            "decision": decision,
        }

        horizon_lookup = selection_lookup(selection_df, scope="horizon", loss_name=loss_name, alpha=alpha)
        horizon_decision = np.empty(test_quantiles.shape[:3], dtype=np.float64)
        for h_idx, horizon in enumerate(horizons):
            q_idx = int(horizon_lookup[int(horizon)])
            horizon_decision[:, :, h_idx] = np.maximum(test_quantiles[:, :, h_idx, q_idx], 0.0)
        horizon_name = f"horizon_crc_{loss_name}_alpha{alpha:g}"
        policies[horizon_name] = {
            "policy_type": "crc",
            "crc_scope": "horizon",
            "loss_name": loss_name,
            "alpha": alpha,
            "decision": horizon_decision,
        }
    return policies


def cvar95(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return float("nan")
    threshold = float(np.quantile(flat, 0.95))
    tail = flat[flat >= threshold]
    return float(np.mean(tail)) if tail.size else 0.0


def risk_resource_metrics(decision: np.ndarray, labels: np.ndarray, scales: dict[int, float], horizons: list[int]) -> dict[str, float]:
    decision = np.asarray(decision, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    shortage = np.maximum(labels - decision, 0.0)
    overage = np.maximum(decision - labels, 0.0)
    scale_arr = np.asarray([scales[int(h)] for h in horizons], dtype=np.float64).reshape(1, 1, -1)
    normalized_shortage = np.minimum(shortage / scale_arr, 1.0)
    return {
        "test_violation_rate": float(np.mean(labels > decision)),
        "test_normalized_shortage_risk": float(np.mean(normalized_shortage)),
        "shortage_frequency": float(np.mean(shortage > 0.0)),
        "mean_shortage": float(np.mean(shortage)),
        "cvar95_shortage": cvar95(shortage),
        "mean_decision_level": float(np.mean(decision)),
        "mean_overage": float(np.mean(overage)),
        "mean_absolute_overage": float(np.mean(np.abs(overage))),
        "mean_label": float(np.mean(labels)),
    }


def build_test_summaries(
    policies: dict[str, dict[str, object]],
    labels: np.ndarray,
    scales: dict[int, float],
    horizons: list[int],
    cost_ratios: list,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, float | str]] = []
    horizon_rows: list[dict[str, float | int | str]] = []
    cost_rows: list[dict[str, float | str]] = []
    for policy_name, info in policies.items():
        decision = np.asarray(info["decision"], dtype=np.float64)
        base = {
            "policy": policy_name,
            "policy_type": str(info.get("policy_type", "")),
            "crc_scope": str(info.get("crc_scope", "")),
            "loss_name": str(info.get("loss_name", "")),
            "alpha": info.get("alpha", ""),
            "cost_ratio_label": str(info.get("cost_ratio_label", "")),
        }
        summary_rows.append({**base, **risk_resource_metrics(decision, labels, scales=scales, horizons=horizons)})
        for h_idx, horizon in enumerate(horizons):
            horizon_scales = {int(horizon): scales[int(horizon)]}
            metrics = risk_resource_metrics(
                decision[:, :, h_idx : h_idx + 1],
                labels[:, :, h_idx : h_idx + 1],
                scales=horizon_scales,
                horizons=[int(horizon)],
            )
            horizon_rows.append({**base, "horizon": int(horizon), **metrics})
        for ratio in cost_ratios:
            cost = newsvendor_cost(decision, labels, c_u=ratio.c_u, c_o=ratio.c_o)
            cost_rows.append(
                {
                    **base,
                    "evaluation_cost_ratio": ratio.label,
                    "c_u": float(ratio.c_u),
                    "c_o": float(ratio.c_o),
                    "tau_star": float(ratio.tau_star),
                    "expected_cost": float(np.mean(cost)),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(horizon_rows), pd.DataFrame(cost_rows)


def build_risk_efficiency_frontier(summary_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "policy",
        "policy_type",
        "crc_scope",
        "loss_name",
        "alpha",
        "test_violation_rate",
        "test_normalized_shortage_risk",
        "mean_decision_level",
        "mean_overage",
        "cvar95_shortage",
    ]
    return summary_df[cols].sort_values(["policy_type", "loss_name", "alpha", "crc_scope", "policy"])


def date_keys_from_manifest(manifest_df: pd.DataFrame, n_test_windows: int, horizons: list[int]) -> np.ndarray:
    required = {"sample_index", "horizon", "target_date"}
    missing = sorted(required - set(manifest_df.columns))
    if missing:
        raise ValueError(f"test_window_manifest.csv is missing columns: {missing}")
    date_keys = np.empty((n_test_windows, len(horizons)), dtype=object)
    lookup = {
        (int(row.sample_index), int(row.horizon)): str(row.target_date)
        for row in manifest_df[["sample_index", "horizon", "target_date"]].itertuples(index=False)
    }
    for sample_idx in range(n_test_windows):
        for h_idx, horizon in enumerate(horizons):
            key = (sample_idx, int(horizon))
            if key not in lookup:
                raise ValueError(f"Missing manifest entry for sample_index={sample_idx}, horizon={horizon}")
            date_keys[sample_idx, h_idx] = lookup[key]
    return date_keys


def per_day_values(values: np.ndarray, date_keys: np.ndarray) -> np.ndarray:
    _, out = aggregate_by_date(np.transpose(values, (0, 2, 1)), date_keys)
    return out


def build_day_bootstrap_ci(
    policies: dict[str, dict[str, object]],
    labels: np.ndarray,
    scales: dict[int, float],
    horizons: list[int],
    cost_ratios: list,
    date_keys: np.ndarray,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    scale_arr = np.asarray([scales[int(h)] for h in horizons], dtype=np.float64).reshape(1, 1, -1)
    symmetric_decision = np.asarray(policies["symmetric_90_upper"]["decision"], dtype=np.float64)
    symmetric_overage = np.maximum(symmetric_decision - labels, 0.0)

    for policy_name, info in policies.items():
        decision = np.asarray(info["decision"], dtype=np.float64)
        shortage = np.maximum(labels - decision, 0.0)
        overage = np.maximum(decision - labels, 0.0)
        metric_arrays = {
            "test_violation_rate": (labels > decision).astype(np.float64),
            "test_normalized_shortage_risk": np.minimum(shortage / scale_arr, 1.0),
            "mean_overage": overage,
            "mean_decision_level": decision,
        }
        if str(info.get("policy_type")) == "crc":
            metric_arrays["overage_reduction_vs_symmetric_90_upper"] = symmetric_overage - overage
        for metric, values in metric_arrays.items():
            day_values = per_day_values(values, date_keys)
            point, low, high = bootstrap_mean_ci(day_values, rng=rng, n_samples=n_samples)
            rows.append(
                {
                    "policy": policy_name,
                    "metric": metric,
                    "evaluation_cost_ratio": "",
                    "point": point,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_level": "day",
                    "bootstrap_units": int(day_values.size),
                    "bootstrap_samples": int(n_samples),
                }
            )

        for ratio in cost_ratios:
            cost = newsvendor_cost(decision, labels, c_u=ratio.c_u, c_o=ratio.c_o)
            day_values = per_day_values(cost, date_keys)
            point, low, high = bootstrap_mean_ci(day_values, rng=rng, n_samples=n_samples)
            rows.append(
                {
                    "policy": policy_name,
                    "metric": "expected_cost",
                    "evaluation_cost_ratio": ratio.label,
                    "point": point,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_level": "day",
                    "bootstrap_units": int(day_values.size),
                    "bootstrap_samples": int(n_samples),
                }
            )
    return pd.DataFrame(rows)


def write_crc_plots(
    selection_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    import matplotlib.pyplot as plt

    written: list[str] = []
    crc_summary = summary_df[summary_df["policy_type"] == "crc"].copy()
    if not crc_summary.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for loss_name, group in crc_summary.groupby("loss_name"):
            horizon_group = group[group["crc_scope"] == "horizon"].sort_values("alpha")
            ax.plot(
                horizon_group["alpha"].astype(float),
                horizon_group["test_normalized_shortage_risk"].astype(float),
                marker="o",
                label=f"{loss_name} horizon CRC",
            )
        ax.set_xlabel("Target alpha")
        ax.set_ylabel("Test normalized shortage risk")
        ax.set_title("CRC test risk vs target alpha")
        ax.legend()
        path = output_dir / "crc_test_risk_vs_alpha.png"
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path.name)

        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        plot_sel = selection_df[selection_df["crc_scope"] == "horizon"].copy()
        for loss_name, group in plot_sel.groupby("loss_name"):
            pooled = group.groupby("alpha", as_index=False)["selected_quantile"].mean()
            ax.plot(pooled["alpha"], pooled["selected_quantile"], marker="o", label=loss_name)
        ax.set_xlabel("Target alpha")
        ax.set_ylabel("Mean selected quantile across horizons")
        ax.set_title("CRC selected quantiles by alpha")
        ax.legend()
        path = output_dir / "crc_selected_quantiles_by_alpha.png"
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path.name)

        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        ax.scatter(
            summary_df["mean_decision_level"].astype(float),
            summary_df["test_normalized_shortage_risk"].astype(float),
            s=22,
        )
        ax.set_xlabel("Mean decision level")
        ax.set_ylabel("Test normalized shortage risk")
        ax.set_title("Risk-efficiency frontier")
        path = output_dir / "crc_risk_efficiency_frontier.png"
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path.name)

        cost_plot = cost_df[cost_df["evaluation_cost_ratio"].isin(["3:1", "5:1", "9:1", "19:1"])].copy()
        cost_plot = cost_plot[cost_plot["policy"].isin(["median_decision", "symmetric_90_upper"]) | (cost_plot["policy_type"] == "crc")]
        fig, ax = plt.subplots(figsize=(10.2, 5.0))
        labels = []
        values = []
        for _, row in cost_plot[cost_plot["evaluation_cost_ratio"] == "5:1"].iterrows():
            labels.append(str(row["policy"]))
            values.append(float(row["expected_cost"]))
        ax.bar(range(len(values)), values)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Expected cost under 5:1")
        ax.set_title("CRC expected cost by policy")
        path = output_dir / "crc_expected_cost_by_policy.png"
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path.name)

        fig, ax = plt.subplots(figsize=(10.2, 5.0))
        plot = summary_df[summary_df["policy"].isin(["median_decision", "symmetric_90_upper"]) | (summary_df["policy_type"] == "crc")]
        ax.bar(range(len(plot)), plot["cvar95_shortage"].astype(float))
        ax.set_xticks(range(len(plot)))
        ax.set_xticklabels(plot["policy"].astype(str), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("CVaR95 shortage")
        ax.set_title("Shortage tail risk by policy")
        path = output_dir / "crc_shortage_cvar_by_policy.png"
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path.name)
    return written


def write_report(output_dir: Path, metadata: dict, selection_df: pd.DataFrame, summary_df: pd.DataFrame, cost_df: pd.DataFrame) -> None:
    crc = summary_df[summary_df["policy_type"] == "crc"].copy()
    symmetric = summary_df[summary_df["policy"] == "symmetric_90_upper"].iloc[0]
    best_crc = crc.sort_values("test_normalized_shortage_risk").iloc[0] if not crc.empty else None
    lines = [
        "# CRC Risk-Control Experiment Report",
        "",
        f"- Created at: `{metadata['created_at']}`",
        f"- Machine: `{metadata['machine']}`",
        f"- Branch: `{metadata['branch']}`",
        f"- Commit: `{metadata['commit']}`",
        "",
        "## No-Leakage Setup",
        "- Calibration predictions were reconstructed from the Stage 1 checkpoint and calibration split.",
        "- Test evaluation used saved Stage 1 test arrays.",
        "- CRC selections were made on calibration data only.",
        "",
        "## Selection Summary",
        selection_df[["crc_scope", "loss_name", "alpha", "horizon", "selected_quantile", "crc_upper", "selection_status"]]
        .head(30)
        .to_string(index=False),
        "",
        "## Main Test Summary",
        summary_df[["policy", "test_violation_rate", "test_normalized_shortage_risk", "mean_decision_level", "mean_overage", "cvar95_shortage"]]
        .head(25)
        .to_string(index=False),
        "",
        "## Interpretation Notes",
        f"- Symmetric 90 upper normalized shortage risk: `{float(symmetric['test_normalized_shortage_risk']):.6f}`.",
    ]
    if best_crc is not None:
        lines.append(
            f"- Lowest CRC normalized shortage risk policy: `{best_crc['policy']}` with risk `{float(best_crc['test_normalized_shortage_risk']):.6f}`."
        )
    cost_5 = cost_df[cost_df["evaluation_cost_ratio"] == "5:1"].sort_values("expected_cost").head(10)
    lines.extend(["", "## Lowest Expected Cost Under 5:1", cost_5[["policy", "expected_cost"]].to_string(index=False), ""])
    (output_dir / "crc_experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def load_calibration_predictions(
    stage1_output_dir: Path,
    metadata: dict,
    use_cuda: bool,
    batch_size_arg: int,
    max_calib_batches: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    checkpoint_path = checkpoint_from_metadata(stage1_output_dir, metadata)
    horizons = [int(h) for h in metadata["horizons"]]
    quantiles = [float(q) for q in metadata["quantiles"]]
    seq_len = int(metadata["seq_len"])
    split_rates = {key: float(value) for key, value in metadata["split_rates"].items()}
    batch_size = int(batch_size_arg or metadata["batch_size"])
    model_name = str(metadata["model_name"])
    data_dir = resolve_path(str(metadata["data_dir"]))
    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
    occ, duration, price_raw, distance, cap = fn.read_dataset_v2(dataset_path_string(data_dir))
    input_series = duration if "dura" in model_name else occ
    calib_loader, _, split_lengths = build_calib_test_loaders(
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
        max_batches=max_calib_batches,
        context="crc calibration",
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    manifest = {
        "checkpoint_path": str(checkpoint_path),
        "data_dir": str(data_dir),
        "device": str(device),
        "batch_size": int(batch_size),
        "max_calib_batches": int(max_calib_batches),
        "split_rates": split_rates,
        "split_lengths": split_lengths,
        "calibration_source": "reconstructed_from_stage1_checkpoint_and_calibration_split",
    }
    return calib_quantiles, calib_labels, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="CRC risk-controlled quantile deployment for the journal branch.")
    parser.add_argument("--stage1-output-dir", default="journal_results/shenzhen_multihorizon/warmstart_raw")
    parser.add_argument("--stage2-output-dir", default="journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw")
    parser.add_argument("--stage4-output-dir", default="journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw")
    parser.add_argument("--reinforcement-dir", default="journal_results/shenzhen_multihorizon/reinforcement")
    parser.add_argument("--paper-assets-dir", default="journal_results/shenzhen_multihorizon/paper_assets")
    parser.add_argument("--output-dir", default="journal_results/shenzhen_multihorizon/crc_risk_control")
    parser.add_argument("--alphas-violation", default=DEFAULT_ALPHA_VIOLATION)
    parser.add_argument("--alphas-shortage", default=DEFAULT_ALPHA_SHORTAGE)
    parser.add_argument("--cost-ratios", default=DEFAULT_COST_RATIOS)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--bootstrap-seed", type=int, default=20260520)
    parser.add_argument("--use-cuda", type=parse_bool, default=True)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--max-calib-batches", type=int, default=0)
    parser.add_argument("--max-test-windows", type=int, default=0)
    parser.add_argument("--save-calib-arrays", type=parse_bool, default=False)
    parser.add_argument("--machine", default="Lenovo")
    args = parser.parse_args()

    stage1_output_dir = resolve_path(args.stage1_output_dir)
    stage2_output_dir = resolve_path(args.stage2_output_dir)
    stage4_output_dir = resolve_path(args.stage4_output_dir)
    reinforcement_dir = resolve_path(args.reinforcement_dir)
    paper_assets_dir = resolve_path(args.paper_assets_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_stage1_metadata(stage1_output_dir)
    horizons = [int(h) for h in metadata["horizons"]]
    quantiles = [float(q) for q in metadata["quantiles"]]
    alphas_violation = parse_float_list(args.alphas_violation)
    alphas_shortage = parse_float_list(args.alphas_shortage)
    cost_ratios = parse_cost_ratios(args.cost_ratios)

    calib_quantiles, calib_labels, calib_manifest = load_calibration_predictions(
        stage1_output_dir=stage1_output_dir,
        metadata=metadata,
        use_cuda=bool(args.use_cuda),
        batch_size_arg=int(args.batch_size),
        max_calib_batches=int(args.max_calib_batches),
    )
    assert_quantile_tensor_shape(calib_quantiles, len(horizons), len(quantiles), context="crc calibration quantiles")
    assert_target_tensor_shape(calib_labels, len(horizons), context="crc calibration labels")
    assert_monotonic_quantiles(calib_quantiles, context="crc calibration quantiles")

    test_quantiles = np.load(stage1_output_dir / "predict_quantiles.npy")
    test_labels = np.load(stage1_output_dir / "label_list.npy")
    if args.max_test_windows > 0:
        test_quantiles = test_quantiles[: args.max_test_windows]
        test_labels = test_labels[: args.max_test_windows]
    assert_quantile_tensor_shape(test_quantiles, len(horizons), len(quantiles), context="crc test quantiles")
    assert_target_tensor_shape(test_labels, len(horizons), context="crc test labels")
    assert_monotonic_quantiles(test_quantiles, context="crc test quantiles")

    scales = shortage_scales(calib_labels, horizons)
    selection_df = build_crc_selection_table(
        calib_quantiles=calib_quantiles,
        calib_labels=calib_labels,
        quantiles=quantiles,
        horizons=horizons,
        alphas_violation=alphas_violation,
        alphas_shortage=alphas_shortage,
        scales=scales,
    )

    thresholds_df = pd.read_csv(stage2_output_dir / "cqr_thresholds.csv")
    decision_thresholds_df = pd.read_csv(stage4_output_dir / "decision_one_sided_thresholds.csv")
    policies = build_policy_decisions(
        test_quantiles=test_quantiles,
        quantiles=quantiles,
        horizons=horizons,
        selection_df=selection_df,
        thresholds_df=thresholds_df,
        decision_thresholds_df=decision_thresholds_df,
        cost_ratios=cost_ratios,
    )
    summary_df, by_horizon_df, cost_df = build_test_summaries(
        policies=policies,
        labels=test_labels,
        scales=scales,
        horizons=horizons,
        cost_ratios=cost_ratios,
    )
    frontier_df = build_risk_efficiency_frontier(summary_df)
    baseline_df = summary_df.copy()

    manifest_df = pd.read_csv(reinforcement_dir / "test_window_manifest.csv")
    try:
        date_keys = date_keys_from_manifest(manifest_df, n_test_windows=test_labels.shape[0], horizons=horizons)
        bootstrap_df = build_day_bootstrap_ci(
            policies=policies,
            labels=test_labels,
            scales=scales,
            horizons=horizons,
            cost_ratios=cost_ratios,
            date_keys=date_keys,
            n_samples=int(args.bootstrap_samples),
            seed=int(args.bootstrap_seed),
        )
        bootstrap_level = "day"
        bootstrap_warning = ""
    except Exception as exc:
        bootstrap_df = pd.DataFrame()
        bootstrap_level = "none"
        bootstrap_warning = str(exc)

    selection_df.to_csv(output_dir / "crc_selection_table.csv", index=False)
    summary_df.to_csv(output_dir / "crc_test_risk_summary.csv", index=False)
    by_horizon_df.to_csv(output_dir / "crc_by_horizon.csv", index=False)
    baseline_df.to_csv(output_dir / "crc_baseline_comparison.csv", index=False)
    frontier_df.to_csv(output_dir / "crc_risk_efficiency_frontier.csv", index=False)
    cost_df.to_csv(output_dir / "crc_cost_ratio_evaluation.csv", index=False)
    bootstrap_df.to_csv(output_dir / "crc_bootstrap_ci_day_level.csv", index=False)

    calib_hashes = {
        "qhat_calib_sha256": array_hash(calib_quantiles),
        "y_calib_sha256": array_hash(calib_labels),
        "qhat_test_sha256": array_hash(test_quantiles),
        "y_test_sha256": array_hash(test_labels),
    }
    (output_dir / "crc_calibration_hashes.json").write_text(json.dumps(calib_hashes, indent=2), encoding="utf-8")

    calib_manifest.update(
        {
            "no_leakage_rule": "CRC selection uses calibration predictions only; final metrics use saved test arrays only.",
            "calib_quantiles_shape": list(calib_quantiles.shape),
            "calib_labels_shape": list(calib_labels.shape),
            "test_quantiles_shape": list(test_quantiles.shape),
            "test_labels_shape": list(test_labels.shape),
            "shortage_scales_by_horizon": {str(k): v for k, v in scales.items()},
            "calibration_arrays_saved": bool(args.save_calib_arrays),
        }
    )
    (output_dir / "crc_calibration_manifest.json").write_text(json.dumps(calib_manifest, indent=2), encoding="utf-8")
    if args.save_calib_arrays:
        np.save(output_dir / "qhat_calib.npy", calib_quantiles)
        np.save(output_dir / "y_calib.npy", calib_labels)

    plots = write_crc_plots(selection_df, summary_df, cost_df, output_dir)
    metadata_out = {
        "stage": "crc_risk_control",
        "goal": "Conformal risk-controlled quantile deployment under service-risk targets",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "machine": args.machine,
        "branch": git_value(["branch", "--show-current"]),
        "commit": git_value(["rev-parse", "--short", "HEAD"]),
        "stage1_output_dir": str(stage1_output_dir),
        "stage2_output_dir": str(stage2_output_dir),
        "stage4_output_dir": str(stage4_output_dir),
        "reinforcement_dir": str(reinforcement_dir),
        "paper_assets_dir": str(paper_assets_dir),
        "output_dir": str(output_dir),
        "horizons": horizons,
        "quantiles": quantiles,
        "losses": CRC_LOSSES,
        "alphas_violation": alphas_violation,
        "alphas_shortage": alphas_shortage,
        "cost_ratios": [ratio.__dict__ for ratio in cost_ratios],
        "bootstrap_level": bootstrap_level,
        "bootstrap_warning": bootstrap_warning,
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "weighted_loss_implemented": False,
        "safety_margin_baseline_implemented": False,
        "global_crc_implemented": True,
        "no_leakage": {
            "calibration_source": "reconstructed_from_stage1_checkpoint_and_calibration_split",
            "test_source": "saved_stage1_test_arrays",
            "selection_uses_test_labels": False,
        },
        "input_shapes": {
            "calib_quantiles": list(calib_quantiles.shape),
            "calib_labels": list(calib_labels.shape),
            "test_quantiles": list(test_quantiles.shape),
            "test_labels": list(test_labels.shape),
        },
        "outputs": {
            "selection_table": "crc_selection_table.csv",
            "test_risk_summary": "crc_test_risk_summary.csv",
            "by_horizon": "crc_by_horizon.csv",
            "baseline_comparison": "crc_baseline_comparison.csv",
            "risk_efficiency_frontier": "crc_risk_efficiency_frontier.csv",
            "cost_ratio_evaluation": "crc_cost_ratio_evaluation.csv",
            "bootstrap_ci": "crc_bootstrap_ci_day_level.csv",
            "calibration_manifest": "crc_calibration_manifest.json",
            "calibration_hashes": "crc_calibration_hashes.json",
            "plots": plots,
            "report": "crc_experiment_report.md",
        },
    }
    (output_dir / "crc_metadata.json").write_text(json.dumps(metadata_out, indent=2), encoding="utf-8")
    write_report(output_dir, metadata_out, selection_df, summary_df, cost_df)
    print(json.dumps(metadata_out, indent=2))


if __name__ == "__main__":
    main()
