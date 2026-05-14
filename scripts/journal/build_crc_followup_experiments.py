from __future__ import annotations

import argparse
import gc
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

from scripts.journal.build_crc_risk_control_experiments import (
    array_hash,
    bounded_violation_loss,
    candidate_quantile_indices,
    crc_select,
    cvar95,
    date_keys_from_manifest,
    load_calibration_predictions,
    normalized_shortage_loss,
    per_day_values,
    risk_resource_metrics,
    shortage_scales,
)
from scripts.journal.build_paper_evidence_assets import bootstrap_mean_ci
from scripts.journal.calibrate_multihorizon_cqr import load_stage1_metadata, resolve_path
from scripts.journal.evaluate_stage4_decision import newsvendor_cost, parse_cost_ratios
from scripts.journal.train_multihorizon_raw import parse_bool, parse_float_list, quantile_index
from utils.model_training.journal_contracts import (
    assert_monotonic_quantiles,
    assert_quantile_tensor_shape,
    assert_target_tensor_shape,
)


MARGIN_QUANTILES = [0.0, 0.25, 0.50, 0.75, 0.80, 0.833333, 0.90, 0.95, 0.975, 0.99]
DEFAULT_ALPHAS = "0.05,0.10,0.20"
DEFAULT_H1_ALPHAS = "0.03,0.04,0.05,0.06,0.08,0.10"
DEFAULT_COST_RATIOS = "3:1,5:1,9:1,19:1"


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


def margin_candidates(residuals: np.ndarray, quantile_grid: list[float] = MARGIN_QUANTILES) -> tuple[np.ndarray, list[float]]:
    values = np.asarray(residuals, dtype=np.float64).reshape(-1)
    candidates = []
    sources = []
    for q in quantile_grid:
        value = float(np.quantile(values, q))
        if not any(abs(value - existing) < 1e-10 for existing in candidates):
            candidates.append(value)
            sources.append(float(q))
    order = np.argsort(candidates)
    return np.asarray(candidates, dtype=np.float64)[order], [sources[int(i)] for i in order]


def select_smallest_feasible(loss_table: np.ndarray, alpha: float, rule: str) -> dict[str, float | int | str]:
    if loss_table.ndim != 2:
        raise AssertionError(f"loss_table must be 2D, got {loss_table.shape}")
    empirical = np.mean(loss_table, axis=0)
    n = int(loss_table.shape[0])
    if rule == "empirical_risk":
        bounds = empirical
    elif rule == "crc_corrected":
        bounds = (n / (n + 1.0)) * empirical + 1.0 / (n + 1.0)
    else:
        raise ValueError(f"Unknown selection rule: {rule}")
    feasible = np.flatnonzero(bounds <= float(alpha) + 1e-12)
    if feasible.size == 0:
        idx = int(loss_table.shape[1] - 1)
        status = "no_candidate_satisfies_alpha"
    else:
        idx = int(feasible[0])
        status = "all_candidates_satisfy_alpha" if feasible.size == loss_table.shape[1] else "interior_candidate_selected"
    return {
        "candidate_index": idx,
        "empirical_risk": float(empirical[idx]),
        "risk_bound": float(bounds[idx]),
        "selection_status": status,
        "feasible_candidate_count": int(feasible.size),
        "n_calibration_samples": n,
    }


def safety_margin_loss_table(q50: np.ndarray, labels: np.ndarray, margins: np.ndarray) -> np.ndarray:
    rows = []
    for margin in margins:
        decision = q50 + float(margin)
        rows.append(bounded_violation_loss(decision, labels).reshape(-1))
    return np.stack(rows, axis=1)


def build_safety_margin_selection(
    calib_quantiles: np.ndarray,
    calib_labels: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    alphas: list[float],
) -> pd.DataFrame:
    q50_idx = quantile_index(quantiles, 0.5)
    q50 = calib_quantiles[..., q50_idx]
    residual = np.maximum(calib_labels - q50, 0.0)
    rows: list[dict[str, float | int | str]] = []

    global_margins, global_sources = margin_candidates(residual)
    global_loss_table = safety_margin_loss_table(q50, calib_labels, global_margins)
    for alpha in alphas:
        for rule in ["empirical_risk", "crc_corrected"]:
            selected = select_smallest_feasible(global_loss_table, alpha=alpha, rule=rule)
            idx = int(selected["candidate_index"])
            rows.append(
                {
                    "policy": f"safety_margin_{rule}_global_alpha{alpha:g}",
                    "selection_rule": rule,
                    "alpha": float(alpha),
                    "margin_scope": "global",
                    "horizon": "all",
                    "selected_margin": float(global_margins[idx]),
                    "selected_margin_source_quantile": float(global_sources[idx]),
                    "candidate_count": int(len(global_margins)),
                    **selected,
                }
            )

    for h_idx, horizon in enumerate(horizons):
        residual_h = residual[:, :, h_idx]
        margins_h, sources_h = margin_candidates(residual_h)
        loss_table_h = safety_margin_loss_table(q50[:, :, h_idx], calib_labels[:, :, h_idx], margins_h)
        for alpha in alphas:
            for rule in ["empirical_risk", "crc_corrected"]:
                selected = select_smallest_feasible(loss_table_h, alpha=alpha, rule=rule)
                idx = int(selected["candidate_index"])
                rows.append(
                    {
                        "policy": f"safety_margin_{rule}_horizon_alpha{alpha:g}",
                        "selection_rule": rule,
                        "alpha": float(alpha),
                        "margin_scope": "horizon",
                        "horizon": int(horizon),
                        "selected_margin": float(margins_h[idx]),
                        "selected_margin_source_quantile": float(sources_h[idx]),
                        "candidate_count": int(len(margins_h)),
                        **selected,
                    }
                )
    return pd.DataFrame(rows)


def build_safety_margin_policies(
    test_quantiles: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    selection_df: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    q50_idx = quantile_index(quantiles, 0.5)
    q50 = test_quantiles[..., q50_idx]
    policies: dict[str, dict[str, object]] = {}
    for (rule, alpha), group in selection_df.groupby(["selection_rule", "alpha"], sort=True):
        global_row = group[group["margin_scope"] == "global"].iloc[0]
        global_margin = float(global_row["selected_margin"])
        policies[f"safety_margin_{rule}_global_alpha{float(alpha):g}"] = {
            "policy_family": "safety_margin",
            "selection_rule": str(rule),
            "alpha": float(alpha),
            "margin_scope": "global",
            "selected_margin": global_margin,
            "decision": np.maximum(q50 + global_margin, 0.0).astype(np.float32),
        }
        horizon_decision = np.empty(q50.shape, dtype=np.float32)
        margins = {}
        for h_idx, horizon in enumerate(horizons):
            row = group[(group["margin_scope"] == "horizon") & (group["horizon"].astype(str) == str(horizon))].iloc[0]
            margin = float(row["selected_margin"])
            margins[int(horizon)] = margin
            horizon_decision[:, :, h_idx] = np.maximum(q50[:, :, h_idx] + margin, 0.0)
        policies[f"safety_margin_{rule}_horizon_alpha{float(alpha):g}"] = {
            "policy_family": "safety_margin",
            "selection_rule": str(rule),
            "alpha": float(alpha),
            "margin_scope": "horizon",
            "selected_margin": json.dumps(margins, sort_keys=True),
            "decision": horizon_decision,
        }
    return policies


def raw_quantile_violation_loss_table(
    quantiles_array: np.ndarray,
    labels: np.ndarray,
    candidate_indices: list[int],
    horizon_idx: int | None = None,
) -> np.ndarray:
    rows = []
    if horizon_idx is None:
        labels_use = labels
        for q_idx in candidate_indices:
            rows.append(bounded_violation_loss(np.maximum(quantiles_array[..., q_idx], 0.0), labels_use).reshape(-1))
    else:
        labels_use = labels[:, :, horizon_idx]
        for q_idx in candidate_indices:
            rows.append(
                bounded_violation_loss(np.maximum(quantiles_array[:, :, horizon_idx, q_idx], 0.0), labels_use).reshape(-1)
            )
    return np.stack(rows, axis=1)


def select_largest_feasible(loss_table: np.ndarray, alpha: float, rule: str) -> dict[str, float | int | str]:
    empirical = np.mean(loss_table, axis=0)
    n = int(loss_table.shape[0])
    if rule == "empirical_risk":
        bounds = empirical
    elif rule == "crc_corrected":
        bounds = (n / (n + 1.0)) * empirical + 1.0 / (n + 1.0)
    else:
        raise ValueError(f"Unknown selector rule: {rule}")
    feasible = np.flatnonzero(bounds <= float(alpha) + 1e-12)
    if feasible.size == 0:
        idx = 0
        status = "no_candidate_satisfies_alpha"
    else:
        idx = int(feasible[-1])
        status = "all_candidates_satisfy_alpha" if feasible.size == loss_table.shape[1] else "interior_candidate_selected"
    return {
        "candidate_rank": idx,
        "empirical_risk": float(empirical[idx]),
        "risk_bound": float(bounds[idx]),
        "selection_status": status,
        "feasible_candidate_count": int(feasible.size),
        "n_calibration_samples": n,
    }


def build_empirical_vs_crc_selection(
    calib_quantiles: np.ndarray,
    calib_labels: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    alphas: list[float],
) -> pd.DataFrame:
    candidate_indices = candidate_quantile_indices(quantiles)
    rows: list[dict[str, float | int | str]] = []
    for alpha in alphas:
        for rule in ["empirical_risk", "crc_corrected"]:
            table = raw_quantile_violation_loss_table(calib_quantiles, calib_labels, candidate_indices)
            selected = select_largest_feasible(table, alpha=alpha, rule=rule)
            q_idx = candidate_indices[int(selected["candidate_rank"])]
            rows.append(
                {
                    "policy": f"raw_quantile_{rule}_global_alpha{alpha:g}",
                    "selector_rule": rule,
                    "alpha": float(alpha),
                    "selector_scope": "global",
                    "horizon": "all",
                    "selected_quantile": float(quantiles[q_idx]),
                    "selected_quantile_index": int(q_idx),
                    **selected,
                }
            )
            for h_idx, horizon in enumerate(horizons):
                table_h = raw_quantile_violation_loss_table(calib_quantiles, calib_labels, candidate_indices, horizon_idx=h_idx)
                selected_h = select_largest_feasible(table_h, alpha=alpha, rule=rule)
                q_idx_h = candidate_indices[int(selected_h["candidate_rank"])]
                rows.append(
                    {
                        "policy": f"raw_quantile_{rule}_horizon_alpha{alpha:g}",
                        "selector_rule": rule,
                        "alpha": float(alpha),
                        "selector_scope": "horizon",
                        "horizon": int(horizon),
                        "selected_quantile": float(quantiles[q_idx_h]),
                        "selected_quantile_index": int(q_idx_h),
                        **selected_h,
                    }
                )
    return pd.DataFrame(rows)


def build_raw_selector_policies(
    test_quantiles: np.ndarray,
    horizons: list[int],
    selection_df: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    policies: dict[str, dict[str, object]] = {}
    for (rule, alpha), group in selection_df.groupby(["selector_rule", "alpha"], sort=True):
        global_row = group[group["selector_scope"] == "global"].iloc[0]
        q_idx = int(global_row["selected_quantile_index"])
        policies[f"raw_quantile_{rule}_global_alpha{float(alpha):g}"] = {
            "policy_family": "raw_selector",
            "selection_rule": str(rule),
            "alpha": float(alpha),
            "selector_scope": "global",
            "selected_quantile": float(global_row["selected_quantile"]),
            "decision": np.maximum(test_quantiles[..., q_idx], 0.0).astype(np.float32),
        }
        decision = np.empty(test_quantiles.shape[:3], dtype=np.float32)
        selected_by_h = {}
        for h_idx, horizon in enumerate(horizons):
            row = group[(group["selector_scope"] == "horizon") & (group["horizon"].astype(str) == str(horizon))].iloc[0]
            h_q_idx = int(row["selected_quantile_index"])
            selected_by_h[int(horizon)] = float(row["selected_quantile"])
            decision[:, :, h_idx] = np.maximum(test_quantiles[:, :, h_idx, h_q_idx], 0.0)
        policies[f"raw_quantile_{rule}_horizon_alpha{float(alpha):g}"] = {
            "policy_family": "raw_selector",
            "selection_rule": str(rule),
            "alpha": float(alpha),
            "selector_scope": "horizon",
            "selected_quantile": json.dumps(selected_by_h, sort_keys=True),
            "decision": decision,
        }
    return policies


def build_h1_alpha_sensitivity(
    calib_quantiles: np.ndarray,
    calib_labels: np.ndarray,
    test_quantiles: np.ndarray,
    test_labels: np.ndarray,
    quantiles: list[float],
    alphas: list[float],
    cost_ratios: list,
) -> pd.DataFrame:
    h_idx = 0
    candidate_indices = candidate_quantile_indices(quantiles)
    table = raw_quantile_violation_loss_table(calib_quantiles, calib_labels, candidate_indices, horizon_idx=h_idx)
    rows = []
    for alpha in alphas:
        selected = select_largest_feasible(table, alpha=alpha, rule="crc_corrected")
        q_idx = candidate_indices[int(selected["candidate_rank"])]
        decision = np.maximum(test_quantiles[:, :, h_idx, q_idx], 0.0)
        labels_h = test_labels[:, :, h_idx]
        shortage = np.maximum(labels_h - decision, 0.0)
        overage = np.maximum(decision - labels_h, 0.0)
        row: dict[str, float | int | str] = {
            "alpha": float(alpha),
            "selected_quantile": float(quantiles[q_idx]),
            "selected_quantile_index": int(q_idx),
            "calib_empirical_risk": float(selected["empirical_risk"]),
            "crc_upper": float(selected["risk_bound"]),
            "selection_status": str(selected["selection_status"]),
            "h1_test_violation_rate": float(np.mean(labels_h > decision)),
            "h1_mean_overage": float(np.mean(overage)),
            "h1_mean_shortage": float(np.mean(shortage)),
            "h1_cvar95_shortage": cvar95(shortage),
        }
        for ratio in cost_ratios:
            row[f"expected_cost_{ratio.label.replace(':', '_')}"] = float(
                np.mean(newsvendor_cost(decision, labels_h, c_u=ratio.c_u, c_o=ratio.c_o))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def block_loss_table(
    quantiles_array: np.ndarray,
    labels: np.ndarray,
    candidate_indices: list[int],
    horizon_idx: int,
) -> np.ndarray:
    rows = []
    labels_h = labels[:, :, horizon_idx]
    for q_idx in candidate_indices:
        loss = bounded_violation_loss(np.maximum(quantiles_array[:, :, horizon_idx, q_idx], 0.0), labels_h)
        rows.append(np.mean(loss, axis=1))
    return np.stack(rows, axis=1)


def build_block_crc_selection(
    calib_quantiles: np.ndarray,
    calib_labels: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    alphas: list[float],
) -> pd.DataFrame:
    candidate_indices = candidate_quantile_indices(quantiles)
    rows = []
    for alpha in alphas:
        for h_idx, horizon in enumerate(horizons):
            table = block_loss_table(calib_quantiles, calib_labels, candidate_indices, horizon_idx=h_idx)
            selected = crc_select(table, alpha=alpha)
            q_idx = candidate_indices[int(selected["candidate_rank"])]
            rows.append(
                {
                    "policy": f"block_crc_horizon_alpha{alpha:g}",
                    "alpha": float(alpha),
                    "horizon": int(horizon),
                    "selected_quantile": float(quantiles[q_idx]),
                    "selected_quantile_index": int(q_idx),
                    "block_definition": "calibration_window_index_mean_over_stations",
                    "num_calib_blocks": int(table.shape[0]),
                    "target_date_available": False,
                    "fallback_used": True,
                    **selected,
                }
            )
    return pd.DataFrame(rows)


def build_block_crc_policies(
    test_quantiles: np.ndarray,
    horizons: list[int],
    selection_df: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    policies = {}
    for alpha, group in selection_df.groupby("alpha", sort=True):
        decision = np.empty(test_quantiles.shape[:3], dtype=np.float32)
        selected = {}
        for h_idx, horizon in enumerate(horizons):
            row = group[group["horizon"].astype(int) == int(horizon)].iloc[0]
            q_idx = int(row["selected_quantile_index"])
            selected[int(horizon)] = float(row["selected_quantile"])
            decision[:, :, h_idx] = np.maximum(test_quantiles[:, :, h_idx, q_idx], 0.0)
        policies[f"block_crc_horizon_alpha{float(alpha):g}"] = {
            "policy_family": "block_crc",
            "alpha": float(alpha),
            "selector_scope": "horizon",
            "selected_quantile": json.dumps(selected, sort_keys=True),
            "decision": decision,
        }
    return policies


def evaluate_policy_dict(
    policies: dict[str, dict[str, object]],
    labels: np.ndarray,
    scales: dict[int, float],
    horizons: list[int],
    cost_ratios: list,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    horizon_rows = []
    cost_rows = []
    for name, info in policies.items():
        decision = np.asarray(info["decision"], dtype=np.float64)
        base = {
            "policy": name,
            "policy_family": str(info.get("policy_family", "")),
            "selection_rule": str(info.get("selection_rule", "")),
            "alpha": info.get("alpha", ""),
            "scope": str(info.get("margin_scope", info.get("selector_scope", ""))),
            "selected_value": str(info.get("selected_margin", info.get("selected_quantile", ""))),
        }
        metrics = risk_resource_metrics(decision, labels, scales=scales, horizons=horizons)
        summary_row = {**base, **metrics}
        for ratio in cost_ratios:
            summary_row[f"expected_cost_{ratio.label.replace(':', '_')}"] = float(
                np.mean(newsvendor_cost(decision, labels, c_u=ratio.c_u, c_o=ratio.c_o))
            )
            cost_rows.append(
                {
                    **base,
                    "evaluation_cost_ratio": ratio.label,
                    "expected_cost": summary_row[f"expected_cost_{ratio.label.replace(':', '_')}"],
                }
            )
        summary_rows.append(summary_row)
        for h_idx, horizon in enumerate(horizons):
            h_metrics = risk_resource_metrics(
                decision[:, :, h_idx : h_idx + 1],
                labels[:, :, h_idx : h_idx + 1],
                scales={int(horizon): scales[int(horizon)]},
                horizons=[int(horizon)],
            )
            horizon_rows.append({**base, "horizon": int(horizon), **h_metrics})
    return pd.DataFrame(summary_rows), pd.DataFrame(horizon_rows), pd.DataFrame(cost_rows)


def add_block_test_metrics(block_summary: pd.DataFrame, policies: dict[str, dict[str, object]], labels: np.ndarray) -> pd.DataFrame:
    if block_summary.empty:
        return block_summary
    out = block_summary.copy()
    block_rates = {}
    for name, info in policies.items():
        decision = np.asarray(info["decision"], dtype=np.float64)
        loss = (labels > decision).astype(np.float64)
        block_rates[name] = float(np.mean(np.mean(loss, axis=1)))
    out["test_block_violation_rate"] = out["policy"].map(block_rates)
    return out


def bootstrap_followup(
    policies: dict[str, dict[str, object]],
    labels: np.ndarray,
    scales: dict[int, float],
    horizons: list[int],
    cost_ratios: list,
    date_keys: np.ndarray,
    n_samples: int,
    seed: int,
    symmetric_decision: np.ndarray | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    symmetric_overage = None
    symmetric_cost = {}
    if symmetric_decision is not None:
        symmetric_overage = np.maximum(symmetric_decision - labels, 0.0)
        for ratio in cost_ratios:
            symmetric_cost[ratio.label] = newsvendor_cost(symmetric_decision, labels, c_u=ratio.c_u, c_o=ratio.c_o)
    for name, info in policies.items():
        decision = np.asarray(info["decision"], dtype=np.float64)
        shortage = np.maximum(labels - decision, 0.0)
        overage = np.maximum(decision - labels, 0.0)
        metric_arrays = {
            "test_violation_rate": (labels > decision).astype(np.float64),
            "mean_overage": overage,
            "mean_shortage": shortage,
        }
        if symmetric_overage is not None:
            metric_arrays["overage_reduction_vs_symmetric_90_upper"] = symmetric_overage - overage
        for metric, arr in metric_arrays.items():
            values = per_day_values(arr, date_keys)
            point, low, high = bootstrap_mean_ci(values, rng=rng, n_samples=n_samples)
            rows.append(
                {
                    "policy": name,
                    "metric": metric,
                    "evaluation_cost_ratio": "",
                    "point": point,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_units": int(values.size),
                    "bootstrap_samples": int(n_samples),
                }
            )
        for ratio in cost_ratios:
            cost = newsvendor_cost(decision, labels, c_u=ratio.c_u, c_o=ratio.c_o)
            values = per_day_values(cost, date_keys)
            point, low, high = bootstrap_mean_ci(values, rng=rng, n_samples=n_samples)
            rows.append(
                {
                    "policy": name,
                    "metric": "expected_cost",
                    "evaluation_cost_ratio": ratio.label,
                    "point": point,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_units": int(values.size),
                    "bootstrap_samples": int(n_samples),
                }
            )
            if ratio.label in symmetric_cost:
                diff = per_day_values(symmetric_cost[ratio.label] - cost, date_keys)
                point, low, high = bootstrap_mean_ci(diff, rng=rng, n_samples=n_samples)
                rows.append(
                    {
                        "policy": name,
                        "metric": "expected_cost_reduction_vs_symmetric_90_upper",
                        "evaluation_cost_ratio": ratio.label,
                        "point": point,
                        "ci_low": low,
                        "ci_high": high,
                        "bootstrap_units": int(diff.size),
                        "bootstrap_samples": int(n_samples),
                    }
                )
    return pd.DataFrame(rows)


def write_followup_plots(
    out_dir: Path,
    policy_df: pd.DataFrame,
    safety_df: pd.DataFrame,
    empirical_df: pd.DataFrame,
    h1_df: pd.DataFrame,
    block_df: pd.DataFrame,
) -> list[str]:
    import matplotlib.pyplot as plt

    written = []
    if not safety_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        plot = safety_df.sort_values(["alpha", "selection_rule", "scope"])
        ax.scatter(plot["mean_overage"].astype(float), plot["test_violation_rate"].astype(float))
        ax.set_xlabel("Mean overage")
        ax.set_ylabel("Test violation rate")
        ax.set_title("Safety-margin risk-efficiency")
        path = out_dir / "crc_safety_margin_vs_crc.png"
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path.name)

    if not empirical_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        plot = empirical_df.sort_values(["alpha", "selection_rule", "scope"])
        ax.scatter(plot["mean_overage"].astype(float), plot["test_violation_rate"].astype(float))
        for _, row in plot.iterrows():
            ax.annotate(str(row["policy"]).replace("raw_quantile_", ""), (float(row["mean_overage"]), float(row["test_violation_rate"])), fontsize=6)
        ax.set_xlabel("Mean overage")
        ax.set_ylabel("Test violation rate")
        ax.set_title("Empirical selector vs CRC-corrected selector")
        path = out_dir / "crc_empirical_vs_crc_selector.png"
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path.name)

    if not h1_df.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(h1_df["alpha"].astype(float), h1_df["h1_test_violation_rate"].astype(float), marker="o", label="H1 test violation")
        ax.plot(h1_df["alpha"].astype(float), h1_df["alpha"].astype(float), linestyle="--", label="target alpha")
        ax.set_xlabel("Alpha")
        ax.set_ylabel("H1 violation")
        ax.set_title("H1 CRC alpha sensitivity")
        ax.legend()
        path = out_dir / "crc_h1_alpha_sensitivity.png"
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path.name)

    if not block_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        plot = block_df.sort_values("alpha")
        ax.scatter(plot["mean_overage"].astype(float), plot["test_violation_rate"].astype(float))
        for _, row in plot.iterrows():
            ax.annotate(f"a={float(row['alpha']):g}", (float(row["mean_overage"]), float(row["test_violation_rate"])), fontsize=7)
        ax.set_xlabel("Mean overage")
        ax.set_ylabel("Station-window violation")
        ax.set_title("Blocked CRC vs station-window risk")
        path = out_dir / "crc_blocked_vs_station_window.png"
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path.name)

    if not policy_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(policy_df["mean_overage"].astype(float), policy_df["test_violation_rate"].astype(float), s=22)
        ax.set_xlabel("Mean overage")
        ax.set_ylabel("Test violation rate")
        ax.set_title("CRC follow-up risk-efficiency frontier")
        path = out_dir / "crc_followup_risk_efficiency_frontier.png"
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path.name)

        fig, ax = plt.subplots(figsize=(9, 5))
        top = policy_df.sort_values("expected_cost_5_1").head(25)
        ax.bar(range(len(top)), top["expected_cost_5_1"].astype(float))
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels(top["policy"].astype(str), rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Expected cost under 5:1")
        ax.set_title("CRC follow-up expected cost by policy")
        path = out_dir / "crc_followup_expected_cost_by_policy.png"
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path.name)

        fig, ax = plt.subplots(figsize=(9, 5))
        top = policy_df.sort_values("cvar95_shortage").head(25)
        ax.bar(range(len(top)), top["cvar95_shortage"].astype(float))
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels(top["policy"].astype(str), rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("CVaR95 shortage")
        ax.set_title("CRC follow-up shortage CVaR by policy")
        path = out_dir / "crc_followup_shortage_cvar_by_policy.png"
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path.name)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(policy_df["mean_overage"].astype(float), policy_df["test_violation_rate"].astype(float), s=22)
        ax.set_xlabel("Mean overage")
        ax.set_ylabel("Test violation rate")
        ax.set_title("Violation vs overage")
        path = out_dir / "crc_followup_violation_vs_overage.png"
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path.name)
    return written


def load_v1_tables(crc_v1_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = crc_v1_dir / "crc_test_risk_summary.csv"
    cost_path = crc_v1_dir / "crc_cost_ratio_evaluation.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    cost = pd.read_csv(cost_path) if cost_path.exists() else pd.DataFrame()
    if not summary.empty:
        summary = summary.copy()
        summary["policy_family"] = summary.get("policy_type", "crc_v1")
        summary["source"] = "crc_v1"
    return summary, cost


def write_report(out_dir: Path, metadata: dict, safety_summary: pd.DataFrame, empirical_summary: pd.DataFrame, h1_df: pd.DataFrame, block_summary: pd.DataFrame, combined: pd.DataFrame) -> None:
    lines = [
        "# CRC Follow-Up Experiment Report",
        "",
        f"- Created at: `{metadata['created_at']}`",
        f"- Machine: `{metadata['machine']}`",
        f"- Branch: `{metadata['branch']}`",
        f"- Commit: `{metadata['commit']}`",
        "",
        "## No-Leakage Verification",
        "- Calibration predictions were reconstructed from the Stage 1 checkpoint and calibration split.",
        "- Safety margins, empirical selectors, CRC-corrected selectors, and block CRC selections used calibration arrays only.",
        "- Test labels were used only for final evaluation.",
        "",
        "## Safety-Margin Baseline",
        safety_summary.sort_values("test_violation_rate")[["policy", "test_violation_rate", "mean_overage", "expected_cost_5_1"]].head(20).to_string(index=False),
        "",
        "## Empirical vs CRC-Corrected Selector",
        empirical_summary.sort_values(["alpha", "scope", "selection_rule"])[["policy", "test_violation_rate", "mean_overage", "expected_cost_5_1"]].head(24).to_string(index=False),
        "",
        "## H1 Alpha Sensitivity",
        h1_df.to_string(index=False),
        "",
        "## Block CRC",
        block_summary[["policy", "test_violation_rate", "test_block_violation_rate", "mean_overage", "expected_cost_5_1"]].to_string(index=False) if not block_summary.empty else "Block CRC not run.",
        "",
        "## Updated Risk-Efficiency Top Policies",
        combined.sort_values("test_violation_rate")[["policy", "source", "test_violation_rate", "mean_overage", "expected_cost_5_1"]].head(30).to_string(index=False),
        "",
    ]
    (out_dir / "crc_followup_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="CRC follow-up experiments for reviewer-defense and paper positioning.")
    parser.add_argument("--stage1-output-dir", default="journal_results/shenzhen_multihorizon/warmstart_raw")
    parser.add_argument("--stage2-output-dir", default="journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw")
    parser.add_argument("--stage4-output-dir", default="journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw")
    parser.add_argument("--reinforcement-dir", default="journal_results/shenzhen_multihorizon/reinforcement")
    parser.add_argument("--paper-assets-dir", default="journal_results/shenzhen_multihorizon/paper_assets")
    parser.add_argument("--crc-v1-dir", default="journal_results/shenzhen_multihorizon/crc_risk_control")
    parser.add_argument("--output-dir", default="journal_results/shenzhen_multihorizon/crc_risk_control_followup")
    parser.add_argument("--alphas-violation", default=DEFAULT_ALPHAS)
    parser.add_argument("--h1-alphas", default=DEFAULT_H1_ALPHAS)
    parser.add_argument("--cost-ratios", default=DEFAULT_COST_RATIOS)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--bootstrap-seed", type=int, default=20260521)
    parser.add_argument("--use-cuda", type=parse_bool, default=True)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--max-calib-batches", type=int, default=0)
    parser.add_argument("--max-test-windows", type=int, default=0)
    parser.add_argument("--enable-block-crc", type=parse_bool, default=True)
    parser.add_argument("--enable-peak-hour-crc", type=parse_bool, default=False)
    parser.add_argument("--save-calibration-arrays", type=parse_bool, default=False)
    parser.add_argument("--machine", default="Lenovo")
    args = parser.parse_args()

    stage1_output_dir = resolve_path(args.stage1_output_dir)
    stage2_output_dir = resolve_path(args.stage2_output_dir)
    reinforcement_dir = resolve_path(args.reinforcement_dir)
    crc_v1_dir = resolve_path(args.crc_v1_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_stage1_metadata(stage1_output_dir)
    horizons = [int(h) for h in metadata["horizons"]]
    quantiles = [float(q) for q in metadata["quantiles"]]
    alphas = parse_float_list(args.alphas_violation)
    h1_alphas = parse_float_list(args.h1_alphas)
    cost_ratios = parse_cost_ratios(args.cost_ratios)

    calib_quantiles, calib_labels, calib_manifest = load_calibration_predictions(
        stage1_output_dir=stage1_output_dir,
        metadata=metadata,
        use_cuda=bool(args.use_cuda),
        batch_size_arg=int(args.batch_size),
        max_calib_batches=int(args.max_calib_batches),
    )
    assert_quantile_tensor_shape(calib_quantiles, len(horizons), len(quantiles), context="followup calibration quantiles")
    assert_target_tensor_shape(calib_labels, len(horizons), context="followup calibration labels")
    assert_monotonic_quantiles(calib_quantiles, context="followup calibration quantiles")

    test_quantiles = np.load(stage1_output_dir / "predict_quantiles.npy")
    test_labels = np.load(stage1_output_dir / "label_list.npy")
    if args.max_test_windows > 0:
        test_quantiles = test_quantiles[: args.max_test_windows]
        test_labels = test_labels[: args.max_test_windows]
    assert_quantile_tensor_shape(test_quantiles, len(horizons), len(quantiles), context="followup test quantiles")
    assert_target_tensor_shape(test_labels, len(horizons), context="followup test labels")
    assert_monotonic_quantiles(test_quantiles, context="followup test quantiles")
    calib_quantiles_shape = list(calib_quantiles.shape)
    calib_labels_shape = list(calib_labels.shape)

    scales = shortage_scales(calib_labels, horizons)
    safety_selection = build_safety_margin_selection(calib_quantiles, calib_labels, quantiles, horizons, alphas)
    safety_policies = build_safety_margin_policies(test_quantiles, quantiles, horizons, safety_selection)
    safety_summary, safety_horizon, safety_cost = evaluate_policy_dict(safety_policies, test_labels, scales, horizons, cost_ratios)

    empirical_selection = build_empirical_vs_crc_selection(calib_quantiles, calib_labels, quantiles, horizons, alphas)
    empirical_policies = build_raw_selector_policies(test_quantiles, horizons, empirical_selection)
    empirical_summary, empirical_horizon, empirical_cost = evaluate_policy_dict(empirical_policies, test_labels, scales, horizons, cost_ratios)

    h1_df = build_h1_alpha_sensitivity(calib_quantiles, calib_labels, test_quantiles, test_labels, quantiles, h1_alphas, cost_ratios)

    block_selection = pd.DataFrame()
    block_summary = pd.DataFrame()
    block_horizon = pd.DataFrame()
    block_cost = pd.DataFrame()
    block_policies: dict[str, dict[str, object]] = {}
    if args.enable_block_crc:
        block_selection = build_block_crc_selection(calib_quantiles, calib_labels, quantiles, horizons, alphas)
        block_policies = build_block_crc_policies(test_quantiles, horizons, block_selection)
        block_summary, block_horizon, block_cost = evaluate_policy_dict(block_policies, test_labels, scales, horizons, cost_ratios)
        block_summary = add_block_test_metrics(block_summary, block_policies, test_labels)

    hashes = {
        "qhat_calib_sha256": array_hash(calib_quantiles),
        "y_calib_sha256": array_hash(calib_labels),
        "qhat_test_sha256": array_hash(test_quantiles),
        "y_test_sha256": array_hash(test_labels),
    }
    if args.save_calibration_arrays:
        np.save(output_dir / "crc_calib_quantiles.npy", calib_quantiles)
        np.save(output_dir / "crc_calib_labels.npy", calib_labels)

    del calib_quantiles, calib_labels
    gc.collect()

    all_followup_policies = {}
    all_followup_policies.update(safety_policies)
    all_followup_policies.update(empirical_policies)
    all_followup_policies.update(block_policies)
    manifest_df = pd.read_csv(reinforcement_dir / "test_window_manifest.csv")
    date_keys = date_keys_from_manifest(manifest_df, n_test_windows=test_labels.shape[0], horizons=horizons)

    v1_summary, v1_cost = load_v1_tables(crc_v1_dir)
    symmetric_decision = None
    if not v1_summary.empty:
        thresholds = pd.read_csv(stage2_output_dir / "cqr_thresholds.csv")
        global_s = float(
            thresholds[
                (thresholds["method"] == "global_cqr")
                & np.isclose(thresholds["delta"].astype(float), 0.1)
            ].iloc[0]["s_hat"]
        )
        q95_idx = quantile_index(quantiles, 0.95)
        symmetric_decision = np.maximum(test_quantiles[..., q95_idx] + global_s, 0.0)
    bootstrap_df = bootstrap_followup(
        policies=all_followup_policies,
        labels=test_labels,
        scales=scales,
        horizons=horizons,
        cost_ratios=cost_ratios,
        date_keys=date_keys,
        n_samples=int(args.bootstrap_samples),
        seed=int(args.bootstrap_seed),
        symmetric_decision=symmetric_decision,
    )

    followup_summary = pd.concat([safety_summary, empirical_summary, block_summary], ignore_index=True)
    followup_horizon = pd.concat([safety_horizon, empirical_horizon, block_horizon], ignore_index=True)
    followup_cost = pd.concat([safety_cost, empirical_cost, block_cost], ignore_index=True)
    followup_summary["source"] = "crc_followup"
    if not v1_summary.empty:
        v1_aligned = v1_summary.copy()
        v1_aligned["source"] = "crc_v1"
        combined = pd.concat([v1_aligned, followup_summary], ignore_index=True, sort=False)
    else:
        combined = followup_summary.copy()

    safety_selection.to_csv(output_dir / "crc_safety_margin_selection.csv", index=False)
    safety_summary.to_csv(output_dir / "crc_safety_margin_test_summary.csv", index=False)
    safety_horizon.to_csv(output_dir / "crc_safety_margin_by_horizon.csv", index=False)
    empirical_selection.to_csv(output_dir / "crc_empirical_vs_corrected_selection.csv", index=False)
    empirical_summary.to_csv(output_dir / "crc_empirical_vs_corrected_test_summary.csv", index=False)
    h1_df.to_csv(output_dir / "crc_h1_alpha_sensitivity.csv", index=False)
    block_selection.to_csv(output_dir / "crc_blocked_selection.csv", index=False)
    block_summary.to_csv(output_dir / "crc_blocked_test_summary.csv", index=False)
    block_horizon.to_csv(output_dir / "crc_blocked_by_horizon.csv", index=False)
    combined.to_csv(output_dir / "crc_followup_policy_comparison.csv", index=False)
    combined.to_csv(output_dir / "crc_followup_risk_efficiency_table.csv", index=False)
    followup_cost.to_csv(output_dir / "crc_followup_cost_table.csv", index=False)
    bootstrap_df.to_csv(output_dir / "crc_followup_bootstrap_ci.csv", index=False)
    (output_dir / "crc_calib_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    calib_manifest.update(
        {
            "no_leakage_rule": "all follow-up policy selection uses reconstructed calibration arrays only",
            "calib_quantiles_shape": calib_quantiles_shape,
            "calib_labels_shape": calib_labels_shape,
            "calibration_arrays_saved": bool(args.save_calibration_arrays),
            "followup_output_dir": str(output_dir),
        }
    )
    (output_dir / "crc_calib_manifest.json").write_text(json.dumps(calib_manifest, indent=2), encoding="utf-8")

    plots = write_followup_plots(output_dir, combined, safety_summary, empirical_summary, h1_df, block_summary)
    metadata_out = {
        "stage": "crc_followup",
        "goal": "Reviewer-defense follow-up experiments for CRC deployment",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "machine": args.machine,
        "branch": git_value(["branch", "--show-current"]),
        "commit": git_value(["rev-parse", "--short", "HEAD"]),
        "stage1_output_dir": str(stage1_output_dir),
        "stage2_output_dir": str(stage2_output_dir),
        "crc_v1_dir": str(crc_v1_dir),
        "output_dir": str(output_dir),
        "horizons": horizons,
        "quantiles": quantiles,
        "alphas": alphas,
        "h1_alphas": h1_alphas,
        "cost_ratios": [ratio.__dict__ for ratio in cost_ratios],
        "experiments": {
            "safety_margin": True,
            "empirical_vs_crc_selector": True,
            "h1_alpha_sensitivity": True,
            "block_crc": bool(args.enable_block_crc),
            "peak_hour_crc": False,
        },
        "no_leakage": {
            "calibration_source": "reconstructed_from_stage1_checkpoint_and_calibration_split",
            "test_source": "saved_stage1_test_arrays",
            "selection_uses_test_labels": False,
        },
        "input_shapes": {
            "test_quantiles": list(test_quantiles.shape),
            "test_labels": list(test_labels.shape),
        },
        "calibration_hashes": hashes,
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "plots": plots,
    }
    (output_dir / "crc_followup_metadata.json").write_text(json.dumps(metadata_out, indent=2), encoding="utf-8")
    write_report(output_dir, metadata_out, safety_summary, empirical_summary, h1_df, block_summary, combined)
    print(json.dumps(metadata_out, indent=2))


if __name__ == "__main__":
    main()
