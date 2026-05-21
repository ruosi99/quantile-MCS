from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.journal.build_crc_risk_control_experiments import (  # noqa: E402
    cvar95,
    date_keys_from_manifest,
    per_day_values,
)
from scripts.journal.build_paper_evidence_assets import bootstrap_mean_ci  # noqa: E402
from scripts.journal.calibrate_multihorizon_cqr import load_stage1_metadata, resolve_path  # noqa: E402
from scripts.journal.evaluate_stage4_decision import (  # noqa: E402
    CostRatio,
    newsvendor_cost,
    parse_cost_ratios,
)
from scripts.journal.train_multihorizon_raw import parse_bool, parse_float_list, quantile_index  # noqa: E402
from utils.model_training.journal_contracts import (  # noqa: E402
    assert_monotonic_quantiles,
    assert_quantile_tensor_shape,
    assert_target_tensor_shape,
)


DEFAULT_COST_RATIOS = "1:1,3:1,5:1,9:1,19:1"
DEFAULT_ALPHAS = "0.05,0.10,0.20"
DEFAULT_H1_ALPHAS = "0.03,0.04,0.05,0.06,0.08,0.10"
MAIN_CRC_SCOPES = ("global", "horizon")


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


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def alpha_token(alpha: float) -> str:
    return f"{float(alpha):g}"


def cost_ratio_to_quantile(c_u: float, c_o: float) -> float:
    if c_u <= 0 or c_o <= 0:
        raise ValueError("Cost ratio values must be positive.")
    return float(c_u / (c_u + c_o))


def quantile_to_implied_ratio(tau: float) -> float:
    tau = float(tau)
    if tau >= 1.0:
        return float("inf")
    if tau <= 0.0:
        return 0.0
    return float(tau / (1.0 - tau))


def expected_cost_column(label: str) -> str:
    return f"expected_cost_{label.replace(':', '_')}"


def threshold_value(thresholds_df: pd.DataFrame, method: str, delta: float, horizon: int | str) -> float:
    rows = thresholds_df[
        (thresholds_df["method"].astype(str) == method)
        & np.isclose(thresholds_df["delta"].astype(float), float(delta))
        & (thresholds_df["horizon"].astype(str) == str(horizon))
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one threshold for {method}, delta={delta}, horizon={horizon}; got {len(rows)}")
    return float(rows.iloc[0]["s_hat"])


def one_sided_threshold(decision_thresholds_df: pd.DataFrame, cost_ratio: str, horizon: int) -> float:
    rows = decision_thresholds_df[
        (decision_thresholds_df["method"].astype(str) == "horizon_cqr")
        & (decision_thresholds_df["cost_ratio"].astype(str) == str(cost_ratio))
        & (decision_thresholds_df["horizon"].astype(str) == str(horizon))
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one Stage 4 one-sided threshold for {cost_ratio}, H{horizon}; got {len(rows)}")
    return float(rows.iloc[0]["s_hat"])


def decision_metrics(
    decision: np.ndarray,
    labels: np.ndarray,
    cost_ratios: list[CostRatio],
    cost_ratio_for_main: CostRatio | None = None,
) -> dict[str, float]:
    decision = np.asarray(decision, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    shortage = np.maximum(labels - decision, 0.0)
    overage = np.maximum(decision - labels, 0.0)
    row = {
        "test_violation_rate": float(np.mean(labels > decision)),
        "mean_overage": float(np.mean(overage)),
        "mean_shortage": float(np.mean(shortage)),
        "cvar95_shortage": cvar95(shortage),
        "mean_decision_level": float(np.mean(decision)),
        "mean_label": float(np.mean(labels)),
    }
    for ratio in cost_ratios:
        row[expected_cost_column(ratio.label)] = float(
            np.mean(newsvendor_cost(decision, labels, c_u=ratio.c_u, c_o=ratio.c_o))
        )
    if cost_ratio_for_main is not None:
        row["expected_cost_under_own_ratio"] = row[expected_cost_column(cost_ratio_for_main.label)]
    return row


def decision_metrics_by_horizon(
    decision: np.ndarray,
    labels: np.ndarray,
    horizons: list[int],
    cost_ratios: list[CostRatio],
    base: dict[str, Any],
    cost_ratio_for_main: CostRatio | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for h_idx, horizon in enumerate(horizons):
        rows.append(
            {
                **base,
                "horizon": int(horizon),
                **decision_metrics(
                    decision[:, :, h_idx],
                    labels[:, :, h_idx],
                    cost_ratios=cost_ratios,
                    cost_ratio_for_main=cost_ratio_for_main,
                ),
            }
        )
    return rows


def load_test_arrays(stage1_output_dir: Path, max_test_windows: int = 0) -> tuple[np.ndarray, np.ndarray]:
    q_path = stage1_output_dir / "predict_quantiles.npy"
    y_path = stage1_output_dir / "label_list.npy"
    if not q_path.exists():
        raise FileNotFoundError(f"Missing Stage 1 test quantiles: {q_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing Stage 1 test labels: {y_path}")
    qhat = np.load(q_path)
    labels = np.load(y_path)
    if max_test_windows > 0:
        qhat = qhat[:max_test_windows]
        labels = labels[:max_test_windows]
    return qhat, labels


def raw_quantile_decision(test_quantiles: np.ndarray, quantiles: list[float], tau: float) -> np.ndarray:
    q_idx = quantile_index(quantiles, float(tau))
    return np.maximum(test_quantiles[..., q_idx], 0.0)


def symmetric_90_upper_decision(
    test_quantiles: np.ndarray,
    quantiles: list[float],
    thresholds_df: pd.DataFrame,
) -> np.ndarray:
    q95_idx = quantile_index(quantiles, 0.95)
    s_hat = threshold_value(thresholds_df, method="global_cqr", delta=0.1, horizon="all")
    return np.maximum(test_quantiles[..., q95_idx] + s_hat, 0.0)


def one_sided_refined_decision(
    test_quantiles: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    ratio: CostRatio,
    decision_thresholds_df: pd.DataFrame,
) -> np.ndarray:
    q_idx = quantile_index(quantiles, ratio.tau_star)
    decision = np.empty(test_quantiles.shape[:3], dtype=np.float64)
    for h_idx, horizon in enumerate(horizons):
        s_hat = one_sided_threshold(decision_thresholds_df, cost_ratio=ratio.label, horizon=int(horizon))
        decision[:, :, h_idx] = np.maximum(test_quantiles[:, :, h_idx, q_idx] + s_hat, 0.0)
    return decision


def selection_rows_for_scope(
    selection_df: pd.DataFrame,
    scope: str,
    alpha: float,
    loss_name: str = "violation",
) -> pd.DataFrame:
    rows = selection_df[
        (selection_df["crc_scope"].astype(str) == str(scope))
        & (selection_df["loss_name"].astype(str) == str(loss_name))
        & np.isclose(selection_df["alpha"].astype(float), float(alpha))
    ].copy()
    if rows.empty:
        raise ValueError(f"Missing CRC selection for scope={scope}, loss={loss_name}, alpha={alpha}")
    return rows


def crc_quantiles_by_horizon(
    selection_df: pd.DataFrame,
    scope: str,
    alpha: float,
    horizons: list[int],
) -> dict[int, float]:
    rows = selection_rows_for_scope(selection_df, scope=scope, alpha=alpha)
    if scope == "global":
        row = rows[rows["horizon"].astype(str) == "all"].iloc[0]
        tau = float(row["selected_quantile"])
        return {int(horizon): tau for horizon in horizons}
    out: dict[int, float] = {}
    for horizon in horizons:
        h_rows = rows[rows["horizon"].astype(str) == str(horizon)]
        if len(h_rows) != 1:
            raise ValueError(f"Missing horizon CRC selection for alpha={alpha}, H{horizon}")
        out[int(horizon)] = float(h_rows.iloc[0]["selected_quantile"])
    return out


def horizon_quantile_decision(
    test_quantiles: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    tau_by_horizon: dict[int, float],
) -> np.ndarray:
    decision = np.empty(test_quantiles.shape[:3], dtype=np.float64)
    for h_idx, horizon in enumerate(horizons):
        q_idx = quantile_index(quantiles, float(tau_by_horizon[int(horizon)]))
        decision[:, :, h_idx] = np.maximum(test_quantiles[:, :, h_idx, q_idx], 0.0)
    return decision


def build_cost_to_risk_mapping(
    test_quantiles: np.ndarray,
    labels: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    cost_ratios: list[CostRatio],
    decision_thresholds_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    for ratio in cost_ratios:
        tau = float(ratio.tau_star)
        q_idx = quantile_index(quantiles, tau)
        for policy_family, decision in [
            ("raw_cost_aligned_quantile", raw_quantile_decision(test_quantiles, quantiles, tau)),
            ("one_sided_refined_cost_aligned_quantile", one_sided_refined_decision(test_quantiles, quantiles, horizons, ratio, decision_thresholds_df)),
        ]:
            base = {
                "policy": f"{policy_family}_{ratio.label}",
                "policy_family": policy_family,
                "preference_language": "cost_ratio",
                "cost_ratio": ratio.label,
                "c_u": float(ratio.c_u),
                "c_o": float(ratio.c_o),
                "tau_cost": tau,
                "deployed_quantile": float(quantiles[q_idx]),
                "nominal_risk": float(1.0 - quantiles[q_idx]),
            }
            metrics = decision_metrics(decision, labels, cost_ratios, cost_ratio_for_main=ratio)
            pooled_rows.append({**base, **metrics, "violation_gap": metrics["test_violation_rate"] - base["nominal_risk"]})
            for row in decision_metrics_by_horizon(decision, labels, horizons, cost_ratios, base, cost_ratio_for_main=ratio):
                row["violation_gap"] = float(row["test_violation_rate"] - row["nominal_risk"])
                horizon_rows.append(row)
    return pd.DataFrame(pooled_rows), pd.DataFrame(horizon_rows)


def build_risk_to_cost_mapping(
    selection_df: pd.DataFrame,
    test_quantiles: np.ndarray,
    labels: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    alphas: list[float],
    cost_ratios: list[CostRatio],
    h1_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    for alpha in alphas:
        for scope in MAIN_CRC_SCOPES:
            tau_by_horizon = crc_quantiles_by_horizon(selection_df, scope=scope, alpha=alpha, horizons=horizons)
            decision = horizon_quantile_decision(test_quantiles, quantiles, horizons, tau_by_horizon)
            tau_values = np.asarray(list(tau_by_horizon.values()), dtype=np.float64)
            policy = f"{scope}_crc_violation_alpha{alpha_token(alpha)}"
            base = {
                "policy": policy,
                "policy_family": "crc_violation",
                "preference_language": "risk_budget",
                "crc_scope": scope,
                "alpha": float(alpha),
                "selected_quantile": float(tau_values[0]) if np.allclose(tau_values, tau_values[0]) else "",
                "mean_selected_quantile": float(np.mean(tau_values)),
                "min_selected_quantile": float(np.min(tau_values)),
                "max_selected_quantile": float(np.max(tau_values)),
                "implied_cost_ratio": quantile_to_implied_ratio(float(np.mean(tau_values))),
                "selected_quantiles_by_horizon": json.dumps(tau_by_horizon, sort_keys=True),
            }
            pooled_rows.append({**base, **decision_metrics(decision, labels, cost_ratios)})
            for h_idx, horizon in enumerate(horizons):
                tau_h = float(tau_by_horizon[int(horizon)])
                h_base = {
                    **base,
                    "horizon": int(horizon),
                    "selected_quantile": tau_h,
                    "implied_cost_ratio": quantile_to_implied_ratio(tau_h),
                }
                horizon_rows.append(
                    {
                        **h_base,
                        **decision_metrics(
                            decision[:, :, h_idx],
                            labels[:, :, h_idx],
                            cost_ratios,
                        ),
                    }
                )

    h1_out = h1_df.copy()
    if not h1_out.empty:
        h1_out["implied_cost_ratio"] = h1_out["selected_quantile"].astype(float).map(quantile_to_implied_ratio)
    return pd.DataFrame(pooled_rows), pd.DataFrame(horizon_rows), h1_out


def build_frontier_policies(
    test_quantiles: np.ndarray,
    labels: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    cost_ratios: list[CostRatio],
    thresholds_df: pd.DataFrame,
    decision_thresholds_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    alphas: list[float],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    policies: dict[str, np.ndarray] = {}
    q50_idx = quantile_index(quantiles, 0.5)
    policies["median_decision"] = np.maximum(test_quantiles[..., q50_idx], 0.0)
    policies["symmetric_90_upper"] = symmetric_90_upper_decision(test_quantiles, quantiles, thresholds_df)
    for ratio in cost_ratios:
        policies[f"raw_cost_aligned_quantile_{ratio.label}"] = raw_quantile_decision(test_quantiles, quantiles, ratio.tau_star)
        policies[f"one_sided_refined_cost_aligned_quantile_{ratio.label}"] = one_sided_refined_decision(
            test_quantiles,
            quantiles,
            horizons,
            ratio,
            decision_thresholds_df,
        )
    for alpha in alphas:
        for scope in MAIN_CRC_SCOPES:
            tau_by_horizon = crc_quantiles_by_horizon(selection_df, scope=scope, alpha=alpha, horizons=horizons)
            policies[f"{scope}_crc_violation_alpha{alpha_token(alpha)}"] = horizon_quantile_decision(
                test_quantiles,
                quantiles,
                horizons,
                tau_by_horizon,
            )

    rows = []
    for policy, decision in policies.items():
        row = {"policy": policy, "source": "derived_preference_mapping"}
        if policy.startswith("raw_cost_aligned"):
            row["policy_family"] = "cost_aligned_raw"
        elif policy.startswith("one_sided"):
            row["policy_family"] = "cost_aligned_one_sided"
        elif "crc_violation" in policy:
            row["policy_family"] = "crc_violation"
        else:
            row["policy_family"] = "baseline"
        rows.append({**row, **decision_metrics(decision, labels, cost_ratios)})
    return pd.DataFrame(rows), policies


def conflict_type(tau_cost: float, tau_crc: float, cost_violation: float, alpha: float) -> str:
    if abs(tau_crc - tau_cost) < 1e-12:
        return "aligned"
    if cost_violation > alpha + 1e-12 and tau_crc > tau_cost:
        return "cost_policy_too_aggressive"
    if cost_violation <= alpha + 1e-12 and tau_crc <= tau_cost:
        return "cost_policy_risk_feasible"
    if tau_crc > tau_cost:
        return "crc_more_conservative_than_needed"
    return "crc_less_conservative_than_cost_policy"


def build_conflict_map(
    cost_mapping: pd.DataFrame,
    frontier_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    horizons: list[int],
    cost_ratios: list[CostRatio],
    alphas: list[float],
) -> pd.DataFrame:
    rows = []
    raw_cost = cost_mapping[cost_mapping["policy_family"] == "raw_cost_aligned_quantile"].set_index("cost_ratio")
    frontier = frontier_df.set_index("policy")
    for ratio in cost_ratios:
        cost_row = raw_cost.loc[ratio.label]
        cost_policy = f"raw_cost_aligned_quantile_{ratio.label}"
        for alpha in alphas:
            for scope in MAIN_CRC_SCOPES:
                tau_by_h = crc_quantiles_by_horizon(selection_df, scope=scope, alpha=alpha, horizons=horizons)
                tau_crc = float(np.mean(list(tau_by_h.values())))
                crc_policy = f"{scope}_crc_violation_alpha{alpha_token(alpha)}"
                crc_row = frontier.loc[crc_policy]
                rows.append(
                    {
                        "cost_ratio": ratio.label,
                        "alpha": float(alpha),
                        "crc_scope": scope,
                        "cost_policy": cost_policy,
                        "crc_policy": crc_policy,
                        "tau_cost": float(ratio.tau_star),
                        "tau_crc": tau_crc,
                        "tau_gap": tau_crc - float(ratio.tau_star),
                        "cost_policy_actual_violation": float(cost_row["test_violation_rate"]),
                        "crc_policy_actual_violation": float(crc_row["test_violation_rate"]),
                        "violation_gap_crc_minus_cost": float(crc_row["test_violation_rate"] - cost_row["test_violation_rate"]),
                        "cost_policy_expected_cost": float(cost_row[expected_cost_column(ratio.label)]),
                        "crc_policy_expected_cost": float(crc_row[expected_cost_column(ratio.label)]),
                        "expected_cost_gap_crc_minus_cost": float(
                            crc_row[expected_cost_column(ratio.label)] - cost_row[expected_cost_column(ratio.label)]
                        ),
                        "cost_policy_mean_overage": float(cost_row["mean_overage"]),
                        "crc_policy_mean_overage": float(crc_row["mean_overage"]),
                        "overage_gap_crc_minus_cost": float(crc_row["mean_overage"] - cost_row["mean_overage"]),
                        "selected_quantiles_by_horizon": json.dumps(tau_by_h, sort_keys=True),
                        "conflict_type": conflict_type(
                            tau_cost=float(ratio.tau_star),
                            tau_crc=tau_crc,
                            cost_violation=float(cost_row["test_violation_rate"]),
                            alpha=float(alpha),
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_phase_gate(conflict_df: pd.DataFrame) -> dict[str, Any]:
    too_aggressive = conflict_df[conflict_df["conflict_type"] == "cost_policy_too_aggressive"]
    nontrivial_tau_gap = conflict_df[np.abs(conflict_df["tau_gap"].astype(float)) >= 0.05]
    max_violation_excess = float(
        np.max(conflict_df["cost_policy_actual_violation"].astype(float) - conflict_df["alpha"].astype(float))
    )
    gate_pass = bool((not too_aggressive.empty) or (not nontrivial_tau_gap.empty) or max_violation_excess > 0.01)
    return {
        "phase_a_gate_pass": gate_pass,
        "reason": "nontrivial preference mismatch" if gate_pass else "near-equivalence between preference languages",
        "too_aggressive_pair_count": int(len(too_aggressive)),
        "nontrivial_tau_gap_pair_count": int(len(nontrivial_tau_gap)),
        "max_cost_policy_violation_minus_alpha": max_violation_excess,
    }


def build_risk_screened_deployment(
    test_quantiles: np.ndarray,
    labels: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    cost_ratios: list[CostRatio],
    alphas: list[float],
    selection_df: pd.DataFrame,
    symmetric_decision: np.ndarray,
    enable: bool,
    gate: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not enable or not bool(gate["phase_a_gate_pass"]):
        empty = pd.DataFrame()
        return empty, empty, empty

    summary_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    for ratio in cost_ratios:
        cost_decision = raw_quantile_decision(test_quantiles, quantiles, ratio.tau_star)
        for alpha in alphas:
            for scope in MAIN_CRC_SCOPES:
                tau_crc_by_h = crc_quantiles_by_horizon(selection_df, scope=scope, alpha=alpha, horizons=horizons)
                crc_decision = horizon_quantile_decision(test_quantiles, quantiles, horizons, tau_crc_by_h)
                screened_tau_by_h = {
                    int(h): max(float(ratio.tau_star), float(tau_crc_by_h[int(h)]))
                    for h in horizons
                }
                screened_decision = horizon_quantile_decision(test_quantiles, quantiles, horizons, screened_tau_by_h)
                parent_decisions = {
                    "cost_only": (cost_decision, ratio.tau_star),
                    "crc_only": (crc_decision, float(np.mean(list(tau_crc_by_h.values())))),
                    "risk_screened_max_tau": (screened_decision, float(np.mean(list(screened_tau_by_h.values())))),
                    "symmetric_90_upper": (symmetric_decision, 0.95),
                }
                cost_metrics = decision_metrics(cost_decision, labels, cost_ratios, cost_ratio_for_main=ratio)
                crc_metrics = decision_metrics(crc_decision, labels, cost_ratios, cost_ratio_for_main=ratio)
                for role, (decision, selected_tau) in parent_decisions.items():
                    metrics = decision_metrics(decision, labels, cost_ratios, cost_ratio_for_main=ratio)
                    base = {
                        "policy": f"{role}_{scope}_alpha{alpha_token(alpha)}_{ratio.label}",
                        "policy_role": role,
                        "cost_ratio": ratio.label,
                        "alpha": float(alpha),
                        "crc_scope": scope,
                        "tau_cost": float(ratio.tau_star),
                        "tau_crc": float(np.mean(list(tau_crc_by_h.values()))),
                        "selected_quantile": float(selected_tau),
                        "selected_quantiles_by_horizon": json.dumps(screened_tau_by_h if role == "risk_screened_max_tau" else tau_crc_by_h, sort_keys=True)
                        if role in {"risk_screened_max_tau", "crc_only"}
                        else "",
                        "risk_target_satisfied": bool(metrics["test_violation_rate"] <= float(alpha) + 1e-12),
                        "cost_increase_vs_cost_only": float(
                            metrics[expected_cost_column(ratio.label)] - cost_metrics[expected_cost_column(ratio.label)]
                        ),
                        "overage_reduction_vs_crc_only": float(crc_metrics["mean_overage"] - metrics["mean_overage"]),
                        "violation_reduction_vs_cost_only": float(
                            cost_metrics["test_violation_rate"] - metrics["test_violation_rate"]
                        ),
                    }
                    summary_rows.append({**base, **metrics})
                    horizon_rows.extend(
                        decision_metrics_by_horizon(
                            decision,
                            labels,
                            horizons,
                            cost_ratios,
                            base,
                            cost_ratio_for_main=ratio,
                        )
                    )
    summary_df = pd.DataFrame(summary_rows)
    comparison_cols = [
        "cost_ratio",
        "alpha",
        "crc_scope",
        "policy_role",
        "selected_quantile",
        "test_violation_rate",
        "risk_target_satisfied",
        "expected_cost_under_own_ratio",
        "mean_overage",
        "mean_shortage",
        "cvar95_shortage",
        "cost_increase_vs_cost_only",
        "overage_reduction_vs_crc_only",
        "violation_reduction_vs_cost_only",
    ]
    return summary_df, pd.DataFrame(horizon_rows), summary_df[comparison_cols].copy()


def build_bootstrap_ci(
    labels: np.ndarray,
    policy_decisions: dict[str, np.ndarray],
    cost_ratios: list[CostRatio],
    date_keys: np.ndarray,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for policy, decision in policy_decisions.items():
        violation = (labels > decision).astype(np.float64)
        overage = np.maximum(decision - labels, 0.0)
        shortage = np.maximum(labels - decision, 0.0)
        metric_arrays = {
            "test_violation_rate": violation,
            "mean_overage": overage,
            "mean_shortage": shortage,
        }
        for metric, values in metric_arrays.items():
            day_values = per_day_values(values, date_keys)
            point, low, high = bootstrap_mean_ci(day_values, rng=rng, n_samples=n_samples)
            rows.append(
                {
                    "policy": policy,
                    "metric": metric,
                    "evaluation_cost_ratio": "",
                    "point": point,
                    "ci_low": low,
                    "ci_high": high,
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
                    "policy": policy,
                    "metric": "expected_cost",
                    "evaluation_cost_ratio": ratio.label,
                    "point": point,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_units": int(day_values.size),
                    "bootstrap_samples": int(n_samples),
                }
            )
    return pd.DataFrame(rows)


def build_key_policy_decisions(
    test_quantiles: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    cost_ratios: list[CostRatio],
    selection_df: pd.DataFrame,
    alphas: list[float],
    symmetric_decision: np.ndarray,
    risk_screened_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    policies: dict[str, np.ndarray] = {"symmetric_90_upper": symmetric_decision}
    for ratio in cost_ratios:
        policies[f"raw_cost_aligned_quantile_{ratio.label}"] = raw_quantile_decision(test_quantiles, quantiles, ratio.tau_star)
    for alpha in alphas:
        for scope in MAIN_CRC_SCOPES:
            tau_by_h = crc_quantiles_by_horizon(selection_df, scope=scope, alpha=alpha, horizons=horizons)
            policies[f"{scope}_crc_violation_alpha{alpha_token(alpha)}"] = horizon_quantile_decision(
                test_quantiles,
                quantiles,
                horizons,
                tau_by_h,
            )
    if not risk_screened_df.empty:
        screened = risk_screened_df[risk_screened_df["policy_role"] == "risk_screened_max_tau"]
        for _, row in screened.iterrows():
            tau_by_h = {int(k): float(v) for k, v in json.loads(row["selected_quantiles_by_horizon"]).items()}
            policies[str(row["policy"])] = horizon_quantile_decision(test_quantiles, quantiles, horizons, tau_by_h)
    return policies


def write_plots(
    output_dir: Path,
    cost_to_risk: pd.DataFrame,
    risk_to_cost: pd.DataFrame,
    frontier: pd.DataFrame,
    conflict: pd.DataFrame,
    risk_screened: pd.DataFrame,
    cost_ratios: list[CostRatio],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    written: list[str] = []

    raw = cost_to_risk[cost_to_risk["policy_family"] == "raw_cost_aligned_quantile"].copy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(raw["nominal_risk"], raw["test_violation_rate"], marker="o")
    ax.plot([0, max(raw["nominal_risk"].max(), raw["test_violation_rate"].max())], [0, max(raw["nominal_risk"].max(), raw["test_violation_rate"].max())], linestyle="--", color="gray")
    for _, row in raw.iterrows():
        ax.annotate(str(row["cost_ratio"]), (float(row["nominal_risk"]), float(row["test_violation_rate"])), fontsize=8)
    ax.set_xlabel("Nominal risk from cost ratio")
    ax.set_ylabel("Actual test violation")
    ax.set_title("Cost-to-risk mapping")
    fig.tight_layout()
    path = output_dir / "cost_to_risk_mapping.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    written.append(path.name)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for scope, group in risk_to_cost.groupby("crc_scope"):
        group = group.sort_values("alpha")
        ax.plot(group["alpha"].astype(float), group["implied_cost_ratio"].astype(float), marker="o", label=f"{scope} CRC")
    ax.set_xlabel("Risk budget alpha")
    ax.set_ylabel("Implied cost ratio")
    ax.set_title("Risk-to-cost mapping")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "risk_to_cost_mapping.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    written.append(path.name)

    fig, ax = plt.subplots(figsize=(8, 5))
    for family, group in frontier.groupby("policy_family"):
        ax.scatter(group["test_violation_rate"], group["mean_overage"], s=28, label=family)
    ax.set_xlabel("Actual violation rate")
    ax.set_ylabel("Mean overage")
    ax.set_title("Risk-resource frontier")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "risk_resource_frontier.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    written.append(path.name)

    n_cols = len(cost_ratios)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 4.2), sharex=True)
    if n_cols == 1:
        axes = [axes]
    for ax, ratio in zip(axes, cost_ratios):
        col = expected_cost_column(ratio.label)
        for family, group in frontier.groupby("policy_family"):
            ax.scatter(group["test_violation_rate"], group[col], s=22, label=family)
        ax.set_title(ratio.label)
        ax.set_xlabel("Violation")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Expected cost")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=4, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    path = output_dir / "risk_cost_frontier_by_ratio.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    written.append(path.name)

    for value_col, filename, title in [
        ("tau_gap", "preference_conflict_heatmap_tau_gap.png", "Tau gap: CRC minus cost"),
        ("violation_gap_crc_minus_cost", "preference_conflict_heatmap_violation_gap.png", "Violation gap: CRC minus cost"),
    ]:
        scopes = list(MAIN_CRC_SCOPES)
        fig, axes = plt.subplots(1, len(scopes), figsize=(5.2 * len(scopes), 4.2), sharey=True)
        if len(scopes) == 1:
            axes = [axes]
        for ax, scope in zip(axes, scopes):
            sub = conflict[conflict["crc_scope"] == scope]
            pivot = sub.pivot(index="cost_ratio", columns="alpha", values=value_col)
            image = ax.imshow(pivot.values.astype(float), aspect="auto", cmap="coolwarm")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{float(v):g}" for v in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel("Alpha")
            ax.set_title(scope)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        axes[0].set_ylabel("Cost ratio")
        fig.suptitle(title)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=220)
        plt.close(fig)
        written.append(path.name)

    if not risk_screened.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for role, group in risk_screened.groupby("policy_role"):
            ax.scatter(group["test_violation_rate"], group["expected_cost_under_own_ratio"], s=26, label=role)
        ax.set_xlabel("Actual violation rate")
        ax.set_ylabel("Expected cost under stated cost ratio")
        ax.set_title("Risk-screened deployment frontier")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = output_dir / "risk_screened_frontier.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        written.append(path.name)

    return written


def write_report(
    output_dir: Path,
    metadata: dict[str, Any],
    cost_to_risk: pd.DataFrame,
    risk_to_cost: pd.DataFrame,
    conflict: pd.DataFrame,
    risk_screened: pd.DataFrame,
) -> None:
    lines = [
        "# Preference Mapping Experiment Report",
        "",
        f"- Created at: `{metadata['created_at']}`",
        f"- Machine: `{metadata['machine']}`",
        f"- Branch: `{metadata['branch']}`",
        f"- Commit: `{metadata['commit']}`",
        "",
        "## No-Leakage Verification",
        "- Cost-aware policies are selected analytically from the cost ratio.",
        "- CRC policies are imported from calibration-only CRC selection outputs.",
        "- Test labels are used only for final evaluation and plotting.",
        "",
        "## Phase A Gate",
        json.dumps(metadata["phase_a_gate"], indent=2),
        "",
        "## Cost-To-Risk Mapping",
        cost_to_risk[cost_to_risk["policy_family"] == "raw_cost_aligned_quantile"][
            ["cost_ratio", "deployed_quantile", "nominal_risk", "test_violation_rate", "mean_overage", "mean_shortage", "expected_cost_under_own_ratio"]
        ].to_string(index=False),
        "",
        "## Risk-To-Cost Mapping",
        risk_to_cost[
            ["crc_scope", "alpha", "mean_selected_quantile", "implied_cost_ratio", "test_violation_rate", "mean_overage", "mean_shortage"]
        ].to_string(index=False),
        "",
        "## Preference Conflict Summary",
        conflict.groupby(["crc_scope", "conflict_type"], as_index=False).size().to_string(index=False),
        "",
    ]
    if not risk_screened.empty:
        lines.extend(
            [
                "## Risk-Screened Deployment",
                risk_screened[
                    [
                        "cost_ratio",
                        "alpha",
                        "crc_scope",
                        "policy_role",
                        "selected_quantile",
                        "test_violation_rate",
                        "risk_target_satisfied",
                        "expected_cost_under_own_ratio",
                        "cost_increase_vs_cost_only",
                        "violation_reduction_vs_cost_only",
                    ]
                ]
                .head(80)
                .to_string(index=False),
                "",
            ]
        )
    else:
        lines.extend(["## Risk-Screened Deployment", "Skipped because the Phase A gate did not pass or risk screening was disabled.", ""])
    (output_dir / "preference_mapping_report.md").write_text("\n".join(lines), encoding="utf-8")


def validate_input_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description="Preference mapping and risk-screened deployment for the journal branch.")
    parser.add_argument("--stage1-output-dir", default="journal_results/shenzhen_multihorizon/warmstart_raw")
    parser.add_argument("--stage2-output-dir", default="journal_results/shenzhen_multihorizon/stage2_cqr/warmstart_raw")
    parser.add_argument("--stage4-output-dir", default="journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw")
    parser.add_argument("--reinforcement-dir", default="journal_results/shenzhen_multihorizon/reinforcement")
    parser.add_argument("--paper-assets-dir", default="journal_results/shenzhen_multihorizon/paper_assets")
    parser.add_argument("--crc-v1-dir", default="journal_results/shenzhen_multihorizon/crc_risk_control")
    parser.add_argument("--crc-followup-dir", default="journal_results/shenzhen_multihorizon/crc_risk_control_followup")
    parser.add_argument("--output-dir", default="journal_results/shenzhen_multihorizon/preference_mapping")
    parser.add_argument("--cost-ratios", default=DEFAULT_COST_RATIOS)
    parser.add_argument("--alphas-violation", default=DEFAULT_ALPHAS)
    parser.add_argument("--h1-alphas", default=DEFAULT_H1_ALPHAS)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--bootstrap-seed", type=int, default=20260601)
    parser.add_argument("--max-test-windows", type=int, default=0)
    parser.add_argument("--max-calib-batches", type=int, default=0)
    parser.add_argument("--enable-risk-screened", type=parse_bool, default=True)
    parser.add_argument("--enable-aggregate-procurement", type=parse_bool, default=False)
    parser.add_argument("--machine", default="Lenovo")
    args = parser.parse_args()

    stage1_output_dir = resolve_path(args.stage1_output_dir)
    stage2_output_dir = resolve_path(args.stage2_output_dir)
    stage4_output_dir = resolve_path(args.stage4_output_dir)
    reinforcement_dir = resolve_path(args.reinforcement_dir)
    paper_assets_dir = resolve_path(args.paper_assets_dir)
    crc_v1_dir = resolve_path(args.crc_v1_dir)
    crc_followup_dir = resolve_path(args.crc_followup_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_input_files(
        [
            stage1_output_dir / "run_metadata.json",
            stage1_output_dir / "predict_quantiles.npy",
            stage1_output_dir / "label_list.npy",
            stage2_output_dir / "cqr_thresholds.csv",
            stage4_output_dir / "decision_one_sided_thresholds.csv",
            crc_v1_dir / "crc_selection_table.csv",
            crc_followup_dir / "crc_h1_alpha_sensitivity.csv",
            reinforcement_dir / "test_window_manifest.csv",
        ]
    )

    metadata = load_stage1_metadata(stage1_output_dir)
    horizons = [int(h) for h in metadata["horizons"]]
    quantiles = [float(q) for q in metadata["quantiles"]]
    cost_ratios = parse_cost_ratios(args.cost_ratios)
    alphas = parse_float_list(args.alphas_violation)
    h1_alphas = parse_float_list(args.h1_alphas)

    test_quantiles, labels = load_test_arrays(stage1_output_dir, max_test_windows=int(args.max_test_windows))
    assert_quantile_tensor_shape(test_quantiles, len(horizons), len(quantiles), context="preference mapping test quantiles")
    assert_target_tensor_shape(labels, len(horizons), context="preference mapping labels")
    assert_monotonic_quantiles(test_quantiles, context="preference mapping test quantiles")

    thresholds_df = pd.read_csv(stage2_output_dir / "cqr_thresholds.csv")
    decision_thresholds_df = pd.read_csv(stage4_output_dir / "decision_one_sided_thresholds.csv")
    selection_df = pd.read_csv(crc_v1_dir / "crc_selection_table.csv")
    h1_df = pd.read_csv(crc_followup_dir / "crc_h1_alpha_sensitivity.csv")

    cost_to_risk, cost_to_risk_horizon = build_cost_to_risk_mapping(
        test_quantiles=test_quantiles,
        labels=labels,
        quantiles=quantiles,
        horizons=horizons,
        cost_ratios=cost_ratios,
        decision_thresholds_df=decision_thresholds_df,
    )
    risk_to_cost, risk_to_cost_horizon, h1_risk_to_cost = build_risk_to_cost_mapping(
        selection_df=selection_df,
        test_quantiles=test_quantiles,
        labels=labels,
        quantiles=quantiles,
        horizons=horizons,
        alphas=alphas,
        cost_ratios=cost_ratios,
        h1_df=h1_df,
    )
    frontier, frontier_policies = build_frontier_policies(
        test_quantiles=test_quantiles,
        labels=labels,
        quantiles=quantiles,
        horizons=horizons,
        cost_ratios=cost_ratios,
        thresholds_df=thresholds_df,
        decision_thresholds_df=decision_thresholds_df,
        selection_df=selection_df,
        alphas=alphas,
    )
    conflict = build_conflict_map(
        cost_mapping=cost_to_risk,
        frontier_df=frontier,
        selection_df=selection_df,
        horizons=horizons,
        cost_ratios=cost_ratios,
        alphas=alphas,
    )
    phase_gate = build_phase_gate(conflict)
    symmetric_decision = frontier_policies["symmetric_90_upper"]
    risk_screened, risk_screened_horizon, risk_screened_comparison = build_risk_screened_deployment(
        test_quantiles=test_quantiles,
        labels=labels,
        quantiles=quantiles,
        horizons=horizons,
        cost_ratios=cost_ratios,
        alphas=alphas,
        selection_df=selection_df,
        symmetric_decision=symmetric_decision,
        enable=bool(args.enable_risk_screened),
        gate=phase_gate,
    )

    manifest_df = pd.read_csv(reinforcement_dir / "test_window_manifest.csv")
    try:
        date_keys = date_keys_from_manifest(manifest_df, n_test_windows=labels.shape[0], horizons=horizons)
        bootstrap_policies = build_key_policy_decisions(
            test_quantiles=test_quantiles,
            quantiles=quantiles,
            horizons=horizons,
            cost_ratios=cost_ratios,
            selection_df=selection_df,
            alphas=alphas,
            symmetric_decision=symmetric_decision,
            risk_screened_df=risk_screened,
        )
        bootstrap_df = build_bootstrap_ci(
            labels=labels,
            policy_decisions=bootstrap_policies,
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

    cost_to_risk.to_csv(output_dir / "cost_to_risk_mapping.csv", index=False)
    cost_to_risk_horizon.to_csv(output_dir / "cost_to_risk_mapping_by_horizon.csv", index=False)
    risk_to_cost.to_csv(output_dir / "risk_to_cost_mapping.csv", index=False)
    risk_to_cost_horizon.to_csv(output_dir / "risk_to_cost_mapping_by_horizon.csv", index=False)
    h1_risk_to_cost.to_csv(output_dir / "h1_risk_to_cost_mapping.csv", index=False)
    frontier.to_csv(output_dir / "preference_equivalence_frontier.csv", index=False)
    conflict.to_csv(output_dir / "preference_conflict_map.csv", index=False)
    risk_screened.to_csv(output_dir / "risk_screened_deployment.csv", index=False)
    risk_screened_horizon.to_csv(output_dir / "risk_screened_by_horizon.csv", index=False)
    risk_screened_comparison.to_csv(output_dir / "risk_screened_policy_comparison.csv", index=False)
    bootstrap_df.to_csv(output_dir / "preference_mapping_bootstrap_ci.csv", index=False)

    plots = write_plots(
        output_dir=output_dir,
        cost_to_risk=cost_to_risk,
        risk_to_cost=risk_to_cost,
        frontier=frontier,
        conflict=conflict,
        risk_screened=risk_screened,
        cost_ratios=cost_ratios,
    )

    metadata_out: dict[str, Any] = {
        "stage": "preference_mapping",
        "goal": "Bridge cost-ratio and service-risk deployment preferences",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "machine": str(args.machine),
        "branch": git_value(["branch", "--show-current"]),
        "commit": git_value(["rev-parse", "--short", "HEAD"]),
        "stage1_output_dir": str(stage1_output_dir),
        "stage2_output_dir": str(stage2_output_dir),
        "stage4_output_dir": str(stage4_output_dir),
        "reinforcement_dir": str(reinforcement_dir),
        "paper_assets_dir": str(paper_assets_dir),
        "crc_v1_dir": str(crc_v1_dir),
        "crc_followup_dir": str(crc_followup_dir),
        "output_dir": str(output_dir),
        "horizons": horizons,
        "quantiles": quantiles,
        "cost_ratios": [ratio.__dict__ for ratio in cost_ratios],
        "alphas_violation": alphas,
        "h1_alphas": h1_alphas,
        "phase_a_gate": phase_gate,
        "risk_screened_enabled": bool(args.enable_risk_screened),
        "risk_screened_executed": bool(not risk_screened.empty),
        "aggregate_procurement_enabled": bool(args.enable_aggregate_procurement),
        "aggregate_procurement_executed": False,
        "no_leakage": {
            "cost_policy_selection": "analytic_tau_from_cost_ratio",
            "crc_policy_selection": "imported_from_existing_calibration_only_crc_outputs",
            "new_risk_screened_max_tau_selection": "deterministic_combination_of_cost_tau_and_crc_selected_tau",
            "selection_uses_test_labels": False,
            "test_labels_used_for": "final evaluation only",
        },
        "input_shapes": {
            "test_quantiles": list(test_quantiles.shape),
            "test_labels": list(labels.shape),
        },
        "bootstrap_level": bootstrap_level,
        "bootstrap_warning": bootstrap_warning,
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "smoke_limits": {
            "max_test_windows": int(args.max_test_windows),
            "max_calib_batches": int(args.max_calib_batches),
        },
        "outputs": {
            "cost_to_risk_mapping": "cost_to_risk_mapping.csv",
            "cost_to_risk_mapping_by_horizon": "cost_to_risk_mapping_by_horizon.csv",
            "risk_to_cost_mapping": "risk_to_cost_mapping.csv",
            "risk_to_cost_mapping_by_horizon": "risk_to_cost_mapping_by_horizon.csv",
            "h1_risk_to_cost_mapping": "h1_risk_to_cost_mapping.csv",
            "preference_equivalence_frontier": "preference_equivalence_frontier.csv",
            "preference_conflict_map": "preference_conflict_map.csv",
            "risk_screened_deployment": "risk_screened_deployment.csv",
            "risk_screened_by_horizon": "risk_screened_by_horizon.csv",
            "risk_screened_policy_comparison": "risk_screened_policy_comparison.csv",
            "preference_mapping_bootstrap_ci": "preference_mapping_bootstrap_ci.csv",
            "plots": plots,
            "report": "preference_mapping_report.md",
        },
    }
    (output_dir / "preference_mapping_metadata.json").write_text(json.dumps(metadata_out, indent=2), encoding="utf-8")
    write_report(output_dir, metadata_out, cost_to_risk, risk_to_cost, conflict, risk_screened)
    print(json.dumps(metadata_out, indent=2))


if __name__ == "__main__":
    main()
