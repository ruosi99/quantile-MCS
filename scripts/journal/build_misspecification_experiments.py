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

from scripts.journal.build_paper_evidence_assets import bootstrap_mean_ci
from scripts.journal.build_reinforcement_experiments import aggregate_by_date
from scripts.journal.calibrate_multihorizon_cqr import resolve_path
from scripts.journal.evaluate_stage4_decision import CostRatio, newsvendor_cost, parse_cost_ratios
from scripts.journal.train_multihorizon_raw import parse_bool, quantile_index
from utils.model_training.journal_contracts import (
    assert_monotonic_quantiles,
    assert_quantile_tensor_shape,
    assert_target_tensor_shape,
)


DEFAULT_COST_RATIOS = "1:1,3:1,5:1,9:1,19:1"
OPTIONAL_EXTRA_RATIOS = "2:1,7:1,12:1"
GATE_PASS_PCT = 10.0
GATE_MARGINAL_PCT = 5.0
UNCERTAINTY_SCENARIOS = [
    {
        "name": "narrow",
        "r_low": 5.0,
        "r_high": 9.0,
        "description": "Operator knows the cost ratio is roughly 5:1 to 9:1",
        "valid_grid_indices": [2, 3],
    },
    {
        "name": "medium",
        "r_low": 3.0,
        "r_high": 19.0,
        "description": "Operator knows the cost ratio is somewhere between 3:1 and 19:1",
        "valid_grid_indices": [1, 2, 3, 4],
    },
    {
        "name": "wide",
        "r_low": 1.0,
        "r_high": 19.0,
        "description": "Operator has high uncertainty across the full five-point grid",
        "valid_grid_indices": [0, 1, 2, 3, 4],
    },
]
ROBUST_STRATEGIES = ["midpoint", "conservative", "minimax_regret", "expected_cost"]


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


def parse_phases(value: str) -> list[int]:
    phases = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not phases:
        raise ValueError("At least one phase is required.")
    unsupported = [phase for phase in phases if phase not in {1, 2, 3}]
    if unsupported:
        raise ValueError(f"Unsupported phase values: {unsupported}")
    return phases


def build_cost_ratio_grid(include_interpolated_ratios: bool = False) -> list[CostRatio]:
    value = DEFAULT_COST_RATIOS
    if include_interpolated_ratios:
        value = f"{value},{OPTIONAL_EXTRA_RATIOS}"
    ratios = parse_cost_ratios(value)
    return sorted(ratios, key=lambda ratio: ratio.tau_star)


def quantile_prediction(
    predict_quantiles: np.ndarray,
    quantiles: list[float],
    tau: float,
    *,
    allow_interpolation: bool,
) -> tuple[np.ndarray, str, float, float]:
    try:
        idx = quantile_index(quantiles, tau)
        q_value = float(quantiles[idx])
        return np.asarray(predict_quantiles[..., idx], dtype=np.float64), "exact", q_value, q_value
    except ValueError:
        if not allow_interpolation:
            raise

    q_grid = np.asarray(quantiles, dtype=np.float64)
    if tau < float(q_grid[0]) or tau > float(q_grid[-1]):
        raise ValueError(f"tau={tau} is outside quantile grid [{q_grid[0]}, {q_grid[-1]}]")
    right = int(np.searchsorted(q_grid, tau, side="right"))
    left = right - 1
    q_left = float(q_grid[left])
    q_right = float(q_grid[right])
    weight = (float(tau) - q_left) / (q_right - q_left)
    pred = (1.0 - weight) * predict_quantiles[..., left] + weight * predict_quantiles[..., right]
    return np.asarray(pred, dtype=np.float64), "linear_interpolation", q_left, q_right


def decision_for_ratio(
    predict_quantiles: np.ndarray,
    quantiles: list[float],
    ratio: CostRatio,
    *,
    allow_interpolation: bool,
) -> tuple[np.ndarray, dict[str, float | str]]:
    decision, method, q_left, q_right = quantile_prediction(
        predict_quantiles,
        quantiles,
        ratio.tau_star,
        allow_interpolation=allow_interpolation,
    )
    info: dict[str, float | str] = {
        "cost_ratio": ratio.label,
        "tau_star": float(ratio.tau_star),
        "quantile_selection_method": method,
        "quantile_left": float(q_left),
        "quantile_right": float(q_right),
    }
    return np.maximum(decision, 0.0), info


def matrix_frame(values: dict[tuple[str, str], float], ratios: list[CostRatio], value_name: str) -> pd.DataFrame:
    rows = []
    for assumed in ratios:
        row: dict[str, float | str] = {"assumed_cost_ratio": assumed.label}
        for true in ratios:
            row[true.label] = float(values[(assumed.label, true.label)])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["value_name"] = value_name
    return df


def long_matrix_frame(
    cost_matrix: dict[tuple[str, str], float],
    regret_matrix: dict[tuple[str, str], float],
    pct_matrix: dict[tuple[str, str], float],
    ratios: list[CostRatio],
) -> pd.DataFrame:
    rows = []
    by_label = {ratio.label: ratio for ratio in ratios}
    for assumed in ratios:
        for true in ratios:
            rows.append(
                {
                    "assumed_cost_ratio": assumed.label,
                    "true_cost_ratio": true.label,
                    "assumed_c_u": float(assumed.c_u),
                    "assumed_c_o": float(assumed.c_o),
                    "assumed_tau": float(assumed.tau_star),
                    "true_c_u": float(true.c_u),
                    "true_c_o": float(true.c_o),
                    "true_tau": float(true.tau_star),
                    "assumption_error": "diagonal"
                    if assumed.label == true.label
                    else ("underestimate" if assumed.tau_star < by_label[true.label].tau_star else "overestimate"),
                    "expected_cost": float(cost_matrix[(assumed.label, true.label)]),
                    "regret_vs_true_aligned": float(regret_matrix[(assumed.label, true.label)]),
                    "pct_regret_vs_true_aligned": float(pct_matrix[(assumed.label, true.label)]),
                }
            )
    return pd.DataFrame(rows)


def build_phase1_aggregate(
    decisions: dict[str, np.ndarray],
    labels: np.ndarray,
    ratios: list[CostRatio],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cost_matrix: dict[tuple[str, str], float] = {}
    for assumed in ratios:
        decision = decisions[assumed.label]
        for true in ratios:
            cost = newsvendor_cost(decision, labels, c_u=true.c_u, c_o=true.c_o)
            cost_matrix[(assumed.label, true.label)] = float(np.mean(cost))

    regret_matrix: dict[tuple[str, str], float] = {}
    pct_matrix: dict[tuple[str, str], float] = {}
    for assumed in ratios:
        for true in ratios:
            baseline = cost_matrix[(true.label, true.label)]
            regret = cost_matrix[(assumed.label, true.label)] - baseline
            regret_matrix[(assumed.label, true.label)] = float(regret)
            pct_matrix[(assumed.label, true.label)] = float(100.0 * regret / baseline) if baseline else float("nan")

    return (
        matrix_frame(cost_matrix, ratios, "expected_cost"),
        matrix_frame(regret_matrix, ratios, "regret_vs_true_aligned"),
        matrix_frame(pct_matrix, ratios, "pct_regret_vs_true_aligned"),
        long_matrix_frame(cost_matrix, regret_matrix, pct_matrix, ratios),
    )


def build_phase1_by_horizon(decisions: dict[str, np.ndarray], labels: np.ndarray, ratios: list[CostRatio], horizons: list[int]) -> pd.DataFrame:
    rows = []
    for h_idx, horizon in enumerate(horizons):
        baseline_by_true: dict[str, float] = {}
        for true in ratios:
            cost = newsvendor_cost(decisions[true.label][:, :, h_idx], labels[:, :, h_idx], c_u=true.c_u, c_o=true.c_o)
            baseline_by_true[true.label] = float(np.mean(cost))
        for assumed in ratios:
            for true in ratios:
                cost = newsvendor_cost(decisions[assumed.label][:, :, h_idx], labels[:, :, h_idx], c_u=true.c_u, c_o=true.c_o)
                expected_cost = float(np.mean(cost))
                baseline = baseline_by_true[true.label]
                regret = expected_cost - baseline
                rows.append(
                    {
                        "horizon": int(horizon),
                        "assumed_cost_ratio": assumed.label,
                        "true_cost_ratio": true.label,
                        "assumed_tau": float(assumed.tau_star),
                        "true_tau": float(true.tau_star),
                        "assumption_error": "diagonal"
                        if assumed.label == true.label
                        else ("underestimate" if assumed.tau_star < true.tau_star else "overestimate"),
                        "expected_cost": expected_cost,
                        "baseline_true_aligned_cost": baseline,
                        "regret_vs_true_aligned": float(regret),
                        "pct_regret_vs_true_aligned": float(100.0 * regret / baseline) if baseline else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def load_station_groups(reinforcement_dir: Path, labels: np.ndarray) -> pd.DataFrame:
    groups_path = reinforcement_dir / "station_zero_ratio_by_station.csv"
    if not groups_path.exists():
        raise FileNotFoundError(f"Missing reinforcement station groups: {groups_path}")
    groups = pd.read_csv(groups_path)
    required = {"station_index", "station_id", "zero_ratio", "zero_group_index", "zero_group"}
    missing = sorted(required - set(groups.columns))
    if missing:
        raise ValueError(f"{groups_path} is missing columns: {missing}")
    if int(groups["station_index"].max()) >= labels.shape[1]:
        raise AssertionError("station group index exceeds labels station dimension")
    return groups.sort_values(["zero_group_index", "station_index"]).reset_index(drop=True)


def build_phase1_by_station_group(
    decisions: dict[str, np.ndarray],
    labels: np.ndarray,
    ratios: list[CostRatio],
    groups_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for group_idx, group in groups_df.groupby("zero_group_index", sort=True):
        station_indices = group["station_index"].to_numpy(dtype=int)
        group_labels = labels[:, station_indices, :]
        baseline_by_true: dict[str, float] = {}
        for true in ratios:
            cost = newsvendor_cost(decisions[true.label][:, station_indices, :], group_labels, c_u=true.c_u, c_o=true.c_o)
            baseline_by_true[true.label] = float(np.mean(cost))
        for assumed in ratios:
            for true in ratios:
                cost = newsvendor_cost(decisions[assumed.label][:, station_indices, :], group_labels, c_u=true.c_u, c_o=true.c_o)
                expected_cost = float(np.mean(cost))
                baseline = baseline_by_true[true.label]
                regret = expected_cost - baseline
                rows.append(
                    {
                        "zero_group_index": int(group_idx),
                        "zero_group": str(group.iloc[0]["zero_group"]),
                        "station_count": int(len(station_indices)),
                        "mean_station_zero_ratio": float(group["zero_ratio"].mean()),
                        "assumed_cost_ratio": assumed.label,
                        "true_cost_ratio": true.label,
                        "assumed_tau": float(assumed.tau_star),
                        "true_tau": float(true.tau_star),
                        "assumption_error": "diagonal"
                        if assumed.label == true.label
                        else ("underestimate" if assumed.tau_star < true.tau_star else "overestimate"),
                        "expected_cost": expected_cost,
                        "baseline_true_aligned_cost": baseline,
                        "regret_vs_true_aligned": float(regret),
                        "pct_regret_vs_true_aligned": float(100.0 * regret / baseline) if baseline else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def build_asymmetry_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    off_diag = long_df[long_df["assumption_error"].isin(["underestimate", "overestimate"])].copy()
    rows = []
    for direction, group in off_diag.groupby("assumption_error", sort=True):
        rows.append(
            {
                "assumption_error": direction,
                "cell_count": int(len(group)),
                "mean_regret": float(group["regret_vs_true_aligned"].mean()),
                "max_regret": float(group["regret_vs_true_aligned"].max()),
                "mean_pct_regret": float(group["pct_regret_vs_true_aligned"].mean()),
                "max_pct_regret": float(group["pct_regret_vs_true_aligned"].max()),
                "positive_regret_share": float((group["regret_vs_true_aligned"] > 1e-12).mean()),
            }
        )
    return pd.DataFrame(rows)


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


def build_bootstrap_ci(
    decisions: dict[str, np.ndarray],
    labels: np.ndarray,
    ratios: list[CostRatio],
    date_keys: np.ndarray,
    *,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    baseline_by_true: dict[str, np.ndarray] = {}
    baseline_day_by_true: dict[str, np.ndarray] = {}
    for true in ratios:
        baseline_cost = newsvendor_cost(decisions[true.label], labels, c_u=true.c_u, c_o=true.c_o)
        baseline_by_true[true.label] = baseline_cost
        _, baseline_day = aggregate_by_date(np.transpose(baseline_cost, (0, 2, 1)), date_keys)
        baseline_day_by_true[true.label] = baseline_day

    rows = []
    for assumed in ratios:
        for true in ratios:
            cost = newsvendor_cost(decisions[assumed.label], labels, c_u=true.c_u, c_o=true.c_o)
            _, cost_day = aggregate_by_date(np.transpose(cost, (0, 2, 1)), date_keys)
            regret_day = cost_day - baseline_day_by_true[true.label]
            point, low, high = bootstrap_mean_ci(regret_day, rng=rng, n_samples=n_samples)
            rows.append(
                {
                    "bootstrap_level": "day",
                    "metric": "regret_vs_true_aligned",
                    "assumed_cost_ratio": assumed.label,
                    "true_cost_ratio": true.label,
                    "assumption_error": "diagonal"
                    if assumed.label == true.label
                    else ("underestimate" if assumed.tau_star < true.tau_star else "overestimate"),
                    "point": point,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_units": int(regret_day.size),
                    "bootstrap_samples": int(n_samples),
                }
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                pct_day = 100.0 * np.divide(regret_day, baseline_day_by_true[true.label])
            pct_day = pct_day[np.isfinite(pct_day)]
            point, low, high = bootstrap_mean_ci(pct_day, rng=rng, n_samples=n_samples)
            rows.append(
                {
                    "bootstrap_level": "day",
                    "metric": "pct_regret_vs_true_aligned",
                    "assumed_cost_ratio": assumed.label,
                    "true_cost_ratio": true.label,
                    "assumption_error": "diagonal"
                    if assumed.label == true.label
                    else ("underestimate" if assumed.tau_star < true.tau_star else "overestimate"),
                    "point": point,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_units": int(pct_day.size),
                    "bootstrap_samples": int(n_samples),
                }
            )
    return pd.DataFrame(rows)


def phase1_gate(long_df: pd.DataFrame) -> dict[str, float | str]:
    off_diag = long_df[long_df["assumption_error"] != "diagonal"].copy()
    max_pct = float(off_diag["pct_regret_vs_true_aligned"].max())
    min_pct = float(off_diag["pct_regret_vs_true_aligned"].min())
    max_row = off_diag.loc[off_diag["pct_regret_vs_true_aligned"].idxmax()]
    if max_pct > GATE_PASS_PCT:
        status = "PASSED"
    elif max_pct >= GATE_MARGINAL_PCT:
        status = "MARGINAL"
    else:
        status = "FAILED"
    return {
        "status": status,
        "max_off_diagonal_pct_regret": max_pct,
        "min_off_diagonal_pct_regret": min_pct,
        "max_pair_assumed_cost_ratio": str(max_row["assumed_cost_ratio"]),
        "max_pair_true_cost_ratio": str(max_row["true_cost_ratio"]),
        "max_pair_assumption_error": str(max_row["assumption_error"]),
        "pass_threshold_pct": GATE_PASS_PCT,
        "marginal_threshold_pct": GATE_MARGINAL_PCT,
    }


def write_phase1_plots(
    regret_matrix_df: pd.DataFrame,
    horizon_df: pd.DataFrame,
    asymmetry_df: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    written: list[str] = []
    heatmap = regret_matrix_df.set_index("assumed_cost_ratio").astype(float)
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    sns.heatmap(heatmap, annot=True, fmt=".3f", cmap="crest", ax=ax)
    ax.set_xlabel("True cost ratio")
    ax.set_ylabel("Assumed cost ratio")
    ax.set_title("Cost-ratio misspecification regret")
    path = output_dir / "misspecification_regret_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    written.append(path.name)

    horizons = list(dict.fromkeys(horizon_df["horizon"].astype(int).tolist()))
    n_cols = len(horizons)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.6 * n_cols, 3.6), sharey=True)
    if n_cols == 1:
        axes = [axes]
    for ax, horizon in zip(axes, horizons):
        subset = horizon_df[horizon_df["horizon"].astype(int) == int(horizon)]
        pivot = subset.pivot(index="assumed_cost_ratio", columns="true_cost_ratio", values="pct_regret_vs_true_aligned")
        sns.heatmap(pivot.astype(float), annot=True, fmt=".1f", cmap="mako", ax=ax, cbar=ax is axes[-1])
        ax.set_title(f"H{horizon}")
        ax.set_xlabel("True")
        ax.set_ylabel("Assumed")
    path = output_dir / "misspecification_by_horizon_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    written.append(path.name)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    plot_df = asymmetry_df.sort_values("assumption_error")
    ax.bar(plot_df["assumption_error"], plot_df["mean_pct_regret"], color=["#4c78a8", "#f58518"][: len(plot_df)])
    ax.set_ylabel("Mean percentage regret")
    ax.set_xlabel("Misspecification direction")
    ax.set_title("Underestimate vs overestimate asymmetry")
    path = output_dir / "misspecification_asymmetry_bar.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    written.append(path.name)
    return written


def ratio_r_value(ratio: CostRatio) -> float:
    return float(ratio.c_u / ratio.c_o)


def ratio_labels(ratios: list[CostRatio]) -> list[str]:
    return [ratio.label for ratio in ratios]


def matrix_csv_to_array(path: Path, ratios: list[CostRatio]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 1 matrix file: {path}")
    df = pd.read_csv(path)
    labels = ratio_labels(ratios)
    if "assumed_cost_ratio" not in df.columns:
        raise ValueError(f"{path} must contain an assumed_cost_ratio column")
    missing_cols = [label for label in labels if label not in df.columns]
    if missing_cols:
        raise ValueError(f"{path} is missing true-ratio columns: {missing_cols}")
    matrix = df.set_index("assumed_cost_ratio").reindex(labels)[labels].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{path} contains non-finite or missing matrix cells")
    return matrix


def strategy_midpoint(scenario: dict[str, object], ratios: list[CostRatio]) -> int:
    r_mid = (float(scenario["r_low"]) + float(scenario["r_high"])) / 2.0
    distances = np.array([abs(ratio_r_value(ratio) - r_mid) for ratio in ratios], dtype=np.float64)
    min_distance = float(np.min(distances))
    tied = np.flatnonzero(np.isclose(distances, min_distance))
    if tied.size == 1:
        return int(tied[0])
    tied_r_values = [ratio_r_value(ratios[int(idx)]) for idx in tied]
    return int(tied[int(np.argmax(tied_r_values))])


def strategy_conservative(scenario: dict[str, object], ratios: list[CostRatio]) -> int:
    valid_indices = [int(idx) for idx in scenario["valid_grid_indices"]]
    valid_r_values = [ratio_r_value(ratios[idx]) for idx in valid_indices]
    return int(valid_indices[int(np.argmax(valid_r_values))])


def strategy_minimax_regret(scenario: dict[str, object], regret_matrix: np.ndarray) -> int:
    valid_cols = [int(idx) for idx in scenario["valid_grid_indices"]]
    worst_case = np.max(regret_matrix[:, valid_cols], axis=1)
    return int(np.argmin(worst_case))


def strategy_expected_cost(scenario: dict[str, object], cost_matrix: np.ndarray) -> int:
    valid_cols = [int(idx) for idx in scenario["valid_grid_indices"]]
    mean_cost = np.mean(cost_matrix[:, valid_cols], axis=1)
    return int(np.argmin(mean_cost))


def robust_strategy_choice(
    strategy: str,
    scenario: dict[str, object],
    ratios: list[CostRatio],
    cost_matrix: np.ndarray,
    regret_matrix: np.ndarray,
) -> int:
    if strategy == "midpoint":
        return strategy_midpoint(scenario, ratios)
    if strategy == "conservative":
        return strategy_conservative(scenario, ratios)
    if strategy == "minimax_regret":
        return strategy_minimax_regret(scenario, regret_matrix)
    if strategy == "expected_cost":
        return strategy_expected_cost(scenario, cost_matrix)
    raise ValueError(f"Unsupported robust strategy: {strategy}")


def evaluate_robust_strategy(
    scenario: dict[str, object],
    strategy: str,
    chosen_idx: int,
    ratios: list[CostRatio],
    cost_matrix: np.ndarray,
    regret_matrix: np.ndarray,
    pct_regret_matrix: np.ndarray,
) -> dict[str, float | int | str]:
    valid_cols = [int(idx) for idx in scenario["valid_grid_indices"]]
    costs = cost_matrix[chosen_idx, valid_cols]
    regrets = regret_matrix[chosen_idx, valid_cols]
    pct_regrets = pct_regret_matrix[chosen_idx, valid_cols]
    oracle_costs = np.diag(cost_matrix)[valid_cols]
    worst_local_idx = int(np.argmax(pct_regrets))
    worst_true_idx = valid_cols[worst_local_idx]
    chosen = ratios[chosen_idx]
    return {
        "scenario": str(scenario["name"]),
        "r_low": float(scenario["r_low"]),
        "r_high": float(scenario["r_high"]),
        "true_ratio_count": int(len(valid_cols)),
        "strategy": strategy,
        "chosen_ratio_label": chosen.label,
        "chosen_ratio_r": ratio_r_value(chosen),
        "chosen_tau": float(chosen.tau_star),
        "chosen_grid_index": int(chosen_idx),
        "mean_cost": float(np.mean(costs)),
        "worst_case_cost": float(np.max(costs)),
        "best_case_cost": float(np.min(costs)),
        "mean_regret": float(np.mean(regrets)),
        "worst_case_regret": float(np.max(regrets)),
        "mean_pct_regret": float(np.mean(pct_regrets)),
        "worst_case_pct_regret": float(np.max(pct_regrets)),
        "mean_oracle_cost": float(np.mean(oracle_costs)),
        "worst_true_cost_ratio": ratios[worst_true_idx].label,
    }


def run_phase2_robust_strategy(
    cost_matrix: np.ndarray,
    regret_matrix: np.ndarray,
    pct_regret_matrix: np.ndarray,
    ratios: list[CostRatio],
) -> pd.DataFrame:
    rows = []
    for scenario in UNCERTAINTY_SCENARIOS:
        for strategy in ROBUST_STRATEGIES:
            chosen_idx = robust_strategy_choice(strategy, scenario, ratios, cost_matrix, regret_matrix)
            rows.append(
                evaluate_robust_strategy(
                    scenario=scenario,
                    strategy=strategy,
                    chosen_idx=chosen_idx,
                    ratios=ratios,
                    cost_matrix=cost_matrix,
                    regret_matrix=regret_matrix,
                    pct_regret_matrix=pct_regret_matrix,
                )
            )
    return pd.DataFrame(rows)


def horizon_matrices_from_long(by_horizon_df: pd.DataFrame, ratios: list[CostRatio]) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    required = {
        "horizon",
        "assumed_cost_ratio",
        "true_cost_ratio",
        "expected_cost",
        "regret_vs_true_aligned",
        "pct_regret_vs_true_aligned",
    }
    missing = sorted(required - set(by_horizon_df.columns))
    if missing:
        raise ValueError(f"misspecification_by_horizon.csv is missing columns: {missing}")

    labels = ratio_labels(ratios)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrices: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for horizon in sorted(by_horizon_df["horizon"].astype(int).unique().tolist()):
        subset = by_horizon_df[by_horizon_df["horizon"].astype(int) == int(horizon)]
        n = len(labels)
        cost = np.full((n, n), np.nan, dtype=np.float64)
        regret = np.full((n, n), np.nan, dtype=np.float64)
        pct = np.full((n, n), np.nan, dtype=np.float64)
        for row in subset.itertuples(index=False):
            assumed = str(getattr(row, "assumed_cost_ratio"))
            true = str(getattr(row, "true_cost_ratio"))
            i = label_to_idx[assumed]
            j = label_to_idx[true]
            cost[i, j] = float(getattr(row, "expected_cost"))
            regret[i, j] = float(getattr(row, "regret_vs_true_aligned"))
            pct[i, j] = float(getattr(row, "pct_regret_vs_true_aligned"))
        if not (np.all(np.isfinite(cost)) and np.all(np.isfinite(regret)) and np.all(np.isfinite(pct))):
            raise ValueError(f"Horizon {horizon} has missing Phase 1 matrix cells")
        matrices[int(horizon)] = (cost, regret, pct)
    return matrices


def run_phase2_by_horizon(by_horizon_df: pd.DataFrame, ratios: list[CostRatio]) -> pd.DataFrame:
    rows = []
    for horizon, (cost, regret, pct) in horizon_matrices_from_long(by_horizon_df, ratios).items():
        horizon_results = run_phase2_robust_strategy(cost, regret, pct, ratios)
        horizon_results.insert(0, "horizon", int(horizon))
        rows.append(horizon_results)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_phase2_plot(results_df: pd.DataFrame, output_dir: Path) -> str:
    import matplotlib.pyplot as plt
    import numpy as np

    scenarios = [str(scenario["name"]) for scenario in UNCERTAINTY_SCENARIOS]
    strategy_labels = {
        "midpoint": "Midpoint",
        "conservative": "Conservative",
        "minimax_regret": "Minimax regret",
        "expected_cost": "Expected cost",
    }
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]

    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    x = np.arange(len(scenarios))
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
    for offset, strategy, color in zip(offsets, ROBUST_STRATEGIES, colors):
        values = []
        annotations = []
        for scenario in scenarios:
            row = results_df[(results_df["scenario"] == scenario) & (results_df["strategy"] == strategy)]
            if len(row) != 1:
                raise ValueError(f"Expected one Phase 2 row for scenario={scenario}, strategy={strategy}; got {len(row)}")
            values.append(float(row.iloc[0]["worst_case_pct_regret"]))
            annotations.append(str(row.iloc[0]["chosen_ratio_label"]))
        bars = ax.bar(x + offset, values, width, label=strategy_labels[strategy], color=color)
        for bar, label in zip(bars, annotations):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 1.0,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    x_labels = [f"{scenario['name']}\n[{scenario['r_low']:.0f}:1-{scenario['r_high']:.0f}:1]" for scenario in UNCERTAINTY_SCENARIOS]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Uncertainty scenario")
    ax.set_ylabel("Worst-case percentage regret (%)")
    ax.set_title("Robust strategy comparison under cost-ratio uncertainty")
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)
    path = output_dir / "robust_strategy_comparison_chart.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path.name


def phase2_summary(results_df: pd.DataFrame, by_horizon_df: pd.DataFrame) -> dict[str, object]:
    rows = []
    for scenario, group in results_df.groupby("scenario", sort=False):
        best_wc = group.loc[group["worst_case_pct_regret"].idxmin()]
        best_mean = group.loc[group["mean_cost"].idxmin()]
        worst = group.loc[group["worst_case_pct_regret"].idxmax()]
        rows.append(
            {
                "scenario": str(scenario),
                "best_worst_case_strategy": str(best_wc["strategy"]),
                "best_worst_case_chosen_ratio": str(best_wc["chosen_ratio_label"]),
                "best_worst_case_pct_regret": float(best_wc["worst_case_pct_regret"]),
                "best_mean_cost_strategy": str(best_mean["strategy"]),
                "best_mean_cost_chosen_ratio": str(best_mean["chosen_ratio_label"]),
                "worst_strategy": str(worst["strategy"]),
                "worst_strategy_pct_regret": float(worst["worst_case_pct_regret"]),
                "worst_minus_best_pct_points": float(worst["worst_case_pct_regret"] - best_wc["worst_case_pct_regret"]),
            }
        )

    horizon_best = []
    if not by_horizon_df.empty:
        for (horizon, scenario), group in by_horizon_df.groupby(["horizon", "scenario"], sort=True):
            best = group.loc[group["worst_case_pct_regret"].idxmin()]
            horizon_best.append(
                {
                    "horizon": int(horizon),
                    "scenario": str(scenario),
                    "best_worst_case_strategy": str(best["strategy"]),
                    "chosen_ratio_label": str(best["chosen_ratio_label"]),
                    "worst_case_pct_regret": float(best["worst_case_pct_regret"]),
                }
            )

    return {
        "aggregate": rows,
        "by_horizon_best_worst_case": horizon_best,
    }


def run_phase2_from_outputs(output_dir: Path, ratios: list[CostRatio]) -> tuple[pd.DataFrame, pd.DataFrame, str, dict[str, object]]:
    cost_matrix = matrix_csv_to_array(output_dir / "misspecification_cost_matrix.csv", ratios)
    regret_matrix = matrix_csv_to_array(output_dir / "misspecification_regret_matrix.csv", ratios)
    pct_regret_matrix = matrix_csv_to_array(output_dir / "misspecification_pct_regret_matrix.csv", ratios)
    results_df = run_phase2_robust_strategy(cost_matrix, regret_matrix, pct_regret_matrix, ratios)
    by_horizon_raw = pd.read_csv(output_dir / "misspecification_by_horizon.csv")
    by_horizon_df = run_phase2_by_horizon(by_horizon_raw, ratios)
    results_df.to_csv(output_dir / "robust_strategy_comparison.csv", index=False)
    by_horizon_df.to_csv(output_dir / "robust_strategy_comparison_by_horizon.csv", index=False)
    plot = write_phase2_plot(results_df, output_dir)
    return results_df, by_horizon_df, plot, phase2_summary(results_df, by_horizon_df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cost-ratio misspecification experiments for the journal branch.")
    parser.add_argument("--stage1-output-dir", default="journal_results/shenzhen_multihorizon/warmstart_raw")
    parser.add_argument("--stage4-output-dir", default="journal_results/shenzhen_multihorizon/stage4_decision/warmstart_raw")
    parser.add_argument("--reinforcement-dir", default="journal_results/shenzhen_multihorizon/reinforcement")
    parser.add_argument("--data-dir", default="data/datasets/ST_EVCDP_v2_canonical")
    parser.add_argument("--output-dir", default="journal_results/shenzhen_multihorizon/misspecification")
    parser.add_argument("--phases", default="1")
    parser.add_argument("--include-interpolated-ratios", default="false")
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--bootstrap-seed", type=int, default=20260507)
    parser.add_argument("--machine", default="Lenovo")
    args = parser.parse_args()

    phases = parse_phases(args.phases)
    if 3 in phases:
        raise NotImplementedError("Phase 3 is intentionally not implemented until Phase 2 shows meaningful strategy differences.")

    stage1_output_dir = resolve_path(args.stage1_output_dir)
    stage4_output_dir = resolve_path(args.stage4_output_dir)
    reinforcement_dir = resolve_path(args.reinforcement_dir)
    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    include_interpolated_ratios = parse_bool(args.include_interpolated_ratios)
    if 2 in phases and include_interpolated_ratios:
        raise ValueError("Official Phase 2 uses the existing five-point grid only; set --include-interpolated-ratios false.")
    ratios = build_cost_ratio_grid(include_interpolated_ratios=include_interpolated_ratios)

    required_stage4 = [
        "decision_summary_by_method_and_cost_ratio.csv",
        "decision_metrics_by_horizon.csv",
        "decision_regret_vs_oracle.csv",
        "decision_one_sided_thresholds.csv",
        "decision_eval_metadata.json",
    ]
    stage4_presence = {name: (stage4_output_dir / name).exists() for name in required_stage4}
    existing_metadata_path = output_dir / "misspecification_metadata.json"
    existing_metadata = load_json(existing_metadata_path) if existing_metadata_path.exists() else {}
    metadata_out = {
        "stage": "misspecification",
        "goal": "Cost-ratio misspecification and robust quantile-selection experiments",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "machine": args.machine,
        "branch": git_value(["branch", "--show-current"]),
        "commit": git_value(["rev-parse", "--short", "HEAD"]),
        "stage1_output_dir": str(stage1_output_dir),
        "stage4_output_dir": str(stage4_output_dir),
        "reinforcement_dir": str(reinforcement_dir),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "phases": phases,
        "include_interpolated_ratios": include_interpolated_ratios,
        "cost_ratios": [ratio.__dict__ for ratio in ratios],
        "stage4_consistency_files_present": stage4_presence,
        "previous_gate": existing_metadata.get("gate") or existing_metadata.get("phase1", {}).get("gate"),
        "outputs": {},
    }

    if 1 in phases:
        metadata = load_json(stage1_output_dir / "run_metadata.json")
        horizons = [int(horizon) for horizon in metadata["horizons"]]
        quantiles = [float(q) for q in metadata["quantiles"]]
        predict_quantiles = np.load(stage1_output_dir / "predict_quantiles.npy")
        labels = np.load(stage1_output_dir / "label_list.npy")
        assert_quantile_tensor_shape(
            predict_quantiles,
            n_horizons=len(horizons),
            n_quantiles=len(quantiles),
            context="misspecification predict_quantiles",
        )
        assert_target_tensor_shape(labels, n_horizons=len(horizons), context="misspecification labels")
        assert_monotonic_quantiles(predict_quantiles, context="misspecification predict_quantiles")

        decisions: dict[str, np.ndarray] = {}
        quantile_selection_rows = []
        for ratio in ratios:
            decision, info = decision_for_ratio(
                predict_quantiles,
                quantiles,
                ratio,
                allow_interpolation=include_interpolated_ratios,
            )
            decisions[ratio.label] = decision
            quantile_selection_rows.append(info)

        cost_df, regret_df, pct_regret_df, long_df = build_phase1_aggregate(decisions, labels, ratios)
        horizon_df = build_phase1_by_horizon(decisions, labels, ratios, horizons)
        groups_df = load_station_groups(reinforcement_dir, labels)
        station_group_df = build_phase1_by_station_group(decisions, labels, ratios, groups_df)
        asymmetry_df = build_asymmetry_summary(long_df)
        manifest_df = pd.read_csv(reinforcement_dir / "test_window_manifest.csv")
        date_keys = date_keys_from_manifest(manifest_df, n_test_windows=labels.shape[0], horizons=horizons)
        bootstrap_df = build_bootstrap_ci(
            decisions,
            labels,
            ratios,
            date_keys,
            n_samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
        gate = phase1_gate(long_df)
        plots = write_phase1_plots(regret_df, horizon_df, asymmetry_df, output_dir)

        cost_df.to_csv(output_dir / "misspecification_cost_matrix.csv", index=False)
        regret_df.to_csv(output_dir / "misspecification_regret_matrix.csv", index=False)
        pct_regret_df.to_csv(output_dir / "misspecification_pct_regret_matrix.csv", index=False)
        horizon_df.to_csv(output_dir / "misspecification_by_horizon.csv", index=False)
        station_group_df.to_csv(output_dir / "misspecification_by_station_group.csv", index=False)
        asymmetry_df.to_csv(output_dir / "misspecification_asymmetry.csv", index=False)
        bootstrap_df.to_csv(output_dir / "misspecification_bootstrap_ci.csv", index=False)

        metadata_out["horizons"] = horizons
        metadata_out["quantiles"] = quantiles
        metadata_out["bootstrap_level"] = "day"
        metadata_out["bootstrap_samples"] = int(args.bootstrap_samples)
        metadata_out["bootstrap_seed"] = int(args.bootstrap_seed)
        metadata_out["gate"] = gate
        metadata_out["phase1"] = {
            "gate": gate,
            "quantile_selection": quantile_selection_rows,
            "input_shapes": {
                "predict_quantiles": list(predict_quantiles.shape),
                "labels": list(labels.shape),
                "date_keys": list(date_keys.shape),
                "station_groups": int(groups_df["zero_group"].nunique()),
            },
            "outputs": {
                "cost_matrix": "misspecification_cost_matrix.csv",
                "regret_matrix": "misspecification_regret_matrix.csv",
                "pct_regret_matrix": "misspecification_pct_regret_matrix.csv",
                "by_horizon": "misspecification_by_horizon.csv",
                "by_station_group": "misspecification_by_station_group.csv",
                "asymmetry": "misspecification_asymmetry.csv",
                "bootstrap_ci": "misspecification_bootstrap_ci.csv",
                "plots": plots,
            },
        }
        metadata_out["outputs"].update(metadata_out["phase1"]["outputs"])

    if 2 in phases:
        results_df, by_horizon_df, plot, summary = run_phase2_from_outputs(output_dir, ratios)
        metadata_out["phase2"] = {
            "goal": "Robust quantile selection under cost-ratio uncertainty",
            "strategy_selection_rule": "true ratios restricted to scenario range; deployed assumed ratio chosen from full five-point grid",
            "scenarios": UNCERTAINTY_SCENARIOS,
            "strategies": ROBUST_STRATEGIES,
            "aggregate_rows": int(len(results_df)),
            "by_horizon_rows": int(len(by_horizon_df)),
            "summary": summary,
            "outputs": {
                "robust_strategy_comparison": "robust_strategy_comparison.csv",
                "robust_strategy_comparison_by_horizon": "robust_strategy_comparison_by_horizon.csv",
                "robust_strategy_comparison_chart": plot,
            },
        }
        metadata_out["outputs"].update(metadata_out["phase2"]["outputs"])

    (output_dir / "misspecification_metadata.json").write_text(json.dumps(metadata_out, indent=2), encoding="utf-8")

    print(json.dumps(metadata_out, indent=2))


if __name__ == "__main__":
    main()
