import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.journal.build_misspecification_experiments import (  # noqa: E402
    UNCERTAINTY_SCENARIOS,
    build_bootstrap_ci,
    build_cost_ratio_grid,
    build_phase1_aggregate,
    build_phase1_by_station_group,
    date_keys_from_manifest,
    decision_for_ratio,
    horizon_matrices_from_long,
    phase1_gate,
    quantile_prediction,
    run_phase2_by_horizon,
    run_phase2_robust_strategy,
    strategy_conservative,
    strategy_expected_cost,
    strategy_midpoint,
    strategy_minimax_regret,
)


def test_quantile_prediction_extracts_exact_quantile():
    pred = np.array([[[[1.0, 2.0, 3.0]]]], dtype=np.float32)
    values, method, q_left, q_right = quantile_prediction(pred, [0.1, 0.5, 0.9], 0.5, allow_interpolation=False)

    np.testing.assert_allclose(values, [[[2.0]]])
    assert method == "exact"
    assert q_left == 0.5
    assert q_right == 0.5


def test_quantile_prediction_interpolates_between_grid_points():
    pred = np.array([[[[10.0, 20.0]]]], dtype=np.float32)
    values, method, q_left, q_right = quantile_prediction(pred, [0.5, 0.9], 0.7, allow_interpolation=True)

    np.testing.assert_allclose(values, [[[15.0]]])
    assert method == "linear_interpolation"
    assert q_left == 0.5
    assert q_right == 0.9


def test_phase1_aggregate_has_zero_diagonal_and_nonnegative_regret_on_ordered_example():
    quantiles = [0.5, 0.75]
    ratios = build_cost_ratio_grid(False)[:2]
    pred = np.array(
        [
            [[[1.0, 3.0]], [[1.0, 3.0]]],
            [[[1.0, 3.0]], [[1.0, 3.0]]],
        ],
        dtype=np.float32,
    )
    labels = np.array([[[1.0], [4.0]], [[1.0], [4.0]]], dtype=np.float32)
    decisions = {}
    for ratio in ratios:
        decision, _ = decision_for_ratio(pred, quantiles, ratio, allow_interpolation=False)
        decisions[ratio.label] = decision

    _, regret_df, pct_df, long_df = build_phase1_aggregate(decisions, labels, ratios)

    for ratio in ratios:
        diagonal = regret_df.loc[regret_df["assumed_cost_ratio"] == ratio.label, ratio.label].iloc[0]
        assert abs(diagonal) < 1e-12
    assert (long_df["regret_vs_true_aligned"] >= -1e-12).all()
    gate = phase1_gate(long_df)
    assert gate["status"] in {"FAILED", "MARGINAL", "PASSED"}
    assert pct_df.shape == (2, 3)


def test_station_group_filtering_preserves_group_counts():
    quantiles = [0.5, 0.75]
    ratios = build_cost_ratio_grid(False)[:2]
    pred = np.array(
        [
            [[[1.0, 3.0]], [[2.0, 4.0]], [[3.0, 5.0]]],
        ],
        dtype=np.float32,
    )
    labels = np.array([[[2.0], [3.0], [4.0]]], dtype=np.float32)
    decisions = {}
    for ratio in ratios:
        decision, _ = decision_for_ratio(pred, quantiles, ratio, allow_interpolation=False)
        decisions[ratio.label] = decision
    groups = pd.DataFrame(
        [
            {"station_index": 0, "station_id": "a", "zero_ratio": 0.0, "zero_group_index": 0, "zero_group": "G1_low_zero"},
            {"station_index": 1, "station_id": "b", "zero_ratio": 0.5, "zero_group_index": 1, "zero_group": "G2_high_zero"},
            {"station_index": 2, "station_id": "c", "zero_ratio": 0.5, "zero_group_index": 1, "zero_group": "G2_high_zero"},
        ]
    )

    out = build_phase1_by_station_group(decisions, labels, ratios, groups)

    counts = out.groupby("zero_group")["station_count"].first().to_dict()
    assert counts == {"G1_low_zero": 1, "G2_high_zero": 2}


def test_date_manifest_mapping_and_day_bootstrap_use_valid_days():
    manifest = pd.DataFrame(
        [
            {"sample_index": 0, "horizon": 1, "target_date": "d1"},
            {"sample_index": 0, "horizon": 3, "target_date": "d1"},
            {"sample_index": 1, "horizon": 1, "target_date": "d2"},
            {"sample_index": 1, "horizon": 3, "target_date": "d3"},
        ]
    )
    date_keys = date_keys_from_manifest(manifest, n_test_windows=2, horizons=[1, 3])
    assert date_keys.shape == (2, 2)
    assert sorted(set(date_keys.reshape(-1))) == ["d1", "d2", "d3"]

    ratios = build_cost_ratio_grid(False)[:2]
    labels = np.array(
        [
            [[1.0, 2.0], [2.0, 3.0]],
            [[1.0, 2.0], [2.0, 3.0]],
        ],
        dtype=np.float32,
    )
    decisions = {
        ratios[0].label: labels.copy() + 0.2,
        ratios[1].label: labels.copy() + 0.1,
    }

    boot = build_bootstrap_ci(decisions, labels, ratios, date_keys, n_samples=5, seed=7)

    assert set(boot["bootstrap_level"]) == {"day"}
    assert boot["bootstrap_units"].min() == 3


def phase2_toy_matrices():
    idx = np.arange(5, dtype=np.float64)
    regret = np.abs(idx[:, None] - idx[None, :])
    baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cost = baseline[None, :] + regret
    pct = 100.0 * regret / baseline[None, :]
    return cost, regret, pct


def test_strategy_midpoint_narrow_uses_conservative_tie_break():
    ratios = build_cost_ratio_grid(False)

    idx = strategy_midpoint(UNCERTAINTY_SCENARIOS[0], ratios)

    assert ratios[idx].label == "9:1"


def test_strategy_conservative_picks_upper_ratio_in_each_scenario():
    ratios = build_cost_ratio_grid(False)

    chosen = [ratios[strategy_conservative(scenario, ratios)].label for scenario in UNCERTAINTY_SCENARIOS]

    assert chosen == ["9:1", "19:1", "19:1"]


def test_minimax_regret_is_no_worse_than_any_action_by_definition():
    _, regret, _ = phase2_toy_matrices()

    for scenario in UNCERTAINTY_SCENARIOS:
        valid_cols = scenario["valid_grid_indices"]
        chosen = strategy_minimax_regret(scenario, regret)
        chosen_worst = np.max(regret[chosen, valid_cols])
        all_worst = np.max(regret[:, valid_cols], axis=1)
        assert chosen_worst <= np.min(all_worst) + 1e-12


def test_expected_cost_is_no_worse_than_any_action_by_definition():
    cost, _, _ = phase2_toy_matrices()

    for scenario in UNCERTAINTY_SCENARIOS:
        valid_cols = scenario["valid_grid_indices"]
        chosen = strategy_expected_cost(scenario, cost)
        chosen_mean = np.mean(cost[chosen, valid_cols])
        all_mean = np.mean(cost[:, valid_cols], axis=1)
        assert chosen_mean <= np.min(all_mean) + 1e-12


def test_phase2_aggregate_output_shape():
    cost, regret, pct = phase2_toy_matrices()
    ratios = build_cost_ratio_grid(False)

    out = run_phase2_robust_strategy(cost, regret, pct, ratios)

    assert len(out) == 12
    assert set(out["scenario"]) == {"narrow", "medium", "wide"}
    assert set(out["strategy"]) == {"midpoint", "conservative", "minimax_regret", "expected_cost"}


def test_phase2_by_horizon_reconstructs_actual_phase1_column_names():
    cost, regret, pct = phase2_toy_matrices()
    ratios = build_cost_ratio_grid(False)
    rows = []
    for horizon in [1, 3, 6, 12, 24]:
        for i, assumed in enumerate(ratios):
            for j, true in enumerate(ratios):
                rows.append(
                    {
                        "horizon": horizon,
                        "assumed_cost_ratio": assumed.label,
                        "true_cost_ratio": true.label,
                        "expected_cost": cost[i, j],
                        "regret_vs_true_aligned": regret[i, j],
                        "pct_regret_vs_true_aligned": pct[i, j],
                    }
                )
    horizon_df = pd.DataFrame(rows)

    matrices = horizon_matrices_from_long(horizon_df, ratios)
    by_horizon = run_phase2_by_horizon(horizon_df, ratios)

    assert set(matrices) == {1, 3, 6, 12, 24}
    assert len(by_horizon) == 60
    assert set(by_horizon["horizon"]) == {1, 3, 6, 12, 24}
