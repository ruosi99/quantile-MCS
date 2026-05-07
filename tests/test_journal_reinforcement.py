import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.journal.build_reinforcement_experiments import (  # noqa: E402
    aggregate_by_date,
    build_deployment_strategy_comparison,
    station_zero_groups,
)
from scripts.journal.evaluate_stage4_decision import parse_cost_ratios  # noqa: E402


def test_station_zero_groups_assigns_ordered_quantile_groups():
    labels = np.array(
        [
            [[0.0], [1.0], [0.0], [0.0]],
            [[1.0], [1.0], [0.0], [0.0]],
            [[1.0], [1.0], [1.0], [0.0]],
            [[1.0], [1.0], [1.0], [1.0]],
        ],
        dtype=np.float32,
    )

    groups = station_zero_groups(labels, station_ids=["a", "b", "c", "d"], n_groups=2)

    assert set(groups["zero_group"]) == {"G1_low_zero", "G2_high_zero"}
    assert groups.loc[groups["station_id"] == "b", "zero_ratio"].iloc[0] == 0.0
    assert groups.loc[groups["station_id"] == "d", "zero_ratio"].iloc[0] == 0.75


def test_deployment_strategy_comparison_reports_expected_strategies():
    quantiles = [0.05, 0.5, 0.75, 0.95]
    horizons = [1]
    pred = np.array(
        [
            [[[0.0, 2.0, 3.0, 5.0]], [[0.0, 2.0, 3.0, 5.0]]],
        ],
        dtype=np.float32,
    )
    labels = np.array([[[4.0], [1.0]]], dtype=np.float32)
    thresholds = pd.DataFrame(
        [
            {"method": "global_cqr", "delta": 0.1, "horizon": "all", "s_hat": 0.0},
            {"method": "horizon_cqr", "delta": 0.1, "horizon": 1, "s_hat": 0.0},
        ]
    )
    decision_thresholds = pd.DataFrame(
        [
            {
                "method": "global_cqr",
                "cost_ratio": "3:1",
                "c_u": 3.0,
                "c_o": 1.0,
                "tau_star": 0.75,
                "target_quantile": 0.75,
                "horizon": "all",
                "s_hat": 0.0,
            },
            {
                "method": "horizon_cqr",
                "cost_ratio": "3:1",
                "c_u": 3.0,
                "c_o": 1.0,
                "tau_star": 0.75,
                "target_quantile": 0.75,
                "horizon": 1,
                "s_hat": 0.0,
            },
        ]
    )

    summary, by_horizon = build_deployment_strategy_comparison(
        predict_quantiles=pred,
        labels=labels,
        quantiles=quantiles,
        horizons=horizons,
        thresholds_df=thresholds,
        decision_thresholds_df=decision_thresholds,
        cost_ratios=parse_cost_ratios("3:1"),
        delta=0.1,
    )

    assert set(summary["strategy"]) == {
        "median_decision",
        "symmetric_interval_upper_bound",
        "raw_cost_aligned_quantile",
        "one_sided_refined_cost_aligned_quantile",
    }
    raw = summary[summary["strategy"] == "raw_cost_aligned_quantile"].iloc[0]
    median = summary[summary["strategy"] == "median_decision"].iloc[0]
    assert raw["expected_cost"] < median["expected_cost"]
    assert len(by_horizon) == 4


def test_aggregate_by_date_averages_station_and_horizon_cells():
    values = np.array(
        [
            [[1.0, 3.0, 5.0], [7.0, 9.0, 11.0]],
            [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]],
        ]
    )  # (T, H, N)
    date_keys = np.array([["d1", "d1"], ["d2", "d2"]])

    dates, out = aggregate_by_date(values, date_keys)

    assert list(dates) == ["d1", "d2"]
    np.testing.assert_allclose(out, [6.0, 7.0])
