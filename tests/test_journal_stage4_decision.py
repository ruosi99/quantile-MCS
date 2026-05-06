import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.journal.evaluate_stage4_decision import (  # noqa: E402
    CostRatio,
    build_one_sided_thresholds,
    build_summary_table,
    compute_one_sided_global_threshold,
    compute_one_sided_horizon_thresholds,
    decision_metric_values,
    evaluate_decision_methods,
    newsvendor_cost,
    parse_cost_ratios,
)


def test_parse_cost_ratios_maps_to_exact_target_quantiles():
    ratios = parse_cost_ratios("1:1,3:1,5:1")

    assert [ratio.label for ratio in ratios] == ["1:1", "3:1", "5:1"]
    assert ratios[0].tau_star == 0.5
    assert ratios[1].tau_star == 0.75
    assert abs(ratios[2].tau_star - 0.8333333333333334) < 1e-12


def test_newsvendor_cost_and_regret_against_oracle():
    decision = np.array([1.0, 5.0, 3.0])
    labels = np.array([3.0, 2.0, 3.0])

    cost = newsvendor_cost(decision, labels, c_u=3.0, c_o=1.0)
    values = decision_metric_values(decision, labels, c_u=3.0, c_o=1.0)

    np.testing.assert_allclose(cost, [6.0, 3.0, 0.0])
    assert values["expected_cost"] == 3.0
    assert values["regret"] == 3.0
    assert values["event_shortage_rate"] == 1 / 3
    assert values["event_overage_rate"] == 1 / 3


def test_one_sided_thresholds_use_global_and_horizon_pools():
    scores = np.array(
        [
            [[0.0, 10.0], [1.0, 20.0]],
            [[2.0, 30.0], [3.0, 40.0]],
        ],
        dtype=np.float32,
    )

    global_s = compute_one_sided_global_threshold(scores, tau_star=0.75)
    horizon_s = compute_one_sided_horizon_thresholds(scores, horizons=[1, 3], tau_star=0.75)

    assert global_s == 30.0
    assert horizon_s[1] == 3.0
    assert horizon_s[3] == 40.0


def test_build_one_sided_thresholds_records_target_quantile():
    quantiles = [0.5, 0.75]
    horizons = [1]
    calib_q = np.array(
        [
            [[[1.0, 2.0]], [[1.0, 2.0]], [[1.0, 2.0]]],
        ],
        dtype=np.float32,
    )
    labels = np.array([[[1.0], [3.0], [5.0]]], dtype=np.float32)

    thresholds_df, lookup = build_one_sided_thresholds(
        calib_quantiles=calib_q,
        calib_labels=labels,
        quantiles=quantiles,
        horizons=horizons,
        cost_ratios=[CostRatio(label="3:1", c_u=3.0, c_o=1.0, tau_star=0.75)],
    )

    assert set(thresholds_df["method"]) == {"global_cqr", "horizon_cqr"}
    assert lookup["3:1"]["q_idx"] == 1
    assert lookup["3:1"]["q_value"] == 0.75
    assert lookup["3:1"]["global"] == 3.0


def test_evaluate_decision_methods_reports_calibrated_cost_gain():
    quantiles = [0.5, 0.75]
    horizons = [1]
    test_q = np.array(
        [
            [[[2.0, 2.0]], [[2.0, 2.0]]],
        ],
        dtype=np.float32,
    )
    labels = np.array([[[4.0], [1.0]]], dtype=np.float32)
    ratios = [CostRatio(label="3:1", c_u=3.0, c_o=1.0, tau_star=0.75)]
    threshold_lookup = {
        "3:1": {
            "q_idx": 1,
            "q_value": 0.75,
            "global": 2.0,
            "horizon": {1: 2.0},
        }
    }

    metrics_df = evaluate_decision_methods(
        test_quantiles=test_q,
        labels=labels,
        quantiles=quantiles,
        horizons=horizons,
        cost_ratios=ratios,
        threshold_lookup=threshold_lookup,
    )
    summary_df = build_summary_table(metrics_df)

    raw = metrics_df[metrics_df["method"] == "raw_target_quantile"].iloc[0]
    global_row = metrics_df[metrics_df["method"] == "global_cqr"].iloc[0]
    oracle = metrics_df[metrics_df["method"] == "oracle_decision"].iloc[0]

    assert raw["expected_cost"] == 3.5
    assert global_row["expected_cost"] == 1.5
    assert oracle["expected_cost"] == 0.0
    assert set(summary_df["method"]) == {
        "median",
        "raw_target_quantile",
        "global_cqr",
        "horizon_cqr",
        "oracle_decision",
    }
