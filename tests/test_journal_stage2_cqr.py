import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.journal.calibrate_multihorizon_cqr import (  # noqa: E402
    apply_cqr_threshold,
    build_comparison_table,
    compute_cqr_scores,
    compute_global_threshold,
    compute_horizon_thresholds,
    evaluate_interval_methods,
)


QUANTILES = [0.1, 0.5, 0.9]


def test_cqr_score_preserves_horizon_axis():
    pred_q = np.array(
        [
            [
                [[1.0, 2.0, 3.0], [10.0, 11.0, 12.0]],
                [[2.0, 3.0, 4.0], [20.0, 21.0, 22.0]],
            ]
        ],
        dtype=np.float32,
    )
    labels = np.array([[[5.0, 13.0], [1.0, 25.0]]], dtype=np.float32)

    scores = compute_cqr_scores(pred_q, labels, quantiles=QUANTILES, delta=0.2)

    assert scores.shape == (1, 2, 2)
    np.testing.assert_allclose(scores[0, :, 0], [2.0, 1.0])
    np.testing.assert_allclose(scores[0, :, 1], [1.0, 3.0])


def test_global_and_horizon_thresholds_use_expected_score_pools():
    scores = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
        ],
        dtype=np.float32,
    )

    global_s = compute_global_threshold(scores, delta=0.1)
    horizon_s = compute_horizon_thresholds(scores, horizons=[1, 3], delta=0.1)

    assert global_s == 40.0
    assert horizon_s[1] == 4.0
    assert horizon_s[3] == 40.0


def test_nonzero_s_hat_expands_interval():
    lower = np.array([[2.0, 4.0]], dtype=np.float32)
    upper = np.array([[5.0, 8.0]], dtype=np.float32)

    cal_lower, cal_upper = apply_cqr_threshold(lower, upper, s_hat=1.5, nonnegative=False)

    np.testing.assert_allclose(cal_lower, [[0.5, 2.5]])
    np.testing.assert_allclose(cal_upper, [[6.5, 9.5]])
    np.testing.assert_allclose(cal_upper - cal_lower, (upper - lower) + 3.0)


def test_evaluate_interval_methods_reports_cqr_improvements():
    test_q = np.array(
        [
            [
                [[1.0, 2.0, 3.0], [10.0, 11.0, 12.0]],
                [[1.0, 2.0, 3.0], [10.0, 11.0, 12.0]],
            ]
        ],
        dtype=np.float32,
    )
    labels = np.array([[[4.0, 14.0], [2.0, 11.0]]], dtype=np.float32)

    metrics_df = evaluate_interval_methods(
        test_quantiles=test_q,
        test_labels=labels,
        quantiles=QUANTILES,
        horizons=[1, 3],
        deltas=[0.2],
        global_thresholds={0.2: 2.0},
        horizon_thresholds={0.2: {1: 1.0, 3: 3.0}},
    )
    comparison_df = build_comparison_table(metrics_df)

    assert set(metrics_df["method"]) == {"raw", "global_cqr", "horizon_cqr"}
    raw_h1 = metrics_df[(metrics_df["method"] == "raw") & (metrics_df["horizon"] == 1)].iloc[0]
    horizon_h1 = metrics_df[(metrics_df["method"] == "horizon_cqr") & (metrics_df["horizon"] == 1)].iloc[0]
    horizon_h3 = metrics_df[(metrics_df["method"] == "horizon_cqr") & (metrics_df["horizon"] == 3)].iloc[0]

    assert raw_h1["PICP"] == 0.5
    assert horizon_h1["PICP"] == 1.0
    assert horizon_h3["s_hat"] == 3.0
    assert (comparison_df["PICP_gain_vs_raw"] >= 0.0).all()
