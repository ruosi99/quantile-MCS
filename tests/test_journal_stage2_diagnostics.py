import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.journal.analyze_stage2_diagnostics import (  # noqa: E402
    build_boundary_rescue_summary,
    build_demand_bin_summary,
    build_zero_positive_summary,
    demand_bin_masks,
)


QUANTILES = [0.1, 0.5, 0.9]
HORIZONS = [1]


def _toy_thresholds():
    return pd.DataFrame(
        [
            {"method": "global_cqr", "delta": 0.2, "horizon": "all", "s_hat": 0.02},
            {"method": "horizon_cqr", "delta": 0.2, "horizon": 1, "s_hat": 0.02},
        ]
    )


def _toy_predictions_and_labels():
    pred_q = np.array(
        [
            [
                [[0.01, 1.0, 2.0]],
                [[0.50, 1.0, 2.0]],
                [[3.00, 3.5, 4.0]],
            ]
        ],
        dtype=np.float32,
    )
    labels = np.array([[[0.0], [1.0], [5.0]]], dtype=np.float32)
    return pred_q, labels


def test_boundary_rescue_counts_zero_clipping_rescue():
    pred_q, labels = _toy_predictions_and_labels()

    df = build_boundary_rescue_summary(
        predict_quantiles=pred_q,
        labels=labels,
        thresholds_df=_toy_thresholds(),
        quantiles=QUANTILES,
        horizons=HORIZONS,
        deltas=[0.2],
        positive_eps=1e-12,
    )
    global_row = df[df["method"] == "global_cqr"].iloc[0]

    assert global_row["raw_miss_count"] == 2
    assert global_row["rescued_count"] == 1
    assert global_row["zero_rescued_count"] == 1
    assert global_row["positive_rescued_count"] == 0
    assert global_row["clipping_rescued_count"] == 1
    assert global_row["zero_share_of_rescues"] == 1.0


def test_zero_positive_summary_separates_positive_coverage():
    pred_q, labels = _toy_predictions_and_labels()

    df = build_zero_positive_summary(
        predict_quantiles=pred_q,
        labels=labels,
        thresholds_df=_toy_thresholds(),
        quantiles=QUANTILES,
        horizons=HORIZONS,
        deltas=[0.2],
        positive_eps=1e-12,
    )
    raw = df[df["method"] == "raw"].iloc[0]
    global_row = df[df["method"] == "global_cqr"].iloc[0]

    assert raw["count_zero"] == 1
    assert raw["count_positive"] == 2
    assert raw["PICP_zero"] == 0.0
    assert raw["PICP_positive"] == 0.5
    assert global_row["PICP_zero"] == 1.0
    assert global_row["PICP_positive"] == 0.5


def test_demand_bin_masks_use_positive_quantiles():
    y = np.array([[0.0, 1.0, 2.0, 10.0]], dtype=np.float32)

    masks, thresholds = demand_bin_masks(y, positive_eps=1e-12)
    mask_by_name = {name: mask for name, mask in masks}

    assert thresholds["positive_q50"] == 2.0
    assert thresholds["positive_q90"] > 8.0
    assert int(mask_by_name["zero"].sum()) == 1
    assert int(mask_by_name["positive_le_q50"].sum()) == 2
    assert int(mask_by_name["positive_q50_q90"].sum()) == 0
    assert int(mask_by_name["positive_gt_q90"].sum()) == 1


def test_demand_bin_summary_has_expected_bins():
    pred_q, labels = _toy_predictions_and_labels()

    df = build_demand_bin_summary(
        predict_quantiles=pred_q,
        labels=labels,
        thresholds_df=_toy_thresholds(),
        quantiles=QUANTILES,
        horizons=HORIZONS,
        deltas=[0.2],
        positive_eps=1e-12,
    )

    assert set(df["demand_bin"]) == {
        "zero",
        "positive_le_q50",
        "positive_q50_q90",
        "positive_gt_q90",
    }
    assert set(df["method"]) == {"raw", "global_cqr", "horizon_cqr"}
