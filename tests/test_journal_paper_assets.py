import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.journal.build_paper_evidence_assets import (  # noqa: E402
    bootstrap_ratio_ci,
    build_coverage_gain_decomposition,
    build_decision_value_attribution,
    build_hard_cell_summary,
    subset_masks,
)


def test_subset_masks_define_zero_positive_and_tail_bins():
    labels = np.array([[0.0, 1.0, 2.0, 10.0, 20.0]], dtype=np.float32)

    masks, thresholds = subset_masks(labels)
    by_name = {name: mask for name, mask in masks}

    assert int(by_name["zero"].sum()) == 1
    assert int(by_name["positive"].sum()) == 4
    assert int(by_name["low_positive"].sum()) == 2
    assert int(by_name["top10"].sum()) == 1
    assert int(by_name["top5"].sum()) == 1
    assert thresholds["positive_q50"] == 6.0


def test_coverage_gain_decomposition_splits_zero_and_positive_rescues():
    boundary = pd.DataFrame(
        [
            {
                "method": "global_cqr",
                "delta": 0.1,
                "horizon": 1,
                "total_count": 100,
                "raw_miss_rate": 0.4,
                "zero_rescued_count": 30,
                "positive_rescued_count": 5,
                "zero_share_of_rescues": 30 / 35,
                "clipping_share_of_rescues": 30 / 35,
                "rescued_share_of_raw_misses": 35 / 40,
            }
        ]
    )

    out = build_coverage_gain_decomposition(boundary).iloc[0]

    assert out["raw_PICP"] == 0.6
    assert out["zero_boundary_rescue_gain"] == 0.3
    assert out["positive_rescue_gain"] == 0.05
    assert out["calibrated_PICP_from_rescues"] == 0.95


def test_decision_value_attribution_computes_gain_shares():
    summary = pd.DataFrame(
        [
            {"method": "median", "cost_ratio": "3:1", "c_u": 3, "c_o": 1, "tau_star": 0.75, "expected_cost": 10.0},
            {"method": "raw_target_quantile", "cost_ratio": "3:1", "c_u": 3, "c_o": 1, "tau_star": 0.75, "expected_cost": 7.0},
            {"method": "global_cqr", "cost_ratio": "3:1", "c_u": 3, "c_o": 1, "tau_star": 0.75, "expected_cost": 6.8},
            {"method": "horizon_cqr", "cost_ratio": "3:1", "c_u": 3, "c_o": 1, "tau_star": 0.75, "expected_cost": 6.5},
            {"method": "oracle_decision", "cost_ratio": "3:1", "c_u": 3, "c_o": 1, "tau_star": 0.75, "expected_cost": 0.0},
        ]
    )

    out = build_decision_value_attribution(summary).iloc[0]

    assert out["quantile_choice_gain"] == 3.0
    assert out["horizon_calibration_gain"] == 0.5
    assert out["total_horizon_gain_vs_median"] == 3.5
    assert abs(out["quantile_choice_share_of_horizon_gain"] - 3.0 / 3.5) < 1e-12


def test_hard_cell_summary_selects_largest_ace():
    cells = pd.DataFrame(
        [
            {"method": "global_cqr", "delta": 0.1, "horizon": 1, "hour": 1, "ACE": 0.01},
            {"method": "global_cqr", "delta": 0.1, "horizon": 1, "hour": 2, "ACE": 0.20},
            {"method": "horizon_cqr", "delta": 0.1, "horizon": 3, "hour": 3, "ACE": 0.10},
        ]
    )

    out = build_hard_cell_summary(cells, delta=0.1, top_k=2)

    assert list(out["ACE"]) == [0.20, 0.10]


def test_bootstrap_ratio_ci_uses_sum_ratio():
    rng = np.random.default_rng(7)
    numerator = np.array([1.0, 1.0, 0.0])
    denominator = np.array([2.0, 2.0, 2.0])

    point, low, high = bootstrap_ratio_ci(numerator, denominator, rng=rng, n_samples=20)

    assert abs(point - 2.0 / 6.0) < 1e-12
    assert low <= point <= high
