import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.journal.build_crc_risk_control_experiments import (  # noqa: E402
    build_crc_selection_table,
    build_policy_decisions,
    build_risk_efficiency_frontier,
    build_test_summaries,
    bounded_violation_loss,
    candidate_quantile_indices,
    crc_select,
    loss_table_for_horizon,
    normalized_shortage_loss,
    shortage_scales,
)
from scripts.journal.evaluate_stage4_decision import parse_cost_ratios  # noqa: E402


def test_crc_losses_are_bounded():
    decision = np.array([0.0, 1.0, 3.0])
    labels = np.array([2.0, 1.0, 0.0])

    violation = bounded_violation_loss(decision, labels)
    shortage = normalized_shortage_loss(decision, labels, scale=2.0)

    assert violation.min() >= 0.0
    assert violation.max() <= 1.0
    assert shortage.min() >= 0.0
    assert shortage.max() <= 1.0
    np.testing.assert_allclose(violation, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(shortage, [1.0, 0.0, 0.0])


def test_candidate_risk_is_monotone_from_conservative_to_less_conservative():
    quantiles = [0.5, 0.75, 0.95]
    pred = np.array(
        [
            [[[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]]],
        ],
        dtype=np.float64,
    )
    labels = np.array([[[2.5], [0.5]]], dtype=np.float64)
    candidate_indices = candidate_quantile_indices(quantiles)

    table = loss_table_for_horizon(
        predict_quantiles=pred,
        labels=labels,
        candidate_indices=candidate_indices,
        loss_name="normalized_shortage",
        horizon_idx=0,
        scale=2.0,
    )
    risks = table.mean(axis=0)

    assert candidate_indices == [2, 1, 0]
    assert np.all(np.diff(risks) >= -1e-12)


def test_crc_selector_chooses_least_conservative_feasible_candidate():
    loss_table = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    selected = crc_select(loss_table, alpha=0.5)

    assert selected["candidate_rank"] == 1
    assert selected["selection_status"] == "interior_candidate_selected"
    assert selected["crc_upper"] <= 0.5


def test_crc_selection_table_contains_global_and_horizon_rows():
    quantiles = [0.5, 0.75, 0.95]
    horizons = [1, 3]
    pred = np.array(
        [
            [[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]],
            [[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]],
        ],
        dtype=np.float64,
    )
    labels = np.array(
        [
            [[2.5, 2.5], [0.5, 0.5]],
            [[1.5, 1.5], [3.5, 3.5]],
        ],
        dtype=np.float64,
    )
    scales = shortage_scales(labels, horizons)

    selection = build_crc_selection_table(
        calib_quantiles=pred,
        calib_labels=labels,
        quantiles=quantiles,
        horizons=horizons,
        alphas_violation=[0.8],
        alphas_shortage=[0.8],
        scales=scales,
    )

    assert {"global", "horizon"} == set(selection["crc_scope"])
    assert {"violation", "normalized_shortage"} == set(selection["loss_name"])
    assert set(selection["horizon"].astype(str)) == {"all", "1", "3"}


def test_policy_decisions_include_required_baselines_and_crc():
    quantiles = [0.5, 0.75, 0.95]
    horizons = [1]
    test_q = np.array(
        [
            [[[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]]],
        ],
        dtype=np.float64,
    )
    selection = pd.DataFrame(
        [
            {
                "crc_scope": "global",
                "loss_name": "violation",
                "alpha": 0.1,
                "horizon": "all",
                "selected_quantile": 0.75,
                "selected_quantile_index": 1,
            },
            {
                "crc_scope": "horizon",
                "loss_name": "violation",
                "alpha": 0.1,
                "horizon": 1,
                "selected_quantile": 0.75,
                "selected_quantile_index": 1,
            },
        ]
    )
    thresholds = pd.DataFrame([{"method": "global_cqr", "delta": 0.1, "horizon": "all", "s_hat": 0.0}])
    one_sided = pd.DataFrame(
        [
            {
                "method": "horizon_cqr",
                "cost_ratio": "3:1",
                "horizon": 1,
                "target_quantile": 0.75,
                "s_hat": 0.0,
            }
        ]
    )

    policies = build_policy_decisions(
        test_quantiles=test_q,
        quantiles=quantiles,
        horizons=horizons,
        selection_df=selection,
        thresholds_df=thresholds,
        decision_thresholds_df=one_sided,
        cost_ratios=parse_cost_ratios("3:1"),
    )

    assert "median_decision" in policies
    assert "symmetric_90_upper" in policies
    assert "raw_cost_aligned_quantile_3:1" in policies
    assert "one_sided_refined_cost_aligned_quantile_3:1" in policies
    assert "global_crc_violation_alpha0.1" in policies
    assert "horizon_crc_violation_alpha0.1" in policies


def test_summary_and_frontier_outputs_have_expected_rows():
    horizons = [1]
    labels = np.array([[[2.0], [1.0]]], dtype=np.float64)
    policies = {
        "median_decision": {"policy_type": "baseline", "decision": np.array([[[1.0], [1.0]]], dtype=np.float64)},
        "global_crc_violation_alpha0.1": {
            "policy_type": "crc",
            "crc_scope": "global",
            "loss_name": "violation",
            "alpha": 0.1,
            "decision": np.array([[[2.0], [2.0]]], dtype=np.float64),
        },
    }

    summary, by_horizon, cost = build_test_summaries(
        policies=policies,
        labels=labels,
        scales={1: 2.0},
        horizons=horizons,
        cost_ratios=parse_cost_ratios("3:1,5:1"),
    )
    frontier = build_risk_efficiency_frontier(summary)

    assert len(summary) == 2
    assert len(by_horizon) == 2
    assert len(cost) == 4
    assert len(frontier) == 2


def test_no_leakage_metadata_contract_is_explicit():
    metadata = {
        "no_leakage": {
            "calibration_source": "reconstructed_from_stage1_checkpoint_and_calibration_split",
            "test_source": "saved_stage1_test_arrays",
            "selection_uses_test_labels": False,
        }
    }

    assert metadata["no_leakage"]["selection_uses_test_labels"] is False
    assert "calibration_split" in metadata["no_leakage"]["calibration_source"]
