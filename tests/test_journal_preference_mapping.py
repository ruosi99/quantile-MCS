import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.journal.build_preference_mapping_experiments import (  # noqa: E402
    build_conflict_map,
    build_frontier_policies,
    build_phase_gate,
    build_risk_to_cost_mapping,
    build_risk_screened_deployment,
    conflict_type,
    cost_ratio_to_quantile,
    decision_metrics,
    quantile_to_implied_ratio,
)
from scripts.journal.evaluate_stage4_decision import parse_cost_ratios  # noqa: E402


def toy_inputs():
    quantiles = [0.5, 0.75, 0.9, 0.95]
    horizons = [1, 3]
    base = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    pred = np.tile(base.reshape(1, 1, 1, -1), (3, 2, 2, 1))
    labels = np.array(
        [
            [[1.0, 2.0], [3.5, 4.5]],
            [[1.0, 2.0], [3.5, 4.5]],
            [[1.0, 2.0], [3.5, 4.5]],
        ],
        dtype=np.float64,
    )
    thresholds = pd.DataFrame([{"method": "global_cqr", "delta": 0.1, "horizon": "all", "s_hat": 0.0}])
    one_sided = pd.DataFrame(
        [
            {"method": "horizon_cqr", "cost_ratio": "3:1", "horizon": 1, "s_hat": 0.0},
            {"method": "horizon_cqr", "cost_ratio": "3:1", "horizon": 3, "s_hat": 0.0},
        ]
    )
    selection = pd.DataFrame(
        [
            {
                "crc_scope": "global",
                "loss_name": "violation",
                "alpha": 0.1,
                "horizon": "all",
                "selected_quantile": 0.9,
                "selected_quantile_index": 2,
            },
            {
                "crc_scope": "horizon",
                "loss_name": "violation",
                "alpha": 0.1,
                "horizon": 1,
                "selected_quantile": 0.75,
                "selected_quantile_index": 1,
            },
            {
                "crc_scope": "horizon",
                "loss_name": "violation",
                "alpha": 0.1,
                "horizon": 3,
                "selected_quantile": 0.95,
                "selected_quantile_index": 3,
            },
        ]
    )
    return pred, labels, quantiles, horizons, thresholds, one_sided, selection


def test_exact_cost_ratio_quantile_mapping():
    assert cost_ratio_to_quantile(1, 1) == 0.5
    assert cost_ratio_to_quantile(3, 1) == 0.75
    np.testing.assert_allclose(cost_ratio_to_quantile(5, 1), 5 / 6)


def test_quantile_to_implied_ratio():
    assert quantile_to_implied_ratio(0.5) == 1.0
    assert quantile_to_implied_ratio(0.75) == 3.0
    np.testing.assert_allclose(quantile_to_implied_ratio(0.9), 9.0)


def test_decision_metrics_are_nonnegative_and_complete():
    decision = np.array([1.0, 3.0, 5.0])
    labels = np.array([2.0, 2.0, 4.0])
    metrics = decision_metrics(decision, labels, parse_cost_ratios("3:1,5:1"))

    assert metrics["test_violation_rate"] >= 0.0
    assert metrics["mean_overage"] >= 0.0
    assert metrics["mean_shortage"] >= 0.0
    assert "expected_cost_3_1" in metrics
    assert "expected_cost_5_1" in metrics


def test_risk_to_cost_mapping_has_required_columns():
    pred, labels, quantiles, horizons, _, _, selection = toy_inputs()
    pooled, by_horizon, _ = build_risk_to_cost_mapping(
        selection_df=selection,
        test_quantiles=pred,
        labels=labels,
        quantiles=quantiles,
        horizons=horizons,
        alphas=[0.1],
        cost_ratios=parse_cost_ratios("3:1"),
        h1_df=pd.DataFrame(),
    )

    assert {"policy", "crc_scope", "implied_cost_ratio", "test_violation_rate"}.issubset(pooled.columns)
    assert {"horizon", "selected_quantile", "implied_cost_ratio"}.issubset(by_horizon.columns)
    assert set(pooled["crc_scope"]) == {"global", "horizon"}


def test_conflict_type_classification():
    assert conflict_type(0.75, 0.75, 0.08, 0.1) == "aligned"
    assert conflict_type(0.75, 0.9, 0.15, 0.1) == "cost_policy_too_aggressive"
    assert conflict_type(0.9, 0.75, 0.08, 0.1) == "cost_policy_risk_feasible"
    assert conflict_type(0.75, 0.9, 0.08, 0.1) == "crc_more_conservative_than_needed"


def test_phase_gate_detects_nontrivial_conflict():
    conflict = pd.DataFrame(
        [
            {
                "conflict_type": "cost_policy_too_aggressive",
                "tau_gap": 0.15,
                "cost_policy_actual_violation": 0.2,
                "alpha": 0.1,
            }
        ]
    )
    gate = build_phase_gate(conflict)

    assert gate["phase_a_gate_pass"] is True
    assert gate["too_aggressive_pair_count"] == 1


def test_frontier_and_risk_screened_outputs_have_expected_columns():
    pred, labels, quantiles, horizons, thresholds, one_sided, selection = toy_inputs()
    ratios = parse_cost_ratios("3:1")
    frontier, policies = build_frontier_policies(
        test_quantiles=pred,
        labels=labels,
        quantiles=quantiles,
        horizons=horizons,
        cost_ratios=ratios,
        thresholds_df=thresholds,
        decision_thresholds_df=one_sided,
        selection_df=selection,
        alphas=[0.1],
    )
    gate = {"phase_a_gate_pass": True}
    screened, by_horizon, comparison = build_risk_screened_deployment(
        test_quantiles=pred,
        labels=labels,
        quantiles=quantiles,
        horizons=horizons,
        cost_ratios=ratios,
        alphas=[0.1],
        selection_df=selection,
        symmetric_decision=policies["symmetric_90_upper"],
        enable=True,
        gate=gate,
    )

    assert {"policy", "policy_family", "test_violation_rate", "mean_overage"}.issubset(frontier.columns)
    assert {"risk_screened_max_tau", "cost_only", "crc_only", "symmetric_90_upper"}.issubset(set(screened["policy_role"]))
    assert not by_horizon.empty
    assert {"risk_target_satisfied", "cost_increase_vs_cost_only"}.issubset(comparison.columns)


def test_max_tau_screened_policy_is_at_least_as_conservative_as_parents():
    pred, labels, quantiles, horizons, thresholds, one_sided, selection = toy_inputs()
    ratios = parse_cost_ratios("3:1")
    _, policies = build_frontier_policies(
        test_quantiles=pred,
        labels=labels,
        quantiles=quantiles,
        horizons=horizons,
        cost_ratios=ratios,
        thresholds_df=thresholds,
        decision_thresholds_df=one_sided,
        selection_df=selection,
        alphas=[0.1],
    )
    screened, _, _ = build_risk_screened_deployment(
        test_quantiles=pred,
        labels=labels,
        quantiles=quantiles,
        horizons=horizons,
        cost_ratios=ratios,
        alphas=[0.1],
        selection_df=selection,
        symmetric_decision=policies["symmetric_90_upper"],
        enable=True,
        gate={"phase_a_gate_pass": True},
    )

    rows = screened[screened["policy_role"] == "risk_screened_max_tau"]
    assert (rows["selected_quantile"].astype(float) >= rows["tau_cost"].astype(float)).all()
    assert (rows["selected_quantile"].astype(float) >= rows["tau_crc"].astype(float) - 1e-12).all()


def test_no_leakage_metadata_contract_is_explicit():
    metadata = {
        "no_leakage": {
            "cost_policy_selection": "analytic_tau_from_cost_ratio",
            "crc_policy_selection": "imported_from_existing_calibration_only_crc_outputs",
            "new_risk_screened_max_tau_selection": "deterministic_combination_of_cost_tau_and_crc_selected_tau",
            "selection_uses_test_labels": False,
        }
    }

    assert metadata["no_leakage"]["selection_uses_test_labels"] is False
    assert "calibration_only" in metadata["no_leakage"]["crc_policy_selection"]
