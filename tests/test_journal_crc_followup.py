import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.journal.build_crc_followup_experiments import (  # noqa: E402
    block_loss_table,
    build_empirical_vs_crc_selection,
    build_h1_alpha_sensitivity,
    build_safety_margin_selection,
    margin_candidates,
    raw_quantile_violation_loss_table,
    safety_margin_loss_table,
    select_largest_feasible,
    select_smallest_feasible,
)
from scripts.journal.build_crc_risk_control_experiments import candidate_quantile_indices  # noqa: E402
from scripts.journal.evaluate_stage4_decision import parse_cost_ratios  # noqa: E402


def toy_quantile_tensor():
    quantiles = [0.5, 0.75, 0.833333333333, 0.9, 0.95]
    horizons = [1, 3]
    base = np.array([1.0, 2.0, 2.5, 3.0, 4.0], dtype=np.float64)
    pred = np.tile(base.reshape(1, 1, 1, -1), (4, 3, 2, 1))
    labels = np.array(
        [
            [[1.0, 2.0], [2.0, 3.0], [5.0, 4.0]],
            [[1.0, 2.0], [2.0, 3.0], [5.0, 4.0]],
            [[1.0, 2.0], [2.0, 3.0], [5.0, 4.0]],
            [[1.0, 2.0], [2.0, 3.0], [5.0, 4.0]],
        ],
        dtype=np.float64,
    )
    return pred, labels, quantiles, horizons


def test_safety_margin_violation_monotonicity():
    q50 = np.array([1.0, 1.0, 1.0])
    labels = np.array([1.5, 2.0, 3.0])
    margins = np.array([0.0, 0.5, 2.0])

    table = safety_margin_loss_table(q50, labels, margins)
    risks = table.mean(axis=0)

    assert np.all(np.diff(risks) <= 1e-12)


def test_raw_quantile_violation_monotonicity():
    pred, labels, quantiles, _ = toy_quantile_tensor()
    candidates = candidate_quantile_indices(quantiles)

    table = raw_quantile_violation_loss_table(pred, labels, candidates, horizon_idx=0)
    risks = table.mean(axis=0)

    assert candidates == [4, 3, 2, 1, 0]
    assert np.all(np.diff(risks) >= -1e-12)


def test_crc_selector_picks_least_conservative_feasible_raw_candidate():
    loss_table = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    selected = select_largest_feasible(loss_table, alpha=0.6, rule="crc_corrected")

    assert selected["candidate_rank"] == 2


def test_empirical_feasible_set_contains_crc_feasible_set():
    loss_table = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    empirical = select_largest_feasible(loss_table, alpha=0.4, rule="empirical_risk")
    corrected = select_largest_feasible(loss_table, alpha=0.4, rule="crc_corrected")

    assert corrected["feasible_candidate_count"] <= empirical["feasible_candidate_count"]


def test_block_aggregation_shape():
    pred, labels, quantiles, _ = toy_quantile_tensor()
    candidates = candidate_quantile_indices(quantiles)

    table = block_loss_table(pred, labels, candidates, horizon_idx=0)

    assert table.shape == (4, 5)
    assert table.min() >= 0.0
    assert table.max() <= 1.0


def test_followup_selection_tables_are_calibration_only_shapes():
    pred, labels, quantiles, horizons = toy_quantile_tensor()

    safety = build_safety_margin_selection(pred, labels, quantiles, horizons, alphas=[0.5])
    raw = build_empirical_vs_crc_selection(pred, labels, quantiles, horizons, alphas=[0.5])

    assert set(safety["margin_scope"]) == {"global", "horizon"}
    assert set(raw["selector_scope"]) == {"global", "horizon"}
    assert safety["n_calibration_samples"].min() > 0
    assert raw["n_calibration_samples"].min() > 0


def test_h1_alpha_sensitivity_outputs_expected_rows():
    pred, labels, quantiles, _ = toy_quantile_tensor()

    out = build_h1_alpha_sensitivity(
        calib_quantiles=pred,
        calib_labels=labels,
        test_quantiles=pred,
        test_labels=labels,
        quantiles=quantiles,
        alphas=[0.3, 0.5],
        cost_ratios=parse_cost_ratios("3:1"),
    )

    assert len(out) == 2
    assert "h1_test_violation_rate" in out.columns
    assert "expected_cost_3_1" in out.columns


def test_margin_candidates_are_unique_and_sorted():
    margins, sources = margin_candidates(np.array([0.0, 0.0, 1.0, 2.0]))

    assert np.all(np.diff(margins) >= -1e-12)
    assert len(margins) == len(set(np.round(margins, 12)))
    assert len(margins) == len(sources)
