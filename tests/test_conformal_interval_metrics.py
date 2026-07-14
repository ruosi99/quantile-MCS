import numpy as np
import pytest

from utils.model_training.conformal import interval_metrics


def test_interval_metrics_wis_is_zero_for_perfect_point_interval() -> None:
    target = np.array([0.0, 2.0, 5.0])

    metrics = interval_metrics(target, target, target, point_pred=target, delta=0.1)

    assert metrics["PICP"] == 1.0
    assert metrics["MPIW"] == 0.0
    assert metrics["WIS"] == 0.0


def test_interval_metrics_wis_matches_single_interval_definition_inside_interval() -> None:
    target = np.array([5.0])
    lower = np.array([0.0])
    upper = np.array([10.0])
    median = np.array([5.0])

    metrics = interval_metrics(target, lower, upper, point_pred=median, delta=0.1)

    # [0.5 * |5-5| + 0.05 * IS_0.1] / 1.5, where IS_0.1 = 10.
    assert metrics["WIS"] == pytest.approx(1.0 / 3.0)


def test_interval_metrics_wis_penalizes_a_missed_interval() -> None:
    target = np.array([0.0])
    lower = np.array([2.0])
    upper = np.array([10.0])
    median = np.array([5.0])

    metrics = interval_metrics(target, lower, upper, point_pred=median, delta=0.1)

    # IS_0.1 = 8 + (2 / 0.1) * 2 = 48.
    expected = (0.5 * 5.0 + 0.05 * 48.0) / 1.5
    assert metrics["WIS"] == pytest.approx(expected)


def test_interval_metrics_rejects_invalid_delta_when_computing_wis() -> None:
    target = np.array([1.0])

    with pytest.raises(ValueError, match="delta must be in"):
        interval_metrics(target, target, target, point_pred=target, delta=0.0)
