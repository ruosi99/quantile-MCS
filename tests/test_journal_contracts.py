import sys
from pathlib import Path

import pytest
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.model_training.journal_contracts import (  # noqa: E402
    DEFAULT_JOURNAL_HORIZONS,
    DEFAULT_JOURNAL_QUANTILES,
    assert_monotonic_quantiles,
    assert_quantile_tensor_shape,
    assert_target_tensor_shape,
    build_horizon_sentinel,
    horizon_means,
)


def test_quantile_tensor_shape_accepts_valid_tensor():
    pred = np.zeros((2, 5, len(DEFAULT_JOURNAL_HORIZONS), len(DEFAULT_JOURNAL_QUANTILES)))
    assert_quantile_tensor_shape(
        pred,
        n_horizons=len(DEFAULT_JOURNAL_HORIZONS),
        n_quantiles=len(DEFAULT_JOURNAL_QUANTILES),
    )


def test_quantile_tensor_shape_rejects_legacy_rank():
    pred = np.zeros((2, 5, len(DEFAULT_JOURNAL_QUANTILES)))
    with pytest.raises(AssertionError):
        assert_quantile_tensor_shape(pred)


def test_target_tensor_shape_rejects_wrong_horizon_axis():
    target = np.zeros((2, 5, 3))
    with pytest.raises(AssertionError):
        assert_target_tensor_shape(target, n_horizons=len(DEFAULT_JOURNAL_HORIZONS))


def test_monotonic_quantiles_reject_crossing():
    pred = np.array([[[[0.1, 0.2, 0.15], [0.2, 0.3, 0.4]]]], dtype=np.float32)
    with pytest.raises(AssertionError):
        assert_monotonic_quantiles(pred)


def test_horizon_sentinel_preserves_horizon_identity():
    sentinel = build_horizon_sentinel(batch=2, nodes=3, horizons=DEFAULT_JOURNAL_HORIZONS)
    means = horizon_means(sentinel)
    expected = np.array(DEFAULT_JOURNAL_HORIZONS, dtype=np.float32)
    assert np.allclose(means, expected)
