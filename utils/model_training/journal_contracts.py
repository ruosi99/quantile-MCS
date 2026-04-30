from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - environment-dependent fallback
    torch = None


DEFAULT_JOURNAL_HORIZONS = [1, 3, 6, 12, 24]
DEFAULT_JOURNAL_QUANTILES = [
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    5.0 / 6.0,
    0.9,
    0.95,
]


def build_horizon_config(
    seq_len: int = 24,
    horizons: Sequence[int] | None = None,
    quantiles: Sequence[float] | None = None,
    split_rates: dict[str, float] | None = None,
) -> dict:
    if horizons is None:
        horizons = DEFAULT_JOURNAL_HORIZONS
    if quantiles is None:
        quantiles = DEFAULT_JOURNAL_QUANTILES
    if split_rates is None:
        split_rates = {
            "train": 0.7,
            "valid": 0.1,
            "calib": 0.1,
            "test": 0.1,
        }

    return {
        "seq_len": int(seq_len),
        "horizons": [int(h) for h in horizons],
        "max_horizon": int(max(horizons)),
        "quantiles": [float(q) for q in quantiles],
        "split_rates": split_rates,
        "tensor_contract": {
            "predict_quantiles": ["B", "N", "H", "Q"],
            "targets": ["B", "N", "H"],
        },
    }


def _shape_of(values) -> tuple[int, ...]:
    return tuple(int(x) for x in values.shape)


def _to_numpy(values) -> np.ndarray:
    if torch is not None and isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def assert_quantile_tensor_shape(
    pred,
    n_horizons: int | None = None,
    n_quantiles: int | None = None,
    context: str = "predict_quantiles",
) -> None:
    shape = _shape_of(pred)
    if len(shape) != 4:
        raise AssertionError(
            f"{context} must have shape (B, N, H, Q), got ndim={len(shape)} and shape={shape}"
        )
    if n_horizons is not None and shape[2] != n_horizons:
        raise AssertionError(
            f"{context} horizon axis mismatch: expected H={n_horizons}, got shape={shape}"
        )
    if n_quantiles is not None and shape[3] != n_quantiles:
        raise AssertionError(
            f"{context} quantile axis mismatch: expected Q={n_quantiles}, got shape={shape}"
        )


def assert_target_tensor_shape(
    target,
    n_horizons: int | None = None,
    context: str = "targets",
) -> None:
    shape = _shape_of(target)
    if len(shape) != 3:
        raise AssertionError(
            f"{context} must have shape (B, N, H), got ndim={len(shape)} and shape={shape}"
        )
    if n_horizons is not None and shape[2] != n_horizons:
        raise AssertionError(
            f"{context} horizon axis mismatch: expected H={n_horizons}, got shape={shape}"
        )


def assert_monotonic_quantiles(
    pred,
    atol: float = 1e-8,
    context: str = "predict_quantiles",
) -> None:
    assert_quantile_tensor_shape(pred, context=context)
    pred_np = _to_numpy(pred)
    deltas = pred_np[..., 1:] - pred_np[..., :-1]
    if np.any(deltas < -atol):
        min_delta = float(deltas.min())
        raise AssertionError(
            f"{context} violates monotonic quantile ordering: minimum adjacent delta={min_delta}"
        )


def horizon_means(values) -> np.ndarray:
    shape = _shape_of(values)
    if len(shape) != 3:
        raise AssertionError(
            f"horizon_means expects a tensor with shape (B, N, H), got shape={shape}"
        )
    values_np = _to_numpy(values)
    return values_np.mean(axis=(0, 1))


def build_horizon_sentinel(batch: int, nodes: int, horizons: Iterable[int]) -> np.ndarray:
    horizon_list = list(horizons)
    base = np.array(horizon_list, dtype=np.float32).reshape(1, 1, -1)
    return np.repeat(np.repeat(base, batch, axis=0), nodes, axis=1)
