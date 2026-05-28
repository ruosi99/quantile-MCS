import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.journal.pretrain_fixed_physics import (  # noqa: E402
    DEFAULT_PRETRAIN_QUANTILES,
    evaluate_model,
    physics_consistency_loss,
    row_normalize_adjacency,
    sample_physics_shift,
)
from utils.model_training.training_utils import CreateDataset  # noqa: E402


def test_row_normalize_adjacency_removes_self_loops_and_normalizes_rows():
    adj = torch.tensor(
        [
            [1.0, 2.0, 0.0],
            [3.0, 1.0, 3.0],
            [0.0, 0.0, 1.0],
        ]
    )

    norm = row_normalize_adjacency(adj)

    assert torch.allclose(torch.diag(norm), torch.zeros(3))
    assert torch.allclose(norm[0].sum(), torch.tensor(1.0))
    assert torch.allclose(norm[1].sum(), torch.tensor(1.0))
    assert torch.allclose(norm[2].sum(), torch.tensor(0.0))


def test_sample_physics_shift_has_expected_shape_and_bounds():
    torch.manual_seed(7)
    adj_norm = row_normalize_adjacency(torch.ones(4, 4))
    laws = torch.tensor([-1.48, -0.74])

    price_change, expected_shift, law = sample_physics_shift(
        batch_size=3,
        node_count=4,
        adj_norm=adj_norm,
        laws=laws,
        perturb_prop=1.0,
        perturb_scale=0.2,
        graph_layers=2,
        graph_decay=0.5,
        clamp=0.3,
        device=torch.device("cpu"),
    )

    assert price_change.shape == (3, 4)
    assert expected_shift.shape == (3, 4)
    assert law.shape == (3,)
    assert float(price_change.abs().max()) <= 0.3 + 1e-6
    assert float(expected_shift.abs().max()) <= 0.8 + 1e-6


class _TinyPriceSensitiveQuantileModel(torch.nn.Module):
    def __init__(self, quantile_count):
        super().__init__()
        self.quantile_count = quantile_count

    def forward(self, demand, price):
        base = demand.mean(dim=-1, keepdim=True) + 0.1 * price.mean(dim=-1, keepdim=True)
        increments = torch.ones(
            demand.shape[0],
            demand.shape[1],
            self.quantile_count - 1,
            device=demand.device,
        )
        return torch.cat([base, base + torch.cumsum(increments, dim=-1)], dim=-1)


def test_physics_consistency_loss_is_finite_for_quantile_model():
    torch.manual_seed(11)
    quantiles = DEFAULT_PRETRAIN_QUANTILES
    model = _TinyPriceSensitiveQuantileModel(quantile_count=len(quantiles))
    demand = torch.rand(2, 3, 4)
    price = torch.rand(2, 3, 4)
    base_pred = model(demand, price)
    adj_norm = row_normalize_adjacency(torch.ones(3, 3))
    laws = torch.tensor([-1.48, -0.74])

    loss = physics_consistency_loss(
        model=model,
        demand=demand,
        price=price,
        base_pred=base_pred,
        adj_norm=adj_norm,
        laws=laws,
        perturb_prop=1.0,
        perturb_scale=0.2,
        graph_layers=1,
        graph_decay=0.5,
        perturb_clamp=0.3,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_fixed_physics_evaluate_can_skip_prediction_arrays(tmp_path):
    occ = np.arange(30, dtype=np.float32).reshape(10, 3)
    prc = occ + 1
    dataset = CreateDataset(occ, prc, seq_l=2, pre_l=1, device=torch.device("cpu"))
    loader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False)
    quantiles = DEFAULT_PRETRAIN_QUANTILES

    report = evaluate_model(
        model=_TinyPriceSensitiveQuantileModel(quantile_count=len(quantiles)),
        test_loader=loader,
        device=torch.device("cpu"),
        cap=np.ones((1, 3), dtype=np.float32),
        quantiles=quantiles,
        output_dir=tmp_path,
        model_name="tiny_fixed_physics",
        max_test_batches=1,
        save_arrays=False,
    )

    assert report["arrays_saved"] is False
    assert report["array_files"] == {}
    assert (tmp_path / "point_metrics_q50.csv").exists()
    assert (tmp_path / "raw_interval_metrics.csv").exists()
    assert not (tmp_path / "predict_quantiles.npy").exists()
