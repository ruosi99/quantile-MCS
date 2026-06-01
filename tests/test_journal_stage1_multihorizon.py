import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import utils.model_training.models as models  # noqa: E402
from scripts.journal.train_multihorizon_raw import (  # noqa: E402
    build_model,
    evaluate_model,
    load_matching_state_dict,
    point_metrics,
    quantile_crossing_metrics,
)
from utils.model_training.loss_functions import QuantileLoss  # noqa: E402
from utils.model_training.training_utils import (  # noqa: E402
    CreateMultiHorizonDataset,
    create_multi_horizon_rnn_data,
)


def test_create_multi_horizon_rnn_data_preserves_horizon_targets():
    data = np.arange(10, dtype=np.float32).reshape(10, 1)
    x, y = create_multi_horizon_rnn_data(data, lookback=3, horizons=[1, 3])

    assert x.shape == (4, 3, 1)
    assert y.shape == (4, 2, 1)
    assert y[0, :, 0].tolist() == [3.0, 5.0]
    assert y[-1, :, 0].tolist() == [6.0, 8.0]


def test_create_multi_horizon_dataset_returns_node_horizon_labels():
    occ = np.arange(24, dtype=np.float32).reshape(8, 3)
    prc = occ + 100
    dataset = CreateMultiHorizonDataset(occ, prc, seq_l=2, horizons=[1, 2, 4], device=torch.device("cpu"))

    demand, price, label = dataset[0]
    assert demand.shape == (3, 2)
    assert price.shape == (3, 2)
    assert label.shape == (3, 3)
    assert label[0].tolist() == [6.0, 9.0, 15.0]


def test_quantile_loss_accepts_multi_horizon_tensors():
    loss_fn = QuantileLoss([0.1, 0.5, 0.9])
    pred = torch.tensor(
        [[[[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]], [[0.5, 1.5, 2.5], [2.0, 3.0, 4.0]]]],
        dtype=torch.float32,
    )
    target = torch.tensor([[[1.0, 2.0], [1.5, 3.0]]], dtype=torch.float32)

    loss = loss_fn(pred, target)
    assert loss.ndim == 0
    assert float(loss) >= 0.0


def test_quantile_loss_can_weight_multi_horizon_tensors():
    loss_fn = QuantileLoss([0.5], horizon_weights=[2.0, 1.0])
    pred = torch.zeros(1, 1, 2, 1)
    target = torch.tensor([[[1.0, 3.0]]])

    loss = loss_fn(pred, target)

    assert torch.isclose(loss, torch.tensor(5.0 / 6.0))


def test_quantile_loss_rejects_mismatched_horizon_weights():
    loss_fn = QuantileLoss([0.5], horizon_weights=[1.0, 1.0, 1.0])
    pred = torch.zeros(1, 1, 2, 1)
    target = torch.zeros(1, 1, 2)

    with pytest.raises(ValueError, match="horizon weight mismatch"):
        loss_fn(pred, target)


def test_pag_informer_quantile_multi_horizon_output_shape_and_ordering():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adjacency = torch.eye(3, dtype=torch.float32, device=device).to_sparse()
    model = models.PAGInformerQuantile(
        a_sparse=adjacency,
        seq=4,
        hidden_dim=8,
        quantiles=[0.1, 0.5, 0.9],
        horizons=[1, 3],
    ).to(device)
    model.eval()

    named_parameters = dict(model.named_parameters())
    assert "gat_layer.head_weights.0" in named_parameters
    assert "gat_layer.head_attn.0" in named_parameters
    assert model.informer.encoder.layers[0].self_attn.batch_first
    assert model.informer.decoder.layers[0].self_attn.batch_first
    assert model.gat_layer.head_weights[0].requires_grad
    assert model.gat_layer.head_attn[0].requires_grad

    occ = torch.rand(2, 3, 4, device=device)
    prc = torch.rand(2, 3, 4, device=device)
    with torch.no_grad():
        pred = model(occ, prc)

    assert pred.shape == (2, 3, 2, 3)
    assert torch.all(pred[..., 1:] >= pred[..., :-1])


def test_pag_informer_quantile_can_freeze_gat_heads_and_use_legacy_transformer_layout():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adjacency = torch.eye(3, dtype=torch.float32, device=device).to_sparse()
    model = models.PAGInformerQuantile(
        a_sparse=adjacency,
        seq=4,
        hidden_dim=8,
        quantiles=[0.1, 0.5, 0.9],
        horizons=[1, 3],
        train_gat_heads=False,
        transformer_batch_first=False,
    ).to(device)
    model.eval()

    assert not model.gat_layer.head_weights[0].requires_grad
    assert not model.gat_layer.head_attn[0].requires_grad
    assert not model.informer.encoder.layers[0].self_attn.batch_first
    assert not model.informer.decoder.layers[0].self_attn.batch_first

    occ = torch.rand(2, 3, 4, device=device)
    prc = torch.rand(2, 3, 4, device=device)
    with torch.no_grad():
        pred = model(occ, prc)

    assert pred.shape == (2, 3, 2, 3)
    assert torch.all(pred[..., 1:] >= pred[..., :-1])


def test_temporal_graph_quantile_output_shape_ordering_and_batch_safety():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adjacency = torch.tensor(
        [
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.25],
            [0.0, 0.25, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    ).to_sparse()
    model = models.TemporalGraphQuantile(
        a_sparse=adjacency,
        seq=4,
        hidden_dim=8,
        quantiles=[0.1, 0.5, 0.9],
        horizons=[1, 3],
        graph_layers=1,
        dropout=0.0,
    ).to(device)
    model.eval()

    occ = torch.rand(2, 3, 4, device=device)
    prc = torch.rand(2, 3, 4, device=device)
    with torch.no_grad():
        pred_batch = model(occ, prc)
        pred_single = model(occ[:1], prc[:1])

    assert pred_batch.shape == (2, 3, 2, 3)
    assert torch.all(pred_batch[..., 1:] >= pred_batch[..., :-1])
    assert pred_batch[..., -1].max().item() < 0.25
    assert torch.allclose(pred_batch[:1], pred_single, atol=1e-6)


def test_stage1_build_model_can_select_temporal_graph_quantile():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adjacency = torch.eye(3, dtype=torch.float32, device=device).to_sparse()
    args = SimpleNamespace(
        architecture="temporal_graph_quantile",
        dropout=0.0,
        freeze_gat_heads=False,
        graph_layers=1,
        hidden_dim=16,
        load_method="",
        seq_len=4,
        temporal_layers=1,
        transformer_batch_first=True,
    )

    model, load_method = build_model(
        args=args,
        adj_sparse=adjacency,
        quantiles=[0.1, 0.5, 0.9],
        horizons=[1, 3],
        device=device,
    )

    assert isinstance(model, models.TemporalGraphQuantile)
    assert model.hidden_dim == 16
    assert "TemporalGraphQuantile" in load_method


def test_multi_scale_temporal_graph_quantile_output_shape_ordering_and_batch_safety():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adjacency = torch.tensor(
        [
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.25],
            [0.0, 0.25, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    ).to_sparse()
    model = models.MultiScaleTemporalGraphQuantile(
        a_sparse=adjacency,
        seq=6,
        short_seq=3,
        hidden_dim=8,
        quantiles=[0.1, 0.5, 0.9],
        horizons=[1, 3],
        graph_layers=1,
        dropout=0.0,
    ).to(device)
    model.eval()

    occ = torch.rand(2, 3, 6, device=device)
    prc = torch.rand(2, 3, 6, device=device)
    with torch.no_grad():
        pred_batch = model(occ, prc)
        pred_single = model(occ[:1], prc[:1])

    assert pred_batch.shape == (2, 3, 2, 3)
    assert torch.all(pred_batch[..., 1:] >= pred_batch[..., :-1])
    assert pred_batch[..., -1].max().item() < 0.25
    assert torch.allclose(pred_batch[:1], pred_single, atol=1e-6)


def test_stage1_build_model_can_select_multi_scale_temporal_graph_quantile():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adjacency = torch.eye(3, dtype=torch.float32, device=device).to_sparse()
    args = SimpleNamespace(
        architecture="multi_scale_temporal_graph_quantile",
        dropout=0.0,
        freeze_gat_heads=False,
        graph_layers=1,
        hidden_dim=16,
        load_method="",
        seq_len=6,
        short_seq=3,
        temporal_layers=1,
        transformer_batch_first=True,
    )

    model, load_method = build_model(
        args=args,
        adj_sparse=adjacency,
        quantiles=[0.1, 0.5, 0.9],
        horizons=[1, 3],
        device=device,
    )

    assert isinstance(model, models.MultiScaleTemporalGraphQuantile)
    assert model.hidden_dim == 16
    assert model.short_seq == 3
    assert "MultiScaleTemporalGraphQuantile" in load_method


def test_stage1_metric_helpers_are_shape_agnostic():
    labels = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    pred = labels.copy()
    metrics = point_metrics(pred, labels)
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0

    pred_q = np.stack([labels, labels + 1.0, labels + 2.0], axis=-1)
    crossing = quantile_crossing_metrics(pred_q)
    assert crossing["crossing_rate"] == 0.0


def test_load_matching_state_dict_skips_mismatched_output_layer():
    source = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))
    target = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Linear(3, 5))

    with torch.no_grad():
        source[0].weight.fill_(1.25)
        source[0].bias.fill_(0.5)
        target[0].weight.zero_()
        target[0].bias.zero_()

    report = load_matching_state_dict(target, source)

    assert report["loaded_key_count"] == 2
    assert report["skipped_key_count"] == 2
    assert torch.allclose(target[0].weight, source[0].weight)
    assert torch.allclose(target[0].bias, source[0].bias)
    skipped = {item["key"]: item["reason"] for item in report["skipped_keys"]}
    assert skipped["2.weight"] == "shape_mismatch"
    assert skipped["2.bias"] == "shape_mismatch"


class _TinyQuantileModel(torch.nn.Module):
    def __init__(self, quantile_count):
        super().__init__()
        self.quantile_count = quantile_count

    def forward(self, demand, price):
        b, n, _ = demand.shape
        base = torch.ones(b, n, 2, 1, device=demand.device)
        increments = torch.ones(b, n, 2, self.quantile_count - 1, device=demand.device)
        return torch.cat([base, base + torch.cumsum(increments, dim=-1)], dim=-1)


def test_evaluate_model_can_skip_large_prediction_arrays(tmp_path):
    occ = np.arange(30, dtype=np.float32).reshape(10, 3)
    prc = occ + 1
    dataset = CreateMultiHorizonDataset(occ, prc, seq_l=2, horizons=[1, 2], device=torch.device("cpu"))
    loader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False)
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.833333333333, 0.9, 0.95]

    report = evaluate_model(
        model=_TinyQuantileModel(quantile_count=len(quantiles)),
        test_loader=loader,
        device=torch.device("cpu"),
        cap=np.ones((1, 3), dtype=np.float32),
        quantiles=quantiles,
        horizons=[1, 2],
        output_dir=tmp_path,
        model_name="tiny",
        max_test_batches=1,
        save_arrays=False,
    )

    assert report["arrays_saved"] is False
    assert report["array_files"] == {}
    assert (tmp_path / "point_metrics_by_horizon.csv").exists()
    assert (tmp_path / "raw_interval_metrics_by_horizon.csv").exists()
    assert not (tmp_path / "predict_quantiles.npy").exists()
    assert not (tmp_path / "label_list.npy").exists()
    assert not (tmp_path / "predict_point_q50.npy").exists()
