import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import utils.model_training.models as models  # noqa: E402
from scripts.journal.train_multihorizon_raw import (  # noqa: E402
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

    occ = torch.rand(2, 3, 4, device=device)
    prc = torch.rand(2, 3, 4, device=device)
    with torch.no_grad():
        pred = model(occ, prc)

    assert pred.shape == (2, 3, 2, 3)
    assert torch.all(pred[..., 1:] >= pred[..., :-1])


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
