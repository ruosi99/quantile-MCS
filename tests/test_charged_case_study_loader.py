import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.model_training.journal_data import (  # noqa: E402
    load_journal_dataset,
    load_journal_dataset_from_metadata,
)


def test_charged_site_loader_aligns_ids_and_uses_volume_target(tmp_path):
    data_dir = tmp_path / "LOA_remove_zero"
    data_dir.mkdir()
    index = pd.date_range("2023-04-01", periods=4, freq="h")
    target_columns = ["0", "10", "2"]

    pd.DataFrame(
        [[1.0, 5.0, 2.0], [0.0, 10.0, 4.0], [2.0, 0.0, 6.0], [3.0, 15.0, 0.0]],
        index=index,
        columns=target_columns,
    ).to_csv(data_dir / "volume.csv")
    pd.DataFrame(
        [[0.5, 2.5, 1.0], [0.0, 5.0, 2.0], [1.0, 0.0, 3.0], [1.5, 7.5, 0.0]],
        index=index,
        columns=target_columns,
    ).to_csv(data_dir / "duration.csv")
    pd.DataFrame(0.1, index=index, columns=target_columns).to_csv(data_dir / "e_price.csv")
    pd.DataFrame(0.2, index=index, columns=target_columns).to_csv(data_dir / "s_price.csv")
    pd.DataFrame(
        {
            "site": [0, 2, 10],
            "longitude": [1.0, 2.0, 3.0],
            "latitude": [4.0, 5.0, 6.0],
            "charger_num": [1, 2, 5],
            "total_duration": [10.0, 20.0, 30.0],
        }
    ).to_csv(data_dir / "sites.csv", index=False)
    pd.DataFrame(
        [[0.0, 4.0, 1.0], [4.0, 0.0, 2.0], [1.0, 2.0, 0.0]],
        index=["0", "2", "10"],
        columns=["0", "2", "10"],
    ).to_csv(data_dir / "distance.csv")

    bundle = load_journal_dataset(
        data_dir=data_dir,
        dataset_family="charged_site",
        target_feature="volume",
        charged_adjacency_k=1,
    )

    assert bundle.node_ids == target_columns
    assert bundle.target_feature == "volume"
    assert bundle.metadata["dataset_family"] == "charged_site"
    assert bundle.metadata["city"] == "LOA"
    assert bundle.metadata["variant"] == "remove_zero"
    np.testing.assert_allclose(bundle.cap, np.array([[1.0, 5.0, 2.0]]))
    np.testing.assert_allclose(
        bundle.target_series[0],
        np.array([1.0, 1.0, 1.0]),
    )
    np.testing.assert_allclose(bundle.price[0], np.array([0.3, 0.3, 0.3]))

    adj = bundle.adjacency
    assert isinstance(adj, torch.Tensor)
    assert adj.shape == (3, 3)
    assert torch.allclose(torch.diag(adj), torch.ones(3))
    assert torch.allclose(adj, adj.T)
    assert torch.isfinite(adj).all()


def test_loader_can_rehydrate_charged_dataset_from_stage1_metadata(tmp_path):
    data_dir = tmp_path / "MEL_remove_zero"
    data_dir.mkdir()
    index = pd.date_range("2023-04-01", periods=3, freq="h")
    columns = ["0", "1"]
    for name in ["volume", "duration"]:
        pd.DataFrame([[2.0, 4.0], [0.0, 8.0], [6.0, 0.0]], index=index, columns=columns).to_csv(
            data_dir / f"{name}.csv"
        )
    pd.DataFrame(0.0, index=index, columns=columns).to_csv(data_dir / "e_price.csv")
    pd.DataFrame(0.0, index=index, columns=columns).to_csv(data_dir / "s_price.csv")
    pd.DataFrame(
        {"site_id": [0, 1], "longitude": [1.0, 2.0], "latitude": [3.0, 4.0], "charger_num": [2, 4]}
    ).to_csv(data_dir / "sites.csv", index=False)
    pd.DataFrame([[0.0, 1.0], [1.0, 0.0]], index=columns, columns=columns).to_csv(data_dir / "distance.csv")
    metadata = {
        "data_dir": str(data_dir),
        "dataset_family": "charged_site",
        "target_feature": "volume",
        "dataset_metadata": {"charged_adjacency_k": 1},
        "model_name": "journal_charged_volume_graph_patchtst",
    }

    bundle = load_journal_dataset_from_metadata(metadata, base_dir=PROJECT_ROOT)

    np.testing.assert_allclose(bundle.cap, np.array([[2.0, 4.0]]))
    np.testing.assert_allclose(bundle.target_series[0], np.array([1.0, 1.0]))
    assert bundle.metadata["site_id_column"] == "site_id"


def test_urbanev_station_loader_keeps_default_duration_target(tmp_path):
    data_dir = tmp_path / "urbanev"
    data_dir.mkdir()
    index = pd.date_range("2022-09-01", periods=3, freq="h")
    columns = ["1001", "1002"]
    pd.DataFrame({"station_id": columns, "longitude": [1.0, 2.0], "latitude": [3.0, 4.0], "pile_count": [2, 4]}).to_csv(
        data_dir / "inf.csv",
        index=False,
    )
    pd.DataFrame([[2.0, 8.0], [0.0, 4.0], [4.0, 0.0]], index=index, columns=columns).to_csv(
        data_dir / "occupancy.csv"
    )
    pd.DataFrame([[1.0, 4.0], [0.0, 8.0], [2.0, 0.0]], index=index, columns=columns).to_csv(
        data_dir / "duration.csv"
    )
    pd.DataFrame(0.1, index=index, columns=columns).to_csv(data_dir / "e_price.csv")
    pd.DataFrame(0.2, index=index, columns=columns).to_csv(data_dir / "s_price.csv")
    pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=columns, columns=columns).to_csv(
        data_dir / "adjacency_matrix.csv"
    )

    bundle = load_journal_dataset(data_dir=data_dir)

    assert bundle.metadata["dataset_family"] == "urbanev_station"
    assert bundle.target_feature == "duration"
    np.testing.assert_allclose(bundle.cap, np.array([[2.0, 4.0]]))
    np.testing.assert_allclose(bundle.target_series[0], np.array([0.5, 1.0]))
    np.testing.assert_allclose(bundle.price[0], np.array([0.3, 0.3]))
