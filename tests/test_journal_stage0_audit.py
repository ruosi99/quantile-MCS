import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.journal.generate_stage0_audit import (  # noqa: E402
    DATA_DIR,
    build_station_manifest,
    compute_split_summary,
)
from utils.model_training.journal_contracts import build_horizon_config  # noqa: E402


def test_stage0_audit_inputs_match_current_dataset():
    occ_df = pd.read_csv(DATA_DIR / "occupancy.csv", index_col=0).fillna(0.0)
    inf_df = pd.read_csv(DATA_DIR / "inf.csv")
    adj_df = pd.read_csv(DATA_DIR / "adjacency_matrix.csv", index_col=0).fillna(0.0)
    duration_df = pd.read_csv(DATA_DIR / "duration.csv", index_col=0).fillna(0.0)

    manifest = build_station_manifest(inf_df, adj_df, occ_df, duration_df)

    assert occ_df.shape[1] == 1682
    assert adj_df.shape == (1682, 1682)
    assert len(manifest) == 1682
    assert manifest["station_id"].iloc[0] == 1001


def test_split_manifest_summary_is_consistent():
    occ_df = pd.read_csv(DATA_DIR / "occupancy.csv", index_col=0).fillna(0.0)
    time_index = pd.to_datetime(occ_df.index)
    config = build_horizon_config()

    split_manifest = compute_split_summary(
        time_index=time_index,
        split_rates=config["split_rates"],
        seq_len=config["seq_len"],
        max_horizon=config["max_horizon"],
    )

    total = sum(item["length"] for item in split_manifest["splits"].values())
    assert total == len(time_index)
    assert split_manifest["max_horizon"] == 24
    assert split_manifest["splits"]["test"]["usable_windows_for_max_horizon"] >= 0


def test_horizon_config_serializable():
    config = build_horizon_config()
    serialized = json.dumps(config)
    recovered = json.loads(serialized)
    assert recovered["horizons"] == [1, 3, 6, 12, 24]
    assert np.isclose(recovered["quantiles"][-3], 5.0 / 6.0)
