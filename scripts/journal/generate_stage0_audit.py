import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils.model_training.journal_contracts import build_horizon_config
DATA_DIR = REPO_ROOT / "data" / "datasets" / "ST_EVCDP_v2_canonical"
OUTPUT_DIR = REPO_ROOT / "docs" / "journal_stage0"


def compute_split_summary(time_index: pd.Index, split_rates: dict[str, float], seq_len: int, max_horizon: int) -> dict:
    total_steps = len(time_index)
    train_end = int(total_steps * split_rates["train"])
    valid_end = int(total_steps * (split_rates["train"] + split_rates["valid"]))
    calib_end = int(total_steps * (split_rates["train"] + split_rates["valid"] + split_rates["calib"]))

    split_bounds = {
        "train": (0, train_end),
        "valid": (train_end, valid_end),
        "calib": (valid_end, calib_end),
        "test": (calib_end, total_steps),
    }

    split_manifest = {}
    for split_name, (start, end) in split_bounds.items():
        split_len = max(end - start, 0)
        usable_windows = max(split_len - seq_len - max_horizon, 0)
        split_manifest[split_name] = {
            "start_index": int(start),
            "end_index_exclusive": int(end),
            "length": int(split_len),
            "start_time": str(time_index[start]) if split_len > 0 else None,
            "end_time": str(time_index[end - 1]) if split_len > 0 else None,
            "usable_windows_for_max_horizon": int(usable_windows),
        }

    return {
        "total_time_steps": int(total_steps),
        "seq_len": int(seq_len),
        "max_horizon": int(max_horizon),
        "splits": split_manifest,
    }


def build_station_manifest(inf_df: pd.DataFrame, adj_df: pd.DataFrame, occ_df: pd.DataFrame, duration_df: pd.DataFrame) -> pd.DataFrame:
    station_ids = occ_df.columns.astype(int)
    aligned_inf = inf_df.set_index("station_id").loc[station_ids].reset_index(names="station_id")
    adj_df = adj_df.copy()
    adj_df.index = adj_df.index.astype(int)
    adj_df.columns = adj_df.columns.astype(int)
    graph_degree = (adj_df.loc[station_ids, station_ids] > 0).sum(axis=1).to_numpy()

    manifest = aligned_inf.copy()
    manifest["graph_degree"] = graph_degree.astype(int)
    manifest["mean_occupancy_norm"] = occ_df.mean(axis=0).to_numpy()
    manifest["std_occupancy_norm"] = occ_df.std(axis=0).to_numpy()
    manifest["mean_duration_norm"] = duration_df.mean(axis=0).to_numpy()
    manifest["std_duration_norm"] = duration_df.std(axis=0).to_numpy()
    manifest["peak_ratio_duration"] = (
        duration_df.quantile(0.95, axis=0).to_numpy() / np.maximum(duration_df.mean(axis=0).to_numpy(), 1e-8)
    )
    return manifest.rename(columns={"station_id": "station_id"})


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    inf_df = pd.read_csv(DATA_DIR / "inf.csv")
    occ_df = pd.read_csv(DATA_DIR / "occupancy.csv", index_col=0)
    duration_df = pd.read_csv(DATA_DIR / "duration.csv", index_col=0)
    adj_df = pd.read_csv(DATA_DIR / "adjacency_matrix.csv", index_col=0)

    occ_df = occ_df.fillna(0.0)
    duration_df = duration_df.fillna(0.0)
    adj_df = adj_df.fillna(0.0)
    time_index = pd.to_datetime(occ_df.index)

    config = build_horizon_config()

    station_manifest = build_station_manifest(inf_df, adj_df, occ_df, duration_df)
    station_manifest.to_csv(OUTPUT_DIR / "station_manifest.csv", index=False)

    split_manifest = compute_split_summary(
        time_index=time_index,
        split_rates=config["split_rates"],
        seq_len=config["seq_len"],
        max_horizon=config["max_horizon"],
    )
    split_manifest["dataset"] = "ST_EVCDP_v2_canonical"
    split_manifest["station_count"] = int(occ_df.shape[1])
    split_manifest["time_start"] = str(time_index[0])
    split_manifest["time_end"] = str(time_index[-1])
    with open(OUTPUT_DIR / "split_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(split_manifest, fh, indent=2)

    station_ids = occ_df.columns.to_numpy()
    np.savez_compressed(
        OUTPUT_DIR / "adjacency_snapshot.npz",
        adjacency=adj_df.to_numpy(dtype=np.float32),
        station_ids=station_ids,
    )

    with open(OUTPUT_DIR / "horizon_config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    print(f"Stage 0 audit artifacts written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
