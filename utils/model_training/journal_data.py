from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


SUPPORTED_DATASET_FAMILIES = ("urbanev_station", "charged_site")
SUPPORTED_TARGET_FEATURES = ("duration", "volume", "occupancy")


@dataclass(frozen=True)
class JournalDatasetBundle:
    occupancy: np.ndarray | None
    duration: np.ndarray | None
    volume: np.ndarray | None
    price: np.ndarray
    adjacency: torch.Tensor
    cap: np.ndarray
    target_series: np.ndarray
    target_feature: str
    node_ids: list[str]
    metadata: dict[str, Any]


def _resolve_data_dir(data_dir: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(data_dir)
    if path.is_absolute() or base_dir is None:
        return path
    return base_dir / path


def _read_time_site_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file not found: {path}")
    df = pd.read_csv(path, index_col=0, header=0)
    try:
        df.index = pd.to_datetime(df.index)
    except (AssertionError, TypeError, ValueError):
        pass
    df.columns = df.columns.astype(str)
    return df.fillna(0.0)


def _read_square_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file not found: {path}")
    df = pd.read_csv(path, index_col=0, header=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df.fillna(0.0)


def _safe_cap(values: pd.Series | np.ndarray) -> np.ndarray:
    cap = np.asarray(values, dtype=float).reshape(1, -1)
    return np.where(np.isfinite(cap) & (cap > 0.0), cap, 1.0)


def _normalise_by_cap(df: pd.DataFrame, cap: np.ndarray) -> np.ndarray:
    return np.asarray(df, dtype=float) / cap


def _select_target(
    *,
    target_feature: str,
    occupancy: np.ndarray | None,
    duration: np.ndarray | None,
    volume: np.ndarray | None,
) -> np.ndarray:
    if target_feature == "occupancy" and occupancy is not None:
        return occupancy
    if target_feature == "duration" and duration is not None:
        return duration
    if target_feature == "volume" and volume is not None:
        return volume
    available = [
        name
        for name, value in (("occupancy", occupancy), ("duration", duration), ("volume", volume))
        if value is not None
    ]
    raise ValueError(f"target_feature={target_feature!r} is unavailable; available={available}")


def _time_bounds(df: pd.DataFrame) -> tuple[str, str]:
    if len(df.index) == 0:
        return "", ""
    return str(df.index[0]), str(df.index[-1])


def _load_urbanev_station(
    data_dir: Path,
    target_feature: str,
) -> JournalDatasetBundle:
    if target_feature not in {"occupancy", "duration"}:
        raise ValueError("UrbanEV station data supports target_feature='duration' or 'occupancy'.")

    inf = pd.read_csv(data_dir / "inf.csv", header=0)
    occ_df = _read_time_site_csv(data_dir / "occupancy.csv")
    duration_df = _read_time_site_csv(data_dir / "duration.csv").reindex(
        index=occ_df.index,
        columns=occ_df.columns,
    ).fillna(0.0)
    e_price_df = _read_time_site_csv(data_dir / "e_price.csv").reindex(
        index=occ_df.index,
        columns=occ_df.columns,
    ).fillna(0.0)
    s_price_df = _read_time_site_csv(data_dir / "s_price.csv").reindex(
        index=occ_df.index,
        columns=occ_df.columns,
    ).fillna(0.0)
    adj_df = _read_square_csv(data_dir / "adjacency_matrix.csv").reindex(
        index=occ_df.columns,
        columns=occ_df.columns,
    ).fillna(0.0)

    inf["station_id"] = inf["station_id"].astype(str)
    inf = inf.set_index("station_id").reindex(occ_df.columns)
    if inf["pile_count"].isna().any():
        missing = inf.index[inf["pile_count"].isna()].tolist()[:10]
        raise ValueError(f"UrbanEV inf.csv is missing pile_count rows for stations: {missing}")

    cap = _safe_cap(inf["pile_count"])
    occupancy = _normalise_by_cap(occ_df, cap)
    duration = _normalise_by_cap(duration_df, cap)
    price = np.asarray(e_price_df + s_price_df, dtype=float)
    adjacency = torch.tensor(np.asarray(adj_df, dtype=float), dtype=torch.float32)
    target_series = _select_target(
        target_feature=target_feature,
        occupancy=occupancy,
        duration=duration,
        volume=None,
    )
    start, end = _time_bounds(occ_df)

    metadata: dict[str, Any] = {
        "dataset_family": "urbanev_station",
        "data_dir": str(data_dir),
        "target_feature": target_feature,
        "node_level": "station",
        "node_count": int(len(occ_df.columns)),
        "node_id_sample": occ_df.columns[:10].tolist(),
        "time_steps": int(len(occ_df)),
        "time_start": start,
        "time_end": end,
        "target_normalized_by": "pile_count",
        "price_source": "e_price.csv + s_price.csv",
        "adjacency_source": "adjacency_matrix.csv",
    }
    return JournalDatasetBundle(
        occupancy=occupancy,
        duration=duration,
        volume=None,
        price=price,
        adjacency=adjacency,
        cap=cap,
        target_series=target_series,
        target_feature=target_feature,
        node_ids=occ_df.columns.tolist(),
        metadata=metadata,
    )


def distance_to_knn_gaussian_adjacency(
    distance: pd.DataFrame | np.ndarray,
    k: int = 8,
    sigma: float | None = None,
) -> tuple[torch.Tensor, float]:
    dist = np.asarray(distance, dtype=float)
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError(f"distance must be a square matrix, got shape={dist.shape}")

    dist = np.nan_to_num(dist, nan=np.inf, posinf=np.inf, neginf=np.inf)
    dist = (dist + dist.T) / 2.0
    positive = dist[np.isfinite(dist) & (dist > 0.0)]
    if sigma is None or sigma <= 0.0:
        sigma = float(np.median(positive)) if positive.size else 1.0
    sigma = max(float(sigma), 1e-6)

    sim = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
    sim[~np.isfinite(sim)] = 0.0
    np.fill_diagonal(sim, 1.0)

    n_nodes = sim.shape[0]
    k = int(k)
    if k <= 0:
        sim = np.eye(n_nodes, dtype=float)
    elif k < n_nodes - 1:
        mask = np.eye(n_nodes, dtype=bool)
        work = sim.copy()
        np.fill_diagonal(work, -np.inf)
        for row_idx in range(n_nodes):
            finite_idx = np.flatnonzero(np.isfinite(work[row_idx]) & (work[row_idx] > 0.0))
            if finite_idx.size == 0:
                continue
            top_count = min(k, finite_idx.size)
            top_idx = finite_idx[np.argsort(work[row_idx, finite_idx])[-top_count:]]
            mask[row_idx, top_idx] = True
        sim = np.where(mask, sim, 0.0)
        sim = np.maximum(sim, sim.T)
        np.fill_diagonal(sim, 1.0)

    return torch.tensor(sim, dtype=torch.float32), sigma


def _read_optional_charged_matrix(
    data_dir: Path,
    filename: str,
    template: pd.DataFrame,
) -> pd.DataFrame:
    path = data_dir / filename
    if not path.exists():
        return pd.DataFrame(0.0, index=template.index, columns=template.columns)
    return _read_time_site_csv(path).reindex(index=template.index, columns=template.columns).fillna(0.0)


def _load_charged_site(
    data_dir: Path,
    target_feature: str,
    charged_adjacency_k: int,
    charged_adjacency_sigma: float | None,
) -> JournalDatasetBundle:
    if target_feature not in {"duration", "volume"}:
        raise ValueError("CHARGED site data supports target_feature='volume' or 'duration'.")

    duration_df = _read_time_site_csv(data_dir / "duration.csv")
    volume_df = _read_time_site_csv(data_dir / "volume.csv").reindex(
        index=duration_df.index,
        columns=duration_df.columns,
    ).fillna(0.0)
    target_template = volume_df if target_feature == "volume" else duration_df
    node_ids = target_template.columns.tolist()

    sites_path = data_dir / "sites.csv"
    if not sites_path.exists():
        raise FileNotFoundError(f"Required dataset file not found: {sites_path}")
    sites_df = pd.read_csv(sites_path, header=0)
    site_id_col = next(
        (col for col in ("site_id", "site", "station_id") if col in sites_df.columns),
        sites_df.columns[0],
    )
    sites_df[site_id_col] = sites_df[site_id_col].astype(str)
    sites_df = sites_df.set_index(site_id_col, drop=False)
    missing_sites = sorted(set(node_ids) - set(sites_df.index))
    if missing_sites:
        raise ValueError(f"sites.csv is missing {len(missing_sites)} ids, sample={missing_sites[:10]}")
    sites_df = sites_df.loc[node_ids]

    distance_df = _read_square_csv(data_dir / "distance.csv").reindex(index=node_ids, columns=node_ids)
    if distance_df.isna().any().any():
        raise ValueError("distance.csv does not cover the same site ids as duration/volume.")
    adjacency, sigma_used = distance_to_knn_gaussian_adjacency(
        distance_df,
        k=charged_adjacency_k,
        sigma=charged_adjacency_sigma,
    )

    cap = _safe_cap(sites_df["charger_num"])
    duration = _normalise_by_cap(duration_df[node_ids], cap)
    volume = _normalise_by_cap(volume_df[node_ids], cap)
    e_price_df = _read_optional_charged_matrix(data_dir, "e_price.csv", target_template)
    s_price_df = _read_optional_charged_matrix(data_dir, "s_price.csv", target_template)
    price = np.asarray(e_price_df + s_price_df, dtype=float)
    target_series = _select_target(
        target_feature=target_feature,
        occupancy=None,
        duration=duration,
        volume=volume,
    )
    start, end = _time_bounds(target_template)
    city = data_dir.name.removesuffix("_remove_zero")
    variant = "remove_zero" if data_dir.name.endswith("_remove_zero") else "full"

    metadata: dict[str, Any] = {
        "dataset_family": "charged_site",
        "data_dir": str(data_dir),
        "city": city,
        "variant": variant,
        "target_feature": target_feature,
        "node_level": "site",
        "node_count": int(len(node_ids)),
        "node_id_sample": node_ids[:10],
        "time_steps": int(len(target_template)),
        "time_start": start,
        "time_end": end,
        "site_id_column": site_id_col,
        "target_normalized_by": "charger_num",
        "price_source": "e_price.csv + s_price.csv",
        "distance_source": "distance.csv",
        "adjacency_transform": "knn_gaussian_from_distance",
        "charged_adjacency_k": int(charged_adjacency_k),
        "charged_adjacency_sigma": float(sigma_used),
        "target_zero_ratio": float(np.mean(np.asarray(target_template, dtype=float) == 0.0)),
    }
    return JournalDatasetBundle(
        occupancy=None,
        duration=duration,
        volume=volume,
        price=price,
        adjacency=adjacency,
        cap=cap,
        target_series=target_series,
        target_feature=target_feature,
        node_ids=node_ids,
        metadata=metadata,
    )


def load_journal_dataset(
    data_dir: str | Path,
    dataset_family: str = "urbanev_station",
    target_feature: str = "duration",
    charged_adjacency_k: int = 8,
    charged_adjacency_sigma: float | None = None,
    base_dir: Path | None = None,
) -> JournalDatasetBundle:
    if dataset_family not in SUPPORTED_DATASET_FAMILIES:
        raise ValueError(f"Unknown dataset_family={dataset_family!r}; expected {SUPPORTED_DATASET_FAMILIES}")
    if target_feature not in SUPPORTED_TARGET_FEATURES:
        raise ValueError(f"Unknown target_feature={target_feature!r}; expected {SUPPORTED_TARGET_FEATURES}")

    resolved = _resolve_data_dir(data_dir, base_dir=base_dir)
    if dataset_family == "urbanev_station":
        return _load_urbanev_station(resolved, target_feature=target_feature)
    return _load_charged_site(
        resolved,
        target_feature=target_feature,
        charged_adjacency_k=charged_adjacency_k,
        charged_adjacency_sigma=charged_adjacency_sigma,
    )


def infer_target_feature_from_metadata(metadata: dict[str, Any]) -> str:
    explicit = metadata.get("target_feature")
    if explicit:
        return str(explicit)

    model_name = str(metadata.get("model_name", "")).lower()
    if "dura" in model_name:
        return "duration"
    if "volume" in model_name or "vol" in model_name:
        return "volume"
    return "occupancy"


def load_journal_dataset_from_metadata(
    metadata: dict[str, Any],
    base_dir: Path,
) -> JournalDatasetBundle:
    dataset_metadata = metadata.get("dataset_metadata")
    if not isinstance(dataset_metadata, dict):
        dataset_metadata = {}

    dataset_family = str(metadata.get("dataset_family", dataset_metadata.get("dataset_family", "urbanev_station")))
    target_feature = infer_target_feature_from_metadata(metadata)
    charged_adjacency_k = int(metadata.get(
        "charged_adjacency_k",
        dataset_metadata.get("charged_adjacency_k", 8),
    ))
    charged_adjacency_sigma_value = metadata.get(
        "charged_adjacency_sigma",
        dataset_metadata.get("charged_adjacency_sigma"),
    )
    charged_adjacency_sigma = (
        None
        if charged_adjacency_sigma_value in (None, "", 0, 0.0)
        else float(charged_adjacency_sigma_value)
    )
    return load_journal_dataset(
        data_dir=str(metadata["data_dir"]),
        dataset_family=dataset_family,
        target_feature=target_feature,
        charged_adjacency_k=charged_adjacency_k,
        charged_adjacency_sigma=charged_adjacency_sigma,
        base_dir=base_dir,
    )
