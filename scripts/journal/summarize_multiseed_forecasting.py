from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def load_metadata(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def parse_seed(run_dir: Path, metadata: dict[str, Any]) -> int | None:
    if "seed" in metadata:
        return int(metadata["seed"])
    match = re.search(r"seed[_-]?(\d+)", run_dir.name)
    return int(match.group(1)) if match else None


def read_metric_file(path: Path, context: dict[str, Any]) -> pd.DataFrame:
    df = pd.read_csv(path)
    for key, value in reversed(list(context.items())):
        df.insert(0, key, value)
    return df


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    flat_columns: list[str] = []
    for column in df.columns:
        if isinstance(column, tuple):
            flat_columns.append("_".join(str(part) for part in column if str(part)))
        else:
            flat_columns.append(str(column))
    df.columns = flat_columns
    return df


def summarize_numeric(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    numeric_cols = [
        column
        for column in df.select_dtypes(include="number").columns
        if column not in set(group_cols + ["seed"])
    ]
    if not numeric_cols:
        return pd.DataFrame()
    summary = df.groupby(group_cols, dropna=False)[numeric_cols].agg(["mean", "std", "count"]).reset_index()
    return flatten_columns(summary)


def summarize_overall(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    numeric_cols = [
        column
        for column in df.select_dtypes(include="number").columns
        if column not in set(group_cols + ["seed", "horizon"])
    ]
    if not numeric_cols:
        return pd.DataFrame()
    per_seed = df.groupby(group_cols + ["seed"], dropna=False)[numeric_cols].mean().reset_index()
    return summarize_numeric(per_seed, group_cols)


def collect_runs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest_rows: list[dict[str, Any]] = []
    point_frames: list[pd.DataFrame] = []
    interval_frames: list[pd.DataFrame] = []
    crossing_frames: list[pd.DataFrame] = []

    for run_dir in sorted(root.glob("*/seed_*")):
        if not run_dir.is_dir():
            continue
        model_key = run_dir.parent.name
        metadata = load_metadata(run_dir)
        seed = parse_seed(run_dir, metadata)
        context = {
            "model_key": model_key,
            "seed": seed,
            "run_dir": str(run_dir),
            "architecture": metadata.get("architecture", ""),
        }
        point_path = run_dir / "point_metrics_by_horizon.csv"
        interval_path = run_dir / "raw_interval_metrics_by_horizon.csv"
        crossing_path = run_dir / "quantile_crossing_metrics_by_horizon.csv"
        manifest_rows.append({
            **context,
            "model_name": metadata.get("model_name", ""),
            "created_at": metadata.get("created_at", ""),
            "commit": metadata.get("commit", ""),
            "point_metrics": point_path.exists(),
            "raw_interval_metrics": interval_path.exists(),
            "crossing_metrics": crossing_path.exists(),
        })
        if point_path.exists():
            point_frames.append(read_metric_file(point_path, context))
        if interval_path.exists():
            interval_frames.append(read_metric_file(interval_path, context))
        if crossing_path.exists():
            crossing_frames.append(read_metric_file(crossing_path, context))

    manifest = pd.DataFrame(manifest_rows)
    point = pd.concat(point_frames, ignore_index=True) if point_frames else pd.DataFrame()
    interval = pd.concat(interval_frames, ignore_index=True) if interval_frames else pd.DataFrame()
    crossing = pd.concat(crossing_frames, ignore_index=True) if crossing_frames else pd.DataFrame()
    return manifest, point, interval, crossing


def write_if_present(df: pd.DataFrame, path: Path) -> None:
    if not df.empty:
        df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize multi-seed Stage 1 forecasting results.")
    parser.add_argument("--root", default="journal_results/shenzhen_multihorizon/multiseed")
    parser.add_argument("--output-dir", default="journal_results/shenzhen_multihorizon/multiseed_summary")
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    if not root.exists():
        raise FileNotFoundError(f"Multi-seed root not found: {root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, point, interval, crossing = collect_runs(root)
    write_if_present(manifest, output_dir / "multiseed_run_manifest.csv")

    write_if_present(point, output_dir / "point_metrics_long.csv")
    write_if_present(
        summarize_numeric(point, ["model_key", "horizon"]),
        output_dir / "point_metrics_by_horizon_summary.csv",
    )
    write_if_present(
        summarize_overall(point, ["model_key"]),
        output_dir / "point_metrics_overall_summary.csv",
    )

    write_if_present(interval, output_dir / "raw_interval_metrics_long.csv")
    write_if_present(
        summarize_numeric(interval, ["model_key", "horizon", "delta"]),
        output_dir / "raw_interval_metrics_by_horizon_summary.csv",
    )
    write_if_present(
        summarize_overall(interval, ["model_key", "delta"]),
        output_dir / "raw_interval_metrics_overall_summary.csv",
    )

    write_if_present(crossing, output_dir / "quantile_crossing_metrics_long.csv")
    write_if_present(
        summarize_numeric(crossing, ["model_key", "horizon"]),
        output_dir / "quantile_crossing_metrics_by_horizon_summary.csv",
    )

    print(f"Runs found: {len(manifest)}")
    if not manifest.empty:
        print(manifest.groupby("model_key")["seed"].nunique().to_string())
    print(f"Saved summaries to: {output_dir}")


if __name__ == "__main__":
    main()
