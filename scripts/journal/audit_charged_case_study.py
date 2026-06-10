from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


DEFAULT_CITIES = ["AMS", "JHB", "LOA", "MEL", "SPO", "SZH"]


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def read_matrix(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, header=0)
    df.columns = df.columns.astype(str)
    return df


def site_id_column(sites: pd.DataFrame) -> str:
    return next((col for col in ("site_id", "site", "station_id") if col in sites.columns), sites.columns[0])


def zero_ratio(df: pd.DataFrame | None) -> float:
    if df is None or df.empty:
        return float("nan")
    values = pd.to_numeric(df.stack(), errors="coerce")
    return float((values == 0.0).mean())


def missing_ratio(df: pd.DataFrame | None) -> float:
    if df is None or df.empty:
        return float("nan")
    return float(df.isna().to_numpy().mean())


def mean_value(df: pd.DataFrame | None) -> float:
    if df is None or df.empty:
        return float("nan")
    values = pd.to_numeric(df.stack(), errors="coerce")
    return float(values.mean())


def inspect_city(data_dir: Path, city: str, variant: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    issues: list[dict[str, object]] = []
    duration = read_matrix(data_dir / "duration.csv")
    volume = read_matrix(data_dir / "volume.csv")
    e_price = read_matrix(data_dir / "e_price.csv")
    s_price = read_matrix(data_dir / "s_price.csv")
    distance = read_matrix(data_dir / "distance.csv")

    sites_path = data_dir / "sites.csv"
    sites = pd.read_csv(sites_path) if sites_path.exists() else None
    if sites is None:
        issues.append({"city": city, "variant": variant, "severity": "error", "issue": "missing sites.csv"})

    target = volume if volume is not None else duration
    target_cols = target.columns.astype(str).tolist() if target is not None else []
    site_count = len(target_cols)
    time_index = pd.to_datetime(target.index, errors="coerce") if target is not None else pd.DatetimeIndex([])
    hourly_contiguous = False
    if len(time_index) > 1 and not time_index.isna().any():
        diffs = pd.Series(time_index).diff().dropna()
        hourly_contiguous = bool((diffs == pd.Timedelta(hours=1)).all())

    if duration is None:
        issues.append({"city": city, "variant": variant, "severity": "warn", "issue": "missing duration.csv"})
    if volume is None:
        issues.append({"city": city, "variant": variant, "severity": "error", "issue": "missing volume.csv"})

    if sites is not None and target is not None:
        id_col = site_id_column(sites)
        site_ids = set(sites[id_col].astype(str))
        missing_in_sites = sorted(set(target_cols) - site_ids)
        extra_in_sites = sorted(site_ids - set(target_cols))
        if missing_in_sites:
            issues.append({
                "city": city,
                "variant": variant,
                "severity": "error",
                "issue": "target columns missing from sites.csv",
                "count": len(missing_in_sites),
                "sample": ",".join(missing_in_sites[:10]),
            })
        if extra_in_sites:
            issues.append({
                "city": city,
                "variant": variant,
                "severity": "info",
                "issue": "sites.csv has ids not present in target columns",
                "count": len(extra_in_sites),
                "sample": ",".join(extra_in_sites[:10]),
            })

    if distance is not None and target is not None:
        dist_ids = set(distance.index.astype(str)) & set(distance.columns.astype(str))
        missing_distance = sorted(set(target_cols) - dist_ids)
        if missing_distance:
            issues.append({
                "city": city,
                "variant": variant,
                "severity": "error",
                "issue": "target columns missing from distance.csv",
                "count": len(missing_distance),
                "sample": ",".join(missing_distance[:10]),
            })

    price_available = e_price is not None and s_price is not None
    price_nonzero_ratio = float("nan")
    if price_available:
        price = e_price.reindex(columns=target_cols).fillna(0.0) + s_price.reindex(columns=target_cols).fillna(0.0)
        price_nonzero_ratio = float((np.asarray(price, dtype=float) != 0.0).mean())

    row = {
        "city": city,
        "variant": variant,
        "data_dir": str(data_dir),
        "site_count": int(site_count),
        "time_steps": int(len(target)) if target is not None else 0,
        "time_start": str(time_index[0]) if len(time_index) else "",
        "time_end": str(time_index[-1]) if len(time_index) else "",
        "hourly_contiguous": hourly_contiguous,
        "duration_available": duration is not None,
        "duration_zero_ratio": zero_ratio(duration),
        "duration_mean": mean_value(duration),
        "duration_missing_ratio": missing_ratio(duration),
        "volume_available": volume is not None,
        "volume_zero_ratio": zero_ratio(volume),
        "volume_mean": mean_value(volume),
        "volume_missing_ratio": missing_ratio(volume),
        "price_available": price_available,
        "price_nonzero_ratio": price_nonzero_ratio,
        "distance_available": distance is not None,
        "issue_count": len(issues),
    }
    return row, issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CHARGED data for the journal case study.")
    parser.add_argument("--charged-root", default="CHARGED/data")
    parser.add_argument("--cities", default=",".join(DEFAULT_CITIES))
    parser.add_argument("--variants", default="remove_zero,full")
    parser.add_argument("--output-dir", default="journal_results/charged_case_study/audit")
    args = parser.parse_args()

    charged_root = resolve_path(args.charged_root)
    output_dir = resolve_path(args.output_dir)
    cities = parse_csv_list(args.cities)
    variants = parse_csv_list(args.variants)

    rows: list[dict[str, object]] = []
    issues: list[dict[str, object]] = []
    for city in cities:
        for variant in variants:
            dir_name = city if variant == "full" else f"{city}_{variant}"
            data_dir = charged_root / dir_name
            if not data_dir.exists():
                issues.append({
                    "city": city,
                    "variant": variant,
                    "severity": "error",
                    "issue": "missing city directory",
                    "data_dir": str(data_dir),
                })
                continue
            row, city_issues = inspect_city(data_dir, city=city, variant=variant)
            rows.append(row)
            issues.extend(city_issues)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(rows)
    issue_df = pd.DataFrame(issues)
    manifest.to_csv(output_dir / "charged_city_manifest.csv", index=False)
    issue_df.to_csv(output_dir / "charged_data_issues.csv", index=False)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "charged_root": str(charged_root),
        "cities": cities,
        "variants": variants,
        "output_files": {
            "manifest": "charged_city_manifest.csv",
            "issues": "charged_data_issues.csv",
        },
    }
    (output_dir / "audit_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(manifest.to_string(index=False))
    if not issue_df.empty:
        print(issue_df.to_string(index=False))


if __name__ == "__main__":
    main()
