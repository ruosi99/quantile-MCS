from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.journal.calibrate_multihorizon_cqr import (
    build_calib_test_loaders,
    checkpoint_from_metadata,
    conformal_quantile,
    load_stage1_metadata,
    predict_loader,
    resolve_path,
)
from scripts.journal.train_multihorizon_raw import parse_bool, quantile_index
from utils.model_training.journal_data import load_journal_dataset_from_metadata
from utils.model_training.journal_contracts import (
    assert_monotonic_quantiles,
    assert_quantile_tensor_shape,
    assert_target_tensor_shape,
)


DEFAULT_COST_RATIOS = "1:1,3:1,5:1,9:1,19:1"
METHODS = ["median", "raw_target_quantile", "global_cqr", "horizon_cqr", "oracle_decision"]


@dataclass(frozen=True)
class CostRatio:
    label: str
    c_u: float
    c_o: float
    tau_star: float


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def format_cost_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def parse_cost_ratios(value: str) -> list[CostRatio]:
    ratios: list[CostRatio] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Cost ratio must use c_u:c_o format, got {token!r}")
        left, right = token.split(":", maxsplit=1)
        c_u = float(left)
        c_o = float(right)
        if c_u <= 0 or c_o <= 0:
            raise ValueError(f"Cost ratio values must be positive, got {token!r}")
        tau_star = c_u / (c_u + c_o)
        ratios.append(
            CostRatio(
                label=f"{format_cost_value(c_u)}:{format_cost_value(c_o)}",
                c_u=c_u,
                c_o=c_o,
                tau_star=tau_star,
            )
        )
    if not ratios:
        raise ValueError("At least one cost ratio is required.")
    return ratios


def newsvendor_cost(decision: np.ndarray, labels: np.ndarray, c_u: float, c_o: float) -> np.ndarray:
    shortage = np.maximum(labels - decision, 0.0)
    overage = np.maximum(decision - labels, 0.0)
    return c_u * shortage + c_o * overage


def decision_metric_values(
    decision: np.ndarray,
    labels: np.ndarray,
    c_u: float,
    c_o: float,
    oracle_cost: np.ndarray | None = None,
) -> dict[str, float]:
    decision = np.asarray(decision, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    cost = newsvendor_cost(decision, labels, c_u=c_u, c_o=c_o)
    if oracle_cost is None:
        oracle_cost = newsvendor_cost(labels, labels, c_u=c_u, c_o=c_o)

    shortage = np.maximum(labels - decision, 0.0)
    overage = np.maximum(decision - labels, 0.0)
    expected_cost = float(np.mean(cost))
    oracle_expected_cost = float(np.mean(oracle_cost))
    return {
        "expected_cost": expected_cost,
        "oracle_expected_cost": oracle_expected_cost,
        "regret": float(expected_cost - oracle_expected_cost),
        "event_shortage_rate": float(np.mean(decision < labels)),
        "shortage_amount": float(np.mean(shortage)),
        "event_overage_rate": float(np.mean(decision > labels)),
        "overage_amount": float(np.mean(overage)),
        "mean_decision": float(np.mean(decision)),
        "mean_label": float(np.mean(labels)),
    }


def one_sided_scores(predict_quantiles: np.ndarray, labels: np.ndarray, q_idx: int) -> np.ndarray:
    return np.asarray(labels, dtype=np.float64) - np.asarray(predict_quantiles[..., q_idx], dtype=np.float64)


def compute_one_sided_global_threshold(scores: np.ndarray, tau_star: float) -> float:
    return conformal_quantile(scores, delta=1.0 - tau_star)


def compute_one_sided_horizon_thresholds(
    scores: np.ndarray,
    horizons: list[int],
    tau_star: float,
) -> dict[int, float]:
    if scores.ndim != 3 or scores.shape[2] != len(horizons):
        raise AssertionError(f"scores must have shape (T, N, H) with H={len(horizons)}, got {scores.shape}")
    return {
        int(horizon): compute_one_sided_global_threshold(scores[:, :, h_idx], tau_star=tau_star)
        for h_idx, horizon in enumerate(horizons)
    }


def clip_decision(decision: np.ndarray, nonnegative: bool = True) -> np.ndarray:
    if not nonnegative:
        return np.asarray(decision, dtype=np.float64)
    return np.maximum(np.asarray(decision, dtype=np.float64), 0.0)


def build_one_sided_thresholds(
    calib_quantiles: np.ndarray,
    calib_labels: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    cost_ratios: list[CostRatio],
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    rows: list[dict[str, float | int | str]] = []
    lookup: dict[str, dict[str, object]] = {}

    for ratio in cost_ratios:
        q_idx = quantile_index(quantiles, ratio.tau_star)
        scores = one_sided_scores(calib_quantiles, calib_labels, q_idx=q_idx)
        global_s = compute_one_sided_global_threshold(scores, tau_star=ratio.tau_star)
        horizon_s = compute_one_sided_horizon_thresholds(scores, horizons=horizons, tau_star=ratio.tau_star)

        lookup[ratio.label] = {
            "q_idx": q_idx,
            "q_value": float(quantiles[q_idx]),
            "global": float(global_s),
            "horizon": horizon_s,
        }
        rows.append(
            {
                "method": "global_cqr",
                "cost_ratio": ratio.label,
                "c_u": ratio.c_u,
                "c_o": ratio.c_o,
                "tau_star": ratio.tau_star,
                "target_quantile": float(quantiles[q_idx]),
                "horizon": "all",
                "s_hat": float(global_s),
                "score_definition": "label_minus_raw_target_quantile",
            }
        )
        for horizon in horizons:
            rows.append(
                {
                    "method": "horizon_cqr",
                    "cost_ratio": ratio.label,
                    "c_u": ratio.c_u,
                    "c_o": ratio.c_o,
                    "tau_star": ratio.tau_star,
                    "target_quantile": float(quantiles[q_idx]),
                    "horizon": int(horizon),
                    "s_hat": float(horizon_s[int(horizon)]),
                    "score_definition": "label_minus_raw_target_quantile",
                }
            )

    return pd.DataFrame(rows), lookup


def evaluate_decision_methods(
    test_quantiles: np.ndarray,
    labels: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    cost_ratios: list[CostRatio],
    threshold_lookup: dict[str, dict[str, object]],
    nonnegative: bool = True,
) -> pd.DataFrame:
    assert_quantile_tensor_shape(
        test_quantiles,
        n_horizons=len(horizons),
        n_quantiles=len(quantiles),
        context="stage4 test_quantiles",
    )
    assert_target_tensor_shape(labels, n_horizons=len(horizons), context="stage4 labels")

    q50_idx = quantile_index(quantiles, 0.5)
    rows: list[dict[str, float | int | str]] = []

    for ratio in cost_ratios:
        info = threshold_lookup[ratio.label]
        target_idx = int(info["q_idx"])
        target_quantile = float(info["q_value"])
        global_s = float(info["global"])
        horizon_thresholds = {int(k): float(v) for k, v in dict(info["horizon"]).items()}

        median_all = clip_decision(test_quantiles[..., q50_idx], nonnegative=nonnegative)
        target_all = clip_decision(test_quantiles[..., target_idx], nonnegative=nonnegative)
        global_all = clip_decision(test_quantiles[..., target_idx] + global_s, nonnegative=nonnegative)
        horizon_all = np.empty_like(target_all, dtype=np.float64)
        for h_idx, horizon in enumerate(horizons):
            horizon_all[:, :, h_idx] = clip_decision(
                test_quantiles[:, :, h_idx, target_idx] + horizon_thresholds[int(horizon)],
                nonnegative=nonnegative,
            )
        oracle_all = clip_decision(labels, nonnegative=nonnegative)

        method_decisions = {
            "median": median_all,
            "raw_target_quantile": target_all,
            "global_cqr": global_all,
            "horizon_cqr": horizon_all,
            "oracle_decision": oracle_all,
        }

        for h_idx, horizon in enumerate(horizons):
            labels_h = labels[:, :, h_idx]
            oracle_cost = newsvendor_cost(labels_h, labels_h, c_u=ratio.c_u, c_o=ratio.c_o)
            for method in METHODS:
                decision_h = method_decisions[method][:, :, h_idx]
                values = decision_metric_values(
                    decision=decision_h,
                    labels=labels_h,
                    c_u=ratio.c_u,
                    c_o=ratio.c_o,
                    oracle_cost=oracle_cost,
                )
                threshold = 0.0
                if method == "global_cqr":
                    threshold = global_s
                elif method == "horizon_cqr":
                    threshold = horizon_thresholds[int(horizon)]

                rows.append(
                    {
                        "method": method,
                        "cost_ratio": ratio.label,
                        "c_u": ratio.c_u,
                        "c_o": ratio.c_o,
                        "tau_star": ratio.tau_star,
                        "target_quantile": target_quantile,
                        "target_quantile_index": target_idx,
                        "horizon": int(horizon),
                        "s_hat": float(threshold),
                        **values,
                    }
                )

    return pd.DataFrame(rows)


def build_summary_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["method", "cost_ratio", "c_u", "c_o", "tau_star", "target_quantile"]
    metric_cols = [
        "expected_cost",
        "oracle_expected_cost",
        "regret",
        "event_shortage_rate",
        "shortage_amount",
        "event_overage_rate",
        "overage_amount",
        "mean_decision",
        "mean_label",
    ]
    summary = metrics_df.groupby(group_cols, as_index=False)[metric_cols].mean()
    horizon_counts = metrics_df.groupby(group_cols, as_index=False)["horizon"].nunique().rename(
        columns={"horizon": "horizon_count"}
    )
    return summary.merge(horizon_counts, on=group_cols, how="left")


def build_regret_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    return summary_df[
        [
            "method",
            "cost_ratio",
            "c_u",
            "c_o",
            "tau_star",
            "target_quantile",
            "expected_cost",
            "oracle_expected_cost",
            "regret",
            "event_shortage_rate",
            "event_overage_rate",
        ]
    ].copy()


def maybe_write_plots(metrics_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    written: list[str] = []
    methods = [method for method in METHODS if method != "oracle_decision"]

    horizon_plot = metrics_df[metrics_df["method"].isin(methods)]
    horizon_plot = horizon_plot.groupby(["method", "horizon"], as_index=False)["expected_cost"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        subset = horizon_plot[horizon_plot["method"] == method]
        ax.plot(subset["horizon"], subset["expected_cost"], marker="o", label=method)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Expected cost")
    ax.set_title("Decision cost by horizon")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    cost_path = output_dir / "decision_cost_by_horizon.png"
    fig.savefig(cost_path, dpi=200)
    plt.close(fig)
    written.append(cost_path.name)

    regret_plot = summary_df[summary_df["method"].isin(methods)]
    fig, ax = plt.subplots(figsize=(9, 5))
    x_labels = list(dict.fromkeys(regret_plot["cost_ratio"].tolist()))
    x = np.arange(len(x_labels))
    width = 0.8 / len(methods)
    for idx, method in enumerate(methods):
        subset = regret_plot[regret_plot["method"] == method].set_index("cost_ratio").reindex(x_labels)
        ax.bar(x + (idx - (len(methods) - 1) / 2) * width, subset["regret"], width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Cost ratio")
    ax.set_ylabel("Regret vs oracle")
    ax.set_title("Decision regret by cost ratio")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    regret_path = output_dir / "decision_regret_by_cost_ratio.png"
    fig.savefig(regret_path, dpi=200)
    plt.close(fig)
    written.append(regret_path.name)

    return written


def load_test_arrays(stage1_output_dir: Path, max_test_windows: int = 0) -> tuple[np.ndarray, np.ndarray]:
    quantiles_path = stage1_output_dir / "predict_quantiles.npy"
    labels_path = stage1_output_dir / "label_list.npy"
    if not quantiles_path.exists():
        raise FileNotFoundError(f"Missing Stage 1 test quantiles: {quantiles_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing Stage 1 test labels: {labels_path}")

    test_quantiles = np.load(quantiles_path)
    labels = np.load(labels_path)
    if max_test_windows > 0:
        test_quantiles = test_quantiles[:max_test_windows]
        labels = labels[:max_test_windows]
    return test_quantiles, labels


def validate_stage2_inputs(stage2_output_dir: Path) -> dict[str, object]:
    expected_files = [
        "cqr_thresholds.csv",
        "global_summary.csv",
        "raw_vs_cqr_comparison_by_horizon.csv",
        "calibrated_interval_metrics_by_horizon.csv",
        "stage2_metadata.json",
    ]
    missing = [name for name in expected_files if not (stage2_output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 2 input files in {stage2_output_dir}: {missing}")
    return json.loads((stage2_output_dir / "stage2_metadata.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 multi-horizon decision-first evaluation.")
    parser.add_argument("--stage1-output-dir", required=True)
    parser.add_argument("--stage2-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--use-cuda", type=parse_bool, default=True)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--cost-ratios", default=DEFAULT_COST_RATIOS)
    parser.add_argument("--max-calib-batches", type=int, default=0)
    parser.add_argument("--max-test-windows", type=int, default=0)
    parser.add_argument("--machine", default="Lenovo")
    args = parser.parse_args()

    stage1_output_dir = resolve_path(args.stage1_output_dir)
    stage2_output_dir = resolve_path(args.stage2_output_dir)
    output_dir = resolve_path(args.output_dir)

    metadata = load_stage1_metadata(stage1_output_dir)
    stage2_metadata = validate_stage2_inputs(stage2_output_dir)
    checkpoint_path = checkpoint_from_metadata(stage1_output_dir, metadata)

    horizons = [int(h) for h in metadata["horizons"]]
    quantiles = [float(q) for q in metadata["quantiles"]]
    cost_ratios = parse_cost_ratios(args.cost_ratios)
    for ratio in cost_ratios:
        quantile_index(quantiles, ratio.tau_star)

    seq_len = int(metadata["seq_len"])
    batch_size = int(args.batch_size or metadata["batch_size"])
    split_rates = {key: float(value) for key, value in metadata["split_rates"].items()}
    model_name = str(metadata["model_name"])
    data_dir = resolve_path(str(metadata["data_dir"]))

    device = torch.device("cuda:0" if args.use_cuda and torch.cuda.is_available() else "cpu")
    dataset_bundle = load_journal_dataset_from_metadata(metadata, base_dir=REPO_ROOT)
    input_series = dataset_bundle.target_series
    price_raw = dataset_bundle.price
    cap = dataset_bundle.cap
    calib_loader, _, split_lengths = build_calib_test_loaders(
        input_series=input_series,
        price_raw=price_raw,
        seq_len=seq_len,
        horizons=horizons,
        batch_size=batch_size,
        split_rates=split_rates,
        device=device,
    )

    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model.to(device)
    calib_quantiles, calib_labels = predict_loader(
        model=model,
        loader=calib_loader,
        device=device,
        cap=cap,
        quantiles=quantiles,
        horizons=horizons,
        max_batches=args.max_calib_batches,
        context="stage4 one-sided calibration",
    )
    assert_quantile_tensor_shape(
        calib_quantiles,
        n_horizons=len(horizons),
        n_quantiles=len(quantiles),
        context="stage4 calibration quantiles",
    )
    assert_target_tensor_shape(calib_labels, n_horizons=len(horizons), context="stage4 calibration labels")

    thresholds_df, threshold_lookup = build_one_sided_thresholds(
        calib_quantiles=calib_quantiles,
        calib_labels=calib_labels,
        quantiles=quantiles,
        horizons=horizons,
        cost_ratios=cost_ratios,
    )
    calib_shapes = {
        "calib_quantiles": list(calib_quantiles.shape),
        "calib_labels": list(calib_labels.shape),
    }
    del calib_quantiles, calib_labels, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    test_quantiles, labels = load_test_arrays(stage1_output_dir, max_test_windows=args.max_test_windows)
    assert_quantile_tensor_shape(
        test_quantiles,
        n_horizons=len(horizons),
        n_quantiles=len(quantiles),
        context="stage4 loaded test quantiles",
    )
    assert_target_tensor_shape(labels, n_horizons=len(horizons), context="stage4 loaded labels")
    assert_monotonic_quantiles(test_quantiles, context="stage4 loaded test quantiles")

    metrics_df = evaluate_decision_methods(
        test_quantiles=test_quantiles,
        labels=labels,
        quantiles=quantiles,
        horizons=horizons,
        cost_ratios=cost_ratios,
        threshold_lookup=threshold_lookup,
        nonnegative=True,
    )
    summary_df = build_summary_table(metrics_df)
    regret_df = build_regret_table(summary_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "decision_metrics_by_horizon.csv", index=False)
    summary_df.to_csv(output_dir / "decision_summary_by_method_and_cost_ratio.csv", index=False)
    regret_df.to_csv(output_dir / "decision_regret_vs_oracle.csv", index=False)
    thresholds_df.to_csv(output_dir / "decision_one_sided_thresholds.csv", index=False)
    plot_files = maybe_write_plots(metrics_df, summary_df, output_dir)

    tau_mapping = [
        {
            "cost_ratio": ratio.label,
            "c_u": ratio.c_u,
            "c_o": ratio.c_o,
            "tau_star": ratio.tau_star,
            "target_quantile": float(quantiles[quantile_index(quantiles, ratio.tau_star)]),
            "target_quantile_index": int(quantile_index(quantiles, ratio.tau_star)),
        }
        for ratio in cost_ratios
    ]
    stage4_metadata = {
        "stage": "4.1-4.2",
        "goal": "Decision-first evaluation under asymmetric newsvendor costs",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "machine": str(args.machine),
        "branch": git_value(["branch", "--show-current"]),
        "commit": git_value(["rev-parse", "--short", "HEAD"]),
        "stage1_output_dir": str(stage1_output_dir),
        "stage2_output_dir": str(stage2_output_dir),
        "stage1_checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "stage1_model_name": model_name,
        "stage2_commit": stage2_metadata.get("commit", "unknown"),
        "data_dir": str(data_dir),
        "dataset_family": dataset_bundle.metadata.get("dataset_family", "urbanev_station"),
        "target_feature": dataset_bundle.target_feature,
        "dataset_metadata": dataset_bundle.metadata,
        "calibration_mode": "direct_one_sided_conformal",
        "score_definition": "label_minus_raw_target_quantile",
        "decision_quantity": "max(raw_target_quantile + s_hat, 0) for calibrated methods",
        "cost_function": "c_u * max(y - q, 0) + c_o * max(q - y, 0)",
        "aggregation_rule": "atomic sample-station-horizon metrics, then simple arithmetic mean across horizons",
        "horizons": horizons,
        "quantiles": quantiles,
        "cost_ratios": tau_mapping,
        "batch_size": batch_size,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "split_rates": split_rates,
        "split_lengths": split_lengths,
        "prediction_shapes": {
            **calib_shapes,
            "test_quantiles": list(test_quantiles.shape),
            "test_labels": list(labels.shape),
        },
        "smoke_limits": {
            "max_calib_batches": args.max_calib_batches,
            "max_test_windows": args.max_test_windows,
        },
        "output_files": {
            "summary": "decision_summary_by_method_and_cost_ratio.csv",
            "horizon_metrics": "decision_metrics_by_horizon.csv",
            "regret": "decision_regret_vs_oracle.csv",
            "metadata": "decision_eval_metadata.json",
            "thresholds": "decision_one_sided_thresholds.csv",
            "plots": plot_files,
        },
    }
    (output_dir / "decision_eval_metadata.json").write_text(
        json.dumps(stage4_metadata, indent=2),
        encoding="utf-8",
    )

    print(summary_df.to_string(index=False))
    print(json.dumps(stage4_metadata, indent=2))


if __name__ == "__main__":
    main()
