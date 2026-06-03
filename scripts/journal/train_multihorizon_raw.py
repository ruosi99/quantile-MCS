from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import utils.model_training.models as models
import utils.model_training.training_utils as fn
from utils.model_training.conformal import interval_metrics
from utils.model_training.journal_contracts import (
    DEFAULT_JOURNAL_HORIZONS,
    DEFAULT_JOURNAL_QUANTILES,
    assert_monotonic_quantiles,
    assert_quantile_tensor_shape,
    assert_target_tensor_shape,
)
from utils.model_training.loss_functions import QuantileLoss


POINT_METRIC_COLUMNS = ["MSE", "RMSE", "MAPE", "RAE", "MAE", "R2", "MedAE", "EVS"]


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y"}


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def quantile_index(quantiles: list[float], value: float) -> int:
    for idx, quantile in enumerate(quantiles):
        if abs(quantile - value) < 1e-9:
            return idx
    raise ValueError(f"Quantile {value} not found in {quantiles}")


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def load_matching_state_dict(
    target_model: torch.nn.Module,
    source_model_or_state: torch.nn.Module | dict[str, torch.Tensor],
) -> dict[str, object]:
    if isinstance(source_model_or_state, torch.nn.Module):
        source_state = source_model_or_state.state_dict()
    else:
        source_state = source_model_or_state

    target_state = target_model.state_dict()
    loadable_state = {}
    loaded_keys = []
    skipped_keys = []

    for key, target_value in target_state.items():
        source_value = source_state.get(key)
        if source_value is None:
            skipped_keys.append({
                "key": key,
                "reason": "missing_from_source",
                "target_shape": list(target_value.shape),
            })
            continue
        if tuple(source_value.shape) != tuple(target_value.shape):
            skipped_keys.append({
                "key": key,
                "reason": "shape_mismatch",
                "source_shape": list(source_value.shape),
                "target_shape": list(target_value.shape),
            })
            continue

        loadable_state[key] = source_value
        loaded_keys.append(key)

    target_model.load_state_dict(loadable_state, strict=False)
    return {
        "loaded_key_count": len(loaded_keys),
        "skipped_key_count": len(skipped_keys),
        "loaded_keys": loaded_keys,
        "skipped_keys": skipped_keys,
    }


def warm_start_model(
    target_model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> dict[str, object]:
    if not checkpoint_path:
        return {"enabled": False}

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {path}")

    source = torch.load(path, map_location=device, weights_only=False)
    if isinstance(source, dict) and "state_dict" in source:
        source = source["state_dict"]

    report = load_matching_state_dict(target_model, source)
    report.update({
        "enabled": True,
        "checkpoint": str(path),
    })
    print(
        "Warm-start loaded "
        f"{report['loaded_key_count']} matching keys; "
        f"skipped {report['skipped_key_count']} keys."
    )
    return report


def point_metrics(test_pre: np.ndarray, test_real: np.ndarray) -> dict[str, float]:
    eps = 0.01
    valid_indices = test_real > eps
    if np.any(valid_indices):
        mape = np.mean(np.abs((test_real[valid_indices] - test_pre[valid_indices]) / test_real[valid_indices])) * 100
    else:
        mape = float("nan")

    mae = np.mean(np.abs(test_real - test_pre))
    mse = np.mean((test_real - test_pre) ** 2)
    rmse = np.sqrt(mse)
    sst = np.sum((test_real - np.mean(test_real)) ** 2) + eps
    ssr = np.sum((test_real - test_pre) ** 2)
    r2 = 1 - (ssr / sst)
    rae = np.sum(np.abs(test_pre - test_real)) / (np.sum(np.abs(test_real - np.mean(test_real))) + eps)
    medae = np.median(np.abs(test_real - test_pre))
    diff = test_real - test_pre
    evs = 1 - (np.var(diff) / (np.var(test_real) + eps))

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "RAE": float(rae),
        "MAE": float(mae),
        "R2": float(r2),
        "MedAE": float(medae),
        "EVS": float(evs),
    }


def quantile_crossing_metrics(pred_q: np.ndarray) -> dict[str, float]:
    diffs = np.diff(pred_q, axis=-1)
    crossing_mask = diffs < 0
    return {
        "crossing_rate": float(crossing_mask.any(axis=-1).mean()),
        "mean_crossing_magnitude": float(np.maximum(-diffs, 0.0).mean()),
        "mean_crossed_pairs": float(crossing_mask.sum(axis=-1).mean()),
        "max_crossing_magnitude": float(np.maximum(-diffs, 0.0).max()),
    }


def limited_loader(loader: DataLoader, max_batches: int):
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        yield batch_idx, batch


def build_dataloaders(
    input_series: np.ndarray,
    price_raw: np.ndarray,
    seq_len: int,
    horizons: list[int],
    batch_size: int,
    split_rates: dict[str, float],
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    train_demand, valid_demand, calib_demand, test_demand = fn.division(
        input_series,
        train_rate=split_rates["train"],
        valid_rate=split_rates["valid"],
        calib_rate=split_rates["calib"],
        test_rate=split_rates["test"],
    )
    train_price, valid_price, calib_price, test_price = fn.division(
        price_raw,
        train_rate=split_rates["train"],
        valid_rate=split_rates["valid"],
        calib_rate=split_rates["calib"],
        test_rate=split_rates["test"],
    )

    train_dataset = fn.CreateMultiHorizonDataset(train_demand, train_price, seq_len, horizons, device)
    valid_dataset = fn.CreateMultiHorizonDataset(valid_demand, valid_price, seq_len, horizons, device)
    test_dataset = fn.CreateMultiHorizonDataset(test_demand, test_price, seq_len, horizons, device)

    loaders = (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True),
        DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False),
    )
    split_lengths = {
        "train": int(len(train_demand)),
        "valid": int(len(valid_demand)),
        "calib": int(len(calib_demand)),
        "test": int(len(test_demand)),
        "train_windows": int(len(train_dataset)),
        "valid_windows": int(len(valid_dataset)),
        "test_windows": int(len(test_dataset)),
        "calib_length_reserved_for_stage2": int(len(calib_demand)),
    }
    return (*loaders, split_lengths)


def build_model(
    args: argparse.Namespace,
    adj_sparse: torch.Tensor,
    quantiles: list[float],
    horizons: list[int],
    device: torch.device,
) -> tuple[torch.nn.Module, str]:
    if args.load_method:
        local_context = {
            "args": args,
            "adj_sparse": adj_sparse,
            "device": device,
            "horizons": horizons,
            "models": models,
            "quantiles": quantiles,
        }
        return eval(args.load_method, globals(), local_context), args.load_method

    if args.architecture == "pag_informer":
        model = models.PAGInformerQuantile(
            a_sparse=adj_sparse,
            seq=args.seq_len,
            hidden_dim=args.hidden_dim,
            quantiles=quantiles,
            horizons=horizons,
            train_gat_heads=not args.freeze_gat_heads,
            transformer_batch_first=args.transformer_batch_first,
        ).to(device)
        load_method = (
            "models.PAGInformerQuantile("
            f"a_sparse=adj_sparse, seq={args.seq_len}, hidden_dim={args.hidden_dim}, "
            "quantiles=quantiles, horizons=horizons, "
            f"train_gat_heads={not args.freeze_gat_heads}, "
            f"transformer_batch_first={args.transformer_batch_first}"
            ").to(device)"
        )
    elif args.architecture == "temporal_graph_quantile":
        model = models.TemporalGraphQuantile(
            a_sparse=adj_sparse,
            seq=args.seq_len,
            hidden_dim=args.hidden_dim,
            quantiles=quantiles,
            horizons=horizons,
            temporal_layers=args.temporal_layers,
            graph_layers=args.graph_layers,
            dropout=args.dropout,
        ).to(device)
        load_method = (
            "models.TemporalGraphQuantile("
            f"a_sparse=adj_sparse, seq={args.seq_len}, hidden_dim={args.hidden_dim}, "
            "quantiles=quantiles, horizons=horizons, "
            f"temporal_layers={args.temporal_layers}, graph_layers={args.graph_layers}, "
            f"dropout={args.dropout}"
            ").to(device)"
        )
    elif args.architecture == "lstm_quantile":
        model = models.LSTMMultiHorizonQuantile(
            a_sparse=adj_sparse,
            seq=args.seq_len,
            hidden_dim=args.hidden_dim,
            quantiles=quantiles,
            horizons=horizons,
            temporal_layers=args.temporal_layers,
            dropout=args.dropout,
        ).to(device)
        load_method = (
            "models.LSTMMultiHorizonQuantile("
            f"a_sparse=adj_sparse, seq={args.seq_len}, hidden_dim={args.hidden_dim}, "
            "quantiles=quantiles, horizons=horizons, "
            f"temporal_layers={args.temporal_layers}, dropout={args.dropout}"
            ").to(device)"
        )
    elif args.architecture == "patchtst_quantile":
        model = models.PatchTSTQuantile(
            a_sparse=adj_sparse,
            seq=args.seq_len,
            hidden_dim=args.hidden_dim,
            quantiles=quantiles,
            horizons=horizons,
            patch_len=args.patch_len,
            patch_stride=args.patch_stride,
            transformer_layers=args.transformer_layers,
            attention_heads=args.attention_heads,
            ff_dim=args.ff_dim or None,
            dropout=args.dropout,
        ).to(device)
        load_method = (
            "models.PatchTSTQuantile("
            f"a_sparse=adj_sparse, seq={args.seq_len}, hidden_dim={args.hidden_dim}, "
            "quantiles=quantiles, horizons=horizons, "
            f"patch_len={args.patch_len}, patch_stride={args.patch_stride}, "
            f"transformer_layers={args.transformer_layers}, attention_heads={args.attention_heads}, "
            f"ff_dim={args.ff_dim or None}, dropout={args.dropout}"
            ").to(device)"
        )
    elif args.architecture == "nlinear_quantile":
        model = models.NLinearQuantile(
            a_sparse=adj_sparse,
            seq=args.seq_len,
            hidden_dim=args.hidden_dim,
            quantiles=quantiles,
            horizons=horizons,
            dropout=args.dropout,
        ).to(device)
        load_method = (
            "models.NLinearQuantile("
            f"a_sparse=adj_sparse, seq={args.seq_len}, hidden_dim={args.hidden_dim}, "
            f"quantiles=quantiles, horizons=horizons, dropout={args.dropout}"
            ").to(device)"
        )
    elif args.architecture == "dlinear_quantile":
        model = models.DLinearQuantile(
            a_sparse=adj_sparse,
            seq=args.seq_len,
            hidden_dim=args.hidden_dim,
            quantiles=quantiles,
            horizons=horizons,
            moving_avg=args.moving_avg,
            dropout=args.dropout,
        ).to(device)
        load_method = (
            "models.DLinearQuantile("
            f"a_sparse=adj_sparse, seq={args.seq_len}, hidden_dim={args.hidden_dim}, "
            "quantiles=quantiles, horizons=horizons, "
            f"moving_avg={args.moving_avg}, dropout={args.dropout}"
            ").to(device)"
        )
    elif args.architecture == "multi_scale_temporal_graph_quantile":
        model = models.MultiScaleTemporalGraphQuantile(
            a_sparse=adj_sparse,
            seq=args.seq_len,
            short_seq=args.short_seq,
            hidden_dim=args.hidden_dim,
            quantiles=quantiles,
            horizons=horizons,
            temporal_layers=args.temporal_layers,
            graph_layers=args.graph_layers,
            dropout=args.dropout,
        ).to(device)
        load_method = (
            "models.MultiScaleTemporalGraphQuantile("
            f"a_sparse=adj_sparse, seq={args.seq_len}, short_seq={args.short_seq}, "
            f"hidden_dim={args.hidden_dim}, quantiles=quantiles, horizons=horizons, "
            f"temporal_layers={args.temporal_layers}, graph_layers={args.graph_layers}, "
            f"dropout={args.dropout}"
            ").to(device)"
        )
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")

    return model, load_method


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    loss_fn: QuantileLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    checkpoint_path: Path,
    epochs: int,
    max_train_batches: int,
    max_valid_batches: int,
) -> dict[str, float | int]:
    best_valid_loss = float("inf")
    best_epoch = -1

    for epoch in tqdm(range(epochs), desc="Stage1 multi-horizon training"):
        model.train()
        train_losses = []
        for _, (demand, price, label) in limited_loader(train_loader, max_train_batches):
            demand, price, label = demand.to(device), price.to(device), label.to(device)
            optimizer.zero_grad()
            pred = model(demand, price)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for _, (demand, price, label) in limited_loader(valid_loader, max_valid_batches):
                demand, price, label = demand.to(device), price.to(device), label.to(device)
                pred = model(demand, price)
                valid_losses.append(float(loss_fn(pred, label).item()))

        valid_loss = float(np.mean(valid_losses)) if valid_losses else float("inf")
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch + 1
            torch.save(model, checkpoint_path)

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        print(f"epoch={epoch + 1} train_loss={train_loss:.8f} valid_loss={valid_loss:.8f}")

    return {"best_valid_loss": best_valid_loss, "best_epoch": best_epoch}


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    cap: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    output_dir: Path,
    model_name: str,
    max_test_batches: int,
    save_arrays: bool,
) -> dict[str, object]:
    model.eval()
    predict_list = []
    label_list = []

    with torch.no_grad():
        for _, (demand, price, label) in limited_loader(test_loader, max_test_batches):
            demand, price = demand.to(device), price.to(device)
            pred_q = model(demand, price)
            if pred_q.ndim == 3 and len(horizons) == 1:
                pred_q = pred_q.unsqueeze(2)
            assert_quantile_tensor_shape(
                pred_q,
                n_horizons=len(horizons),
                n_quantiles=len(quantiles),
                context="stage1 model output",
            )
            assert_target_tensor_shape(label, n_horizons=len(horizons), context="stage1 labels")
            predict_list.append(pred_q.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

    predict_q_normalized = np.concatenate(predict_list, axis=0)
    label_normalized = np.concatenate(label_list, axis=0)

    cap_pred = cap.reshape(1, -1, 1, 1)
    cap_label = cap.reshape(1, -1, 1)
    predict_q = predict_q_normalized * cap_pred
    labels = label_normalized * cap_label
    assert_monotonic_quantiles(predict_q, context="stage1 saved predict_quantiles")

    output_dir.mkdir(parents=True, exist_ok=True)

    q50_idx = quantile_index(quantiles, 0.5)
    point_pred = predict_q[..., q50_idx]

    array_files = {}
    if save_arrays:
        np.save(output_dir / "predict_quantiles.npy", predict_q)
        np.save(output_dir / "label_list.npy", labels)
        np.save(output_dir / "predict_point_q50.npy", point_pred)
        array_files = {
            "predict_quantiles_file": "predict_quantiles.npy",
            "label_file": "label_list.npy",
            "predict_point_q50_file": "predict_point_q50.npy",
        }

    point_rows = []
    crossing_rows = []
    for h_idx, horizon in enumerate(horizons):
        metrics = point_metrics(point_pred[:, :, h_idx], labels[:, :, h_idx])
        point_rows.append({"horizon": horizon, **metrics})
        crossing_rows.append({"horizon": horizon, **quantile_crossing_metrics(predict_q[:, :, h_idx, :])})

    point_df = pd.DataFrame(point_rows, columns=["horizon", *POINT_METRIC_COLUMNS])
    point_df.to_csv(output_dir / "point_metrics_by_horizon.csv", index=False)

    crossing_df = pd.DataFrame(crossing_rows)
    crossing_df.to_csv(output_dir / "quantile_crossing_metrics_by_horizon.csv", index=False)

    interval_rows = []
    for delta in [0.1, 0.2, 0.4]:
        li = quantile_index(quantiles, delta / 2.0)
        ui = quantile_index(quantiles, 1.0 - delta / 2.0)
        for h_idx, horizon in enumerate(horizons):
            metrics = interval_metrics(
                labels[:, :, h_idx],
                predict_q[:, :, h_idx, li],
                predict_q[:, :, h_idx, ui],
                point_pred=point_pred[:, :, h_idx],
                delta=delta,
            )
            interval_rows.append({
                "horizon": horizon,
                "delta": delta,
                "lower_quantile": quantiles[li],
                "upper_quantile": quantiles[ui],
                **metrics,
            })

    interval_df = pd.DataFrame(interval_rows)
    interval_df.to_csv(output_dir / "raw_interval_metrics_by_horizon.csv", index=False)

    return {
        "output_dir": str(output_dir),
        "predict_quantiles_shape": list(predict_q.shape),
        "label_shape": list(labels.shape),
        "point_metrics_file": "point_metrics_by_horizon.csv",
        "raw_interval_metrics_file": "raw_interval_metrics_by_horizon.csv",
        "crossing_metrics_file": "quantile_crossing_metrics_by_horizon.csv",
        "arrays_saved": save_arrays,
        "array_files": array_files,
        "model_name": model_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate the journal Stage 1 raw multi-horizon baseline.")
    parser.add_argument("--data-dir", default="data/datasets/ST_EVCDP_v2_canonical/")
    parser.add_argument("--output-dir", default="journal_results/shenzhen_multihorizon/raw")
    parser.add_argument("--model-name", default="journal_dura_pag_informer_quantile_multihorizon_raw")
    parser.add_argument("--load-method", default="")
    parser.add_argument(
        "--architecture",
        choices=[
            "pag_informer",
            "temporal_graph_quantile",
            "lstm_quantile",
            "patchtst_quantile",
            "nlinear_quantile",
            "dlinear_quantile",
            "multi_scale_temporal_graph_quantile",
        ],
        default="pag_informer",
    )
    parser.add_argument("--warm-start-checkpoint", default="")
    parser.add_argument("--use-cuda", type=parse_bool, default=True)
    parser.add_argument("--train", type=parse_bool, default=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--short-seq", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--temporal-layers", type=int, default=1)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--patch-len", type=int, default=8)
    parser.add_argument("--patch-stride", type=int, default=4)
    parser.add_argument("--moving-avg", type=int, default=7)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_JOURNAL_HORIZONS))
    parser.add_argument("--horizon-loss-weights", default="")
    parser.add_argument("--quantiles", default=",".join(f"{q:.12g}" for q in DEFAULT_JOURNAL_QUANTILES))
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-valid-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    parser.add_argument("--freeze-gat-heads", type=parse_bool, default=False)
    parser.add_argument("--transformer-batch-first", type=parse_bool, default=True)
    parser.add_argument("--save-arrays", type=parse_bool, default=True)
    args = parser.parse_args()

    horizons = parse_int_list(args.horizons)
    horizon_loss_weights = parse_float_list(args.horizon_loss_weights) if args.horizon_loss_weights else []
    if horizon_loss_weights and len(horizon_loss_weights) != len(horizons):
        raise ValueError(
            "--horizon-loss-weights must have the same length as --horizons: "
            f"got {len(horizon_loss_weights)} weights for {len(horizons)} horizons"
        )
    quantiles = parse_float_list(args.quantiles)
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / f"{args.model_name}_checkpoint.pt"

    fn.set_seed(seed=2023, flag=True)
    device = torch.device("cuda:0" if args.use_cuda and torch.cuda.is_available() else "cpu")

    occ, duration, price_raw, distance, cap = fn.read_dataset_v2(args.data_dir)
    adj_sparse = distance.to_sparse().to(device)
    input_series = duration if "dura" in args.model_name else occ

    split_rates = {"train": 0.7, "valid": 0.1, "calib": 0.1, "test": 0.1}
    train_loader, valid_loader, test_loader, split_lengths = build_dataloaders(
        input_series=input_series,
        price_raw=price_raw,
        seq_len=args.seq_len,
        horizons=horizons,
        batch_size=args.batch_size,
        split_rates=split_rates,
        device=device,
    )

    model, load_method = build_model(args, adj_sparse, quantiles, horizons, device)
    warm_start_report = warm_start_model(model, args.warm_start_checkpoint, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = QuantileLoss(
        quantiles=quantiles,
        horizon_weights=horizon_loss_weights or None,
    ).to(device)

    train_report: dict[str, float | int | str]
    if args.train:
        output_dir.mkdir(parents=True, exist_ok=True)
        train_report = train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            checkpoint_path=checkpoint_path,
            epochs=args.epochs,
            max_train_batches=args.max_train_batches,
            max_valid_batches=args.max_valid_batches,
        )
    else:
        train_report = {"status": "training skipped"}

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {checkpoint_path}")

    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    eval_report = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        cap=cap,
        quantiles=quantiles,
        horizons=horizons,
        output_dir=output_dir,
        model_name=args.model_name,
        max_test_batches=args.max_test_batches,
        save_arrays=args.save_arrays,
    )

    metadata = {
        "stage": "1.1",
        "goal": "Direct multi-horizon raw quantile forecasting baseline",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "branch": git_value(["branch", "--show-current"]),
        "commit": git_value(["rev-parse", "--short", "HEAD"]),
        "data_dir": args.data_dir,
        "output_dir": str(output_dir),
        "model_name": args.model_name,
        "architecture": args.architecture,
        "load_method": load_method,
        "seq_len": args.seq_len,
        "short_seq": args.short_seq,
        "hidden_dim": args.hidden_dim,
        "temporal_layers": args.temporal_layers,
        "graph_layers": args.graph_layers,
        "patch_len": args.patch_len,
        "patch_stride": args.patch_stride,
        "moving_avg": args.moving_avg,
        "transformer_layers": args.transformer_layers,
        "attention_heads": args.attention_heads,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "horizons": horizons,
        "horizon_loss_weights": horizon_loss_weights,
        "quantiles": quantiles,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "freeze_gat_heads": args.freeze_gat_heads,
        "transformer_batch_first": args.transformer_batch_first,
        "save_arrays": args.save_arrays,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "split_rates": split_rates,
        "split_lengths": split_lengths,
        "warm_start": warm_start_report,
        "train_report": train_report,
        "eval_report": eval_report,
        "smoke_limits": {
            "max_train_batches": args.max_train_batches,
            "max_valid_batches": args.max_valid_batches,
            "max_test_batches": args.max_test_batches,
        },
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
