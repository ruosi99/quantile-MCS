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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import utils.model_training.models as models
import utils.model_training.training_utils as fn
from scripts.journal.train_multihorizon_raw import (
    load_matching_state_dict,
    point_metrics,
    quantile_index,
    quantile_crossing_metrics,
)
from utils.model_training.conformal import interval_metrics
from utils.model_training.loss_functions import QuantileLoss


DEFAULT_PRETRAIN_QUANTILES = [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]
POINT_METRIC_COLUMNS = ["MSE", "RMSE", "MAPE", "RAE", "MAE", "R2", "MedAE", "EVS"]


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y"}


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


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


def row_normalize_adjacency(adj_dense: torch.Tensor) -> torch.Tensor:
    adj = adj_dense.float().clone()
    adj = torch.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
    adj = torch.clamp(adj, min=0.0)
    adj.fill_diagonal_(0.0)
    row_sum = adj.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return adj / row_sum


def sample_physics_shift(
    batch_size: int,
    node_count: int,
    adj_norm: torch.Tensor,
    laws: torch.Tensor,
    perturb_prop: float,
    perturb_scale: float,
    graph_layers: int,
    graph_decay: float,
    clamp: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active = torch.rand(batch_size, node_count, device=device) < perturb_prop
    price_change = torch.randn(batch_size, node_count, device=device) * perturb_scale
    price_change = torch.where(active, price_change, torch.zeros_like(price_change))
    price_change = torch.clamp(price_change, -clamp, clamp)

    law_idx = torch.randint(low=0, high=laws.numel(), size=(batch_size,), device=device)
    law = laws[law_idx].view(batch_size, 1)
    direct_shift = law * price_change

    expected_shift = direct_shift
    hop_shift = direct_shift
    for layer_idx in range(graph_layers):
        hop_shift = torch.matmul(hop_shift, adj_norm.t())
        expected_shift = expected_shift + (graph_decay ** (layer_idx + 1)) * hop_shift

    expected_shift = torch.clamp(expected_shift, -0.8, 0.8)
    return price_change, expected_shift, law.squeeze(1)


def physics_consistency_loss(
    model: torch.nn.Module,
    demand: torch.Tensor,
    price: torch.Tensor,
    base_pred: torch.Tensor,
    adj_norm: torch.Tensor,
    laws: torch.Tensor,
    perturb_prop: float,
    perturb_scale: float,
    graph_layers: int,
    graph_decay: float,
    perturb_clamp: float,
) -> torch.Tensor:
    batch_size, node_count, _ = price.shape
    price_change, expected_shift, _ = sample_physics_shift(
        batch_size=batch_size,
        node_count=node_count,
        adj_norm=adj_norm,
        laws=laws,
        perturb_prop=perturb_prop,
        perturb_scale=perturb_scale,
        graph_layers=graph_layers,
        graph_decay=graph_decay,
        clamp=perturb_clamp,
        device=price.device,
    )
    perturbed_price = price * (1.0 + price_change.unsqueeze(-1)).clamp_min(0.0)
    perturbed_pred = model(demand, perturbed_price)

    target_pred = base_pred.detach() * (1.0 + expected_shift.unsqueeze(-1)).clamp_min(0.05)
    return F.smooth_l1_loss(perturbed_pred, target_pred)


def build_single_horizon_loaders(
    input_series: np.ndarray,
    price_raw: np.ndarray,
    seq_len: int,
    pred_len: int,
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

    train_dataset = fn.CreateDataset(train_demand, train_price, seq_len, pred_len, device)
    valid_dataset = fn.CreateDataset(valid_demand, valid_price, seq_len, pred_len, device)
    test_dataset = fn.CreateDataset(test_demand, test_price, seq_len, pred_len, device)

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
        "calib_length_reserved": int(len(calib_demand)),
    }
    return (*loaders, split_lengths)


def limited_loader(loader: DataLoader, max_batches: int):
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        yield batch_idx, batch


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    loss_fn: QuantileLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    checkpoint_path: Path,
    epochs: int,
    physics_weight: float,
    adj_norm: torch.Tensor,
    laws: torch.Tensor,
    perturb_prop: float,
    perturb_scale: float,
    graph_layers: int,
    graph_decay: float,
    perturb_clamp: float,
    max_train_batches: int,
    max_valid_batches: int,
) -> dict[str, float | int]:
    best_valid_loss = float("inf")
    best_epoch = -1
    best_train_loss = float("nan")
    best_supervised_loss = float("nan")
    best_physics_loss = float("nan")

    for epoch in tqdm(range(epochs), desc="Fixed physics pretraining"):
        model.train()
        train_losses = []
        supervised_losses = []
        physics_losses = []

        for _, (demand, price, label) in limited_loader(train_loader, max_train_batches):
            demand, price, label = demand.to(device), price.to(device), label.to(device)
            optimizer.zero_grad()
            pred = model(demand, price)
            supervised_loss = loss_fn(pred, label)
            if physics_weight > 0.0:
                consistency_loss = physics_consistency_loss(
                    model=model,
                    demand=demand,
                    price=price,
                    base_pred=pred,
                    adj_norm=adj_norm,
                    laws=laws,
                    perturb_prop=perturb_prop,
                    perturb_scale=perturb_scale,
                    graph_layers=graph_layers,
                    graph_decay=graph_decay,
                    perturb_clamp=perturb_clamp,
                )
            else:
                consistency_loss = pred.new_tensor(0.0)

            loss = supervised_loss + physics_weight * consistency_loss
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))
            supervised_losses.append(float(supervised_loss.item()))
            physics_losses.append(float(consistency_loss.item()))

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for _, (demand, price, label) in limited_loader(valid_loader, max_valid_batches):
                demand, price, label = demand.to(device), price.to(device), label.to(device)
                pred = model(demand, price)
                valid_losses.append(float(loss_fn(pred, label).item()))

        valid_loss = float(np.mean(valid_losses)) if valid_losses else float("inf")
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        supervised_mean = float(np.mean(supervised_losses)) if supervised_losses else float("nan")
        physics_mean = float(np.mean(physics_losses)) if physics_losses else float("nan")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch + 1
            best_train_loss = train_loss
            best_supervised_loss = supervised_mean
            best_physics_loss = physics_mean
            torch.save(model, checkpoint_path)

        print(
            f"epoch={epoch + 1} train_loss={train_loss:.8f} "
            f"supervised_loss={supervised_mean:.8f} "
            f"physics_loss={physics_mean:.8f} "
            f"valid_loss={valid_loss:.8f}"
        )

    return {
        "best_valid_loss": best_valid_loss,
        "best_epoch": best_epoch,
        "best_train_loss": best_train_loss,
        "best_supervised_loss": best_supervised_loss,
        "best_physics_loss": best_physics_loss,
    }


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    cap: np.ndarray,
    quantiles: list[float],
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
            predict_list.append(pred_q.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

    predict_q_normalized = np.concatenate(predict_list, axis=0)
    label_normalized = np.concatenate(label_list, axis=0)

    cap_pred = cap.reshape(1, -1, 1)
    cap_label = cap.reshape(1, -1)
    predict_q = predict_q_normalized * cap_pred
    labels = label_normalized * cap_label

    output_dir.mkdir(parents=True, exist_ok=True)
    q50_idx = quantile_index(quantiles, 0.5)
    point_pred = predict_q[:, :, q50_idx]

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

    point_df = pd.DataFrame(
        [{**point_metrics(point_pred, labels)}],
        columns=POINT_METRIC_COLUMNS,
    )
    point_df.to_csv(output_dir / "point_metrics_q50.csv", index=False)

    crossing_df = pd.DataFrame([quantile_crossing_metrics(predict_q)])
    crossing_df.to_csv(output_dir / "quantile_crossing_metrics.csv", index=False)

    interval_rows = []
    for delta in [0.1, 0.2, 0.4]:
        li = quantile_index(quantiles, delta / 2.0)
        ui = quantile_index(quantiles, 1.0 - delta / 2.0)
        metrics = interval_metrics(
            labels,
            predict_q[:, :, li],
            predict_q[:, :, ui],
            point_pred=point_pred,
            delta=delta,
        )
        interval_rows.append({
            "delta": delta,
            "lower_quantile": quantiles[li],
            "upper_quantile": quantiles[ui],
            **metrics,
        })

    interval_df = pd.DataFrame(interval_rows)
    interval_df.to_csv(output_dir / "raw_interval_metrics.csv", index=False)

    return {
        "output_dir": str(output_dir),
        "predict_quantiles_shape": list(predict_q.shape),
        "label_shape": list(labels.shape),
        "point_metrics_file": "point_metrics_q50.csv",
        "raw_interval_metrics_file": "raw_interval_metrics.csv",
        "crossing_metrics_file": "quantile_crossing_metrics.csv",
        "arrays_saved": save_arrays,
        "array_files": array_files,
        "model_name": model_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed-architecture H=1 quantile pretraining with physics consistency regularization."
    )
    parser.add_argument("--data-dir", default="data/datasets/ST_EVCDP_v2_canonical/")
    parser.add_argument("--output-dir", default="journal_results/pretrain/fixed_physics")
    parser.add_argument("--model-name", default="journal_dura_pag_informer_quantile_fixed_physics_pretrain")
    parser.add_argument("--warm-start-checkpoint", default="")
    parser.add_argument("--use-cuda", type=parse_bool, default=True)
    parser.add_argument("--train", type=parse_bool, default=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--pred-len", type=int, default=1)
    parser.add_argument("--quantiles", default=",".join(f"{q:.12g}" for q in DEFAULT_PRETRAIN_QUANTILES))
    parser.add_argument("--physics-weight", type=float, default=0.05)
    parser.add_argument("--law-list", default="-1.48,-0.74")
    parser.add_argument("--perturb-prop", type=float, default=0.4)
    parser.add_argument("--perturb-scale", type=float, default=0.2)
    parser.add_argument("--perturb-clamp", type=float, default=0.5)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--graph-decay", type=float, default=0.5)
    parser.add_argument("--train-gat-heads", type=parse_bool, default=True)
    parser.add_argument("--transformer-batch-first", type=parse_bool, default=True)
    parser.add_argument("--save-arrays", type=parse_bool, default=True)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-valid-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    args = parser.parse_args()

    quantiles = parse_float_list(args.quantiles)
    laws = parse_float_list(args.law_list)
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / f"{args.model_name}_checkpoint.pt"

    fn.set_seed(seed=2023, flag=True)
    device = torch.device("cuda:0" if args.use_cuda and torch.cuda.is_available() else "cpu")

    occ, duration, price_raw, distance, cap = fn.read_dataset_v2(args.data_dir)
    adj_sparse = distance.to_sparse().to(device)
    adj_norm = row_normalize_adjacency(distance.to(device))
    law_tensor = torch.tensor(laws, dtype=torch.float32, device=device)
    input_series = duration if "dura" in args.model_name else occ

    split_rates = {"train": 0.7, "valid": 0.1, "calib": 0.1, "test": 0.1}
    train_loader, valid_loader, test_loader, split_lengths = build_single_horizon_loaders(
        input_series=input_series,
        price_raw=price_raw,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        split_rates=split_rates,
        device=device,
    )

    model = models.PAGInformerQuantile(
        a_sparse=adj_sparse,
        seq=args.seq_len,
        pred_len=args.pred_len,
        quantiles=quantiles,
        horizons=[args.pred_len],
        train_gat_heads=args.train_gat_heads,
        transformer_batch_first=args.transformer_batch_first,
    ).to(device)
    warm_start_report = warm_start_model(model, args.warm_start_checkpoint, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = QuantileLoss(quantiles=quantiles).to(device)

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
            physics_weight=args.physics_weight,
            adj_norm=adj_norm,
            laws=law_tensor,
            perturb_prop=args.perturb_prop,
            perturb_scale=args.perturb_scale,
            graph_layers=args.graph_layers,
            graph_decay=args.graph_decay,
            perturb_clamp=args.perturb_clamp,
            max_train_batches=args.max_train_batches,
            max_valid_batches=args.max_valid_batches,
        )
    else:
        train_report = {"status": "training skipped"}

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Pretraining checkpoint not found: {checkpoint_path}")

    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    eval_report = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        cap=cap,
        quantiles=quantiles,
        output_dir=output_dir,
        model_name=args.model_name,
        max_test_batches=args.max_test_batches,
        save_arrays=args.save_arrays,
    )

    metadata = {
        "stage": "fixed_physics_pretrain",
        "goal": "Architecture-consistent H=1 quantile pretraining with physics consistency regularization",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "branch": git_value(["branch", "--show-current"]),
        "commit": git_value(["rev-parse", "--short", "HEAD"]),
        "data_dir": args.data_dir,
        "output_dir": str(output_dir),
        "model_name": args.model_name,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "quantiles": quantiles,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "physics": {
            "weight": args.physics_weight,
            "laws": laws,
            "perturb_prop": args.perturb_prop,
            "perturb_scale": args.perturb_scale,
            "perturb_clamp": args.perturb_clamp,
            "graph_layers": args.graph_layers,
            "graph_decay": args.graph_decay,
        },
        "architecture": {
            "train_gat_heads": args.train_gat_heads,
            "transformer_batch_first": args.transformer_batch_first,
        },
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
    (output_dir / "pretrain_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
