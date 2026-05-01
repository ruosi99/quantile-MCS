from __future__ import annotations

import json
from pathlib import Path

from utils.model_training.journal_contracts import (
    DEFAULT_JOURNAL_QUANTILES,
    build_horizon_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE0_DIR = REPO_ROOT / "docs" / "journal_stage0"
DATA_DIR = "data/datasets/ST_EVCDP_v2_canonical/"
MODEL_NAME = "dura_pag_informer_quantile_on_pretrain"
LOAD_METHOD = "models.PAGInformerQuantile(a_sparse=adj_sparse, seq=seq_len, quantiles=quantiles).to(device)"
LEGACY_H1_QUANTILES = [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]
H1_QUANTILES = LEGACY_H1_QUANTILES


def _quantile_arg(quantiles: list[float]) -> str:
    return ",".join(f"{q:.12g}" for q in quantiles)


def build_h1_run_spec() -> dict:
    config = build_horizon_config(horizons=[1], quantiles=H1_QUANTILES)
    return {
        "stage": "0.3",
        "goal": "H=[1] parity gate for the journal path",
        "dataset": "ST_EVCDP_v2_canonical",
        "legacy_entry": "train.py",
        "journal_mode": "single-horizon parity",
        "output_root": "journal_results/stage0_h1_parity",
        "train_args": {
            "data_dir": DATA_DIR,
            "model_name": MODEL_NAME,
            "load_method": LOAD_METHOD,
            "is_train": False,
            "is_pre_train": True,
            "use_cuda": True,
            "batch_size": 8,
            "quantiles": config["quantiles"],
        },
        "parity_targets": {
            "legacy_reference_dir": "canonical_main_results",
            "journal_candidate_dir": "journal_results/stage0_h1_parity",
            "reference_point_metrics_csv": "canonical_point_q50.csv",
            "candidate_point_metrics_csv": f"{MODEL_NAME}_1bs8_point_q50.csv",
            "predict_quantiles_npy": "predict_quantiles.npy",
            "label_npy": "label_list.npy",
        },
    }


def build_multihorizon_stub_spec() -> dict:
    config = build_horizon_config()
    return {
        "stage": "1.1",
        "goal": "Direct multi-horizon raw forecasting baseline",
        "status": "stage1 raw path available; calibration remains pending",
        "dataset": "ST_EVCDP_v2_canonical",
        "planned_output_root": "journal_results/shenzhen_multihorizon/raw",
        "horizon_config": config,
        "expected_training_entry": "scripts/journal/run_stage1_multihorizon_raw.sh",
        "expected_python_entry": "scripts/journal/train_multihorizon_raw.py",
    }


def write_stage0_run_specs() -> list[Path]:
    STAGE0_DIR.mkdir(parents=True, exist_ok=True)
    h1_path = STAGE0_DIR / "journal_h1_run_spec.json"
    multi_path = STAGE0_DIR / "journal_multihorizon_stub_run_spec.json"
    h1_path.write_text(json.dumps(build_h1_run_spec(), indent=2), encoding="utf-8")
    multi_path.write_text(json.dumps(build_multihorizon_stub_spec(), indent=2), encoding="utf-8")
    return [h1_path, multi_path]


def build_h1_train_command() -> str:
    spec = build_h1_run_spec()
    args = spec["train_args"]
    quantile_arg = _quantile_arg(args["quantiles"])
    return (
        "python train.py "
        f'--data_dir "{args["data_dir"]}" '
        f'--model_name {args["model_name"]} '
        f'--load_method "{args["load_method"]}" '
        f'--is_train {str(args["is_train"])} '
        f'--is_pre_train {str(args["is_pre_train"])} '
        f'--use_cuda {str(args["use_cuda"])} '
        f'--batch_size {args["batch_size"]} '
        f'--quantiles {quantile_arg}'
    )
