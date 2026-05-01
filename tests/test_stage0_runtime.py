import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.journal.check_h1_parity import compare_arrays, compare_metrics  # noqa: E402
from scripts.journal.stage0_runtime import (  # noqa: E402
    build_h1_run_spec,
    build_multihorizon_stub_spec,
)


def test_h1_run_spec_uses_single_horizon():
    spec = build_h1_run_spec()
    assert spec["train_args"]["quantiles"][0] == 0.05
    assert spec["goal"] == "H=[1] parity gate for the journal path"
    assert spec["journal_mode"] == "single-horizon parity"


def test_multihorizon_stub_marks_placeholder_status():
    spec = build_multihorizon_stub_spec()
    assert spec["status"] == "stage1 raw path available; calibration remains pending"
    assert spec["horizon_config"]["horizons"] == [1, 3, 6, 12, 24]
    assert spec["expected_training_entry"] == "scripts/journal/run_stage1_multihorizon_raw.sh"


def test_compare_arrays_reports_zero_error_for_identical_inputs():
    arr = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    report = compare_arrays(arr, arr.copy())
    assert report["mae"] == 0.0
    assert report["max_abs_error"] == 0.0


def test_compare_metrics_reports_column_differences(tmp_path: Path):
    ref = pd.DataFrame([{"MAE": 1.0, "RMSE": 2.0}])
    cand = pd.DataFrame([{"MAE": 1.2, "RMSE": 2.5}])
    ref_path = tmp_path / "ref.csv"
    cand_path = tmp_path / "cand.csv"
    ref.to_csv(ref_path, index=False)
    cand.to_csv(cand_path, index=False)

    report = compare_metrics(ref_path, cand_path)
    assert np.isclose(report["MAE"]["abs_diff"], 0.2)
    assert np.isclose(report["RMSE"]["abs_diff"], 0.5)


def test_run_spec_is_json_serializable():
    payload = build_h1_run_spec()
    json.loads(json.dumps(payload))
