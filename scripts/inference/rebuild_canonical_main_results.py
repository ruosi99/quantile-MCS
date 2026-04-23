import argparse
import os
import shutil

import numpy as np
import pandas as pd


QUANTILES = [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]
DELTA_LIST = [0.1, 0.2, 0.4]


def metrics_new(test_pre, test_real):
    eps = 0.01
    valid_indices = test_real > eps
    valid_real = test_real[valid_indices]
    valid_pred = test_pre[valid_indices]
    mape = np.mean(np.abs((valid_real - valid_pred) / valid_real)) * 100
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


def interval_metrics(y_true, lower, upper, point_pred, delta):
    covered = (y_true >= lower) & (y_true <= upper)
    picp = covered.mean()
    mpiw = (upper - lower).mean()

    alpha = delta / 2.0
    diff_l = lower - y_true
    loss_l = np.maximum(alpha * diff_l, (alpha - 1) * diff_l)
    diff_u = upper - y_true
    loss_u = np.maximum((1 - alpha) * diff_u, -alpha * diff_u)
    diff_m = point_pred - y_true
    loss_m = np.maximum(0.5 * diff_m, -0.5 * diff_m)
    wis = (loss_l + loss_u + loss_m).mean()
    return {"PICP": float(picp), "MPIW": float(mpiw), "WIS": float(wis)}


def main():
    parser = argparse.ArgumentParser(description="Rebuild isolated canonical main-model results from saved canonical artifacts.")
    parser.add_argument(
        "--source_result_dir",
        type=str,
        default="quantile_model/dura_pag_informer_quantile_on_pretrain_results",
        help="Existing trusted canonical artifact folder.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/datasets/ST_EVCDP_v2_canonical",
        help="Canonical dataset directory.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="canonical_main_results",
        help="Isolated output directory for rebuilt canonical outputs.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    arrays_to_copy = [
        "predict_quantiles.npy",
        "predict_point_q50.npy",
        "label_list.npy",
        "cqr_L_delta0.1.npy",
        "cqr_U_delta0.1.npy",
        "cqr_s_hat_delta0.1.npy",
        "cqr_L_delta0.2.npy",
        "cqr_U_delta0.2.npy",
        "cqr_s_hat_delta0.2.npy",
        "cqr_L_delta0.4.npy",
        "cqr_U_delta0.4.npy",
        "cqr_s_hat_delta0.4.npy",
    ]
    for name in arrays_to_copy:
        shutil.copy2(os.path.join(args.source_result_dir, name), os.path.join(args.out_dir, name))

    labels = np.load(os.path.join(args.out_dir, "label_list.npy"))
    point_pred = np.load(os.path.join(args.out_dir, "predict_point_q50.npy"))
    pred_quantiles = np.load(os.path.join(args.out_dir, "predict_quantiles.npy"))

    point_df = pd.DataFrame([metrics_new(point_pred, labels)])
    point_df.to_csv(os.path.join(args.out_dir, "canonical_point_q50.csv"), index=False)

    raw_rows = []
    cal_rows = []
    for delta in DELTA_LIST:
        li = QUANTILES.index(delta / 2.0)
        ui = QUANTILES.index(1.0 - delta / 2.0)

        raw_lower = pred_quantiles[:, :, li]
        raw_upper = pred_quantiles[:, :, ui]
        raw_metrics = interval_metrics(labels, raw_lower, raw_upper, point_pred, delta)
        raw_metrics["delta"] = delta
        raw_rows.append(raw_metrics)
        pd.DataFrame([raw_metrics]).to_csv(os.path.join(args.out_dir, f"raw_interval_delta{delta:.1f}.csv"), index=False)

        cal_lower = np.load(os.path.join(args.out_dir, f"cqr_L_delta{delta:.1f}.npy"))
        cal_upper = np.load(os.path.join(args.out_dir, f"cqr_U_delta{delta:.1f}.npy"))
        cal_metrics = interval_metrics(labels, cal_lower, cal_upper, point_pred, delta)
        cal_metrics["delta"] = delta
        cal_rows.append(cal_metrics)
        pd.DataFrame([cal_metrics]).to_csv(
            os.path.join(args.out_dir, f"calibrated_interval_delta{delta:.1f}.csv"),
            index=False,
        )

    pd.DataFrame(raw_rows).to_csv(os.path.join(args.out_dir, "raw_interval_summary.csv"), index=False)
    pd.DataFrame(cal_rows).to_csv(os.path.join(args.out_dir, "calibrated_interval_summary.csv"), index=False)

    duration = pd.read_csv(os.path.join(args.data_path, "duration.csv"), index_col=0)
    duration.index = pd.to_datetime(duration.index)
    metadata = pd.DataFrame(
        [
            {
                "dataset_path": args.data_path,
                "source_result_dir": args.source_result_dir,
                "n_rows": len(duration),
                "n_stations": duration.shape[1],
                "split_rule": "train=0.7, valid=0.1, calib=0.1, test=0.1, seq_len=24",
                "saved_label_shape": "x".join(str(x) for x in labels.shape),
            }
        ]
    )
    metadata.to_csv(os.path.join(args.out_dir, "canonical_metadata.csv"), index=False)

    print(f"Rebuilt canonical main results in: {args.out_dir}")
    print("Point metrics:")
    print(point_df.to_string(index=False))
    print("Raw interval summary:")
    print(pd.DataFrame(raw_rows).to_string(index=False))
    print("Calibrated interval summary:")
    print(pd.DataFrame(cal_rows).to_string(index=False))


if __name__ == "__main__":
    main()
