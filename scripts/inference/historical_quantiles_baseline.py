import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


QUANTILES = [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]
DELTA_LIST = [0.1, 0.2, 0.4]
RISK_SCENARIOS = [
    {"scenario": "Balanced", "c_over": 1.0, "c_under": 1.0, "recommended_strategy": "q0.50"},
    {"scenario": "Shortage-4x", "c_over": 1.0, "c_under": 4.0, "recommended_strategy": "q0.80"},
    {"scenario": "Shortage-9x", "c_over": 1.0, "c_under": 9.0, "recommended_strategy": "q0.90"},
    {"scenario": "Shortage-19x", "c_over": 1.0, "c_under": 19.0, "recommended_strategy": "q0.95"},
]
STRATEGY_TO_INDEX = {"q0.50": 3, "q0.80": 4, "q0.90": 5, "q0.95": 6}


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


def build_historical_quantiles(train_df, quantiles):
    hour_quantiles = {}
    for hour in range(24):
        hour_slice = train_df[train_df.index.hour == hour]
        hour_quantiles[hour] = np.quantile(hour_slice.to_numpy(), quantiles, axis=0).T
    return hour_quantiles


def make_test_predictions(test_target_df, hour_quantiles):
    preds = np.zeros((len(test_target_df), test_target_df.shape[1], len(QUANTILES)), dtype=np.float64)
    for idx, timestamp in enumerate(test_target_df.index):
        preds[idx] = hour_quantiles[int(timestamp.hour)]
    return preds


def compute_risk_table(pred_quantiles, labels):
    rows = []
    for scenario in RISK_SCENARIOS:
        scenario_costs = {}
        for strategy_name, q_idx in STRATEGY_TO_INDEX.items():
            decision = pred_quantiles[:, :, q_idx]
            over = np.maximum(decision - labels, 0.0)
            under = np.maximum(labels - decision, 0.0)
            cost = scenario["c_over"] * over + scenario["c_under"] * under
            total_cost = float(cost.sum())
            scenario_costs[strategy_name] = total_cost
            rows.append(
                {
                    "scenario": scenario["scenario"],
                    "c_over": scenario["c_over"],
                    "c_under": scenario["c_under"],
                    "recommended_strategy": scenario["recommended_strategy"],
                    "strategy": strategy_name,
                    "total_cost": total_cost,
                    "mean_cost": float(cost.mean()),
                }
            )

        q50_total = scenario_costs["q0.50"]
        best_total = min(scenario_costs.values())
        for row in rows[-len(STRATEGY_TO_INDEX):]:
            row["cost_vs_q50_pct"] = 100.0 * row["total_cost"] / q50_total
            row["improvement_vs_q50_pct"] = 100.0 * (q50_total - row["total_cost"]) / q50_total
            row["is_best"] = row["total_cost"] == best_total

    return pd.DataFrame(rows)


def plot_interval_example(labels, point_pred, lower, upper, timestamps, save_path):
    n_plot = min(120, len(timestamps))
    t = np.arange(n_plot)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.fill_between(t, lower[:n_plot, 0], upper[:n_plot, 0], color="#4c956c", alpha=0.25, label="90% interval")
    ax.plot(t, labels[:n_plot, 0], color="#1f4e79", linewidth=2.2, label="Ground truth")
    ax.plot(t, point_pred[:n_plot, 0], color="#d17a22", linestyle="--", linewidth=2.0, label="Historical q0.50")
    tick_idx = np.arange(0, n_plot, 12)
    tick_labels = [timestamps[i].strftime("%m-%d %H:%M") for i in tick_idx]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_title("Historical Quantiles Baseline: Example Test Forecast", fontsize=14, fontweight="bold")
    ax.set_xlabel("Test time step")
    ax.set_ylabel("Duration demand")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Hour-of-day historical quantiles baseline.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/datasets/ST_EVCDP_v2_canonical",
        help="Dataset directory containing duration.csv.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="canonical_baseline_results",
        help="Output directory for baseline artifacts.",
    )
    parser.add_argument("--seq_len", type=int, default=24, help="Sequence length used by the main model split.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    duration_df = pd.read_csv(os.path.join(args.data_path, "duration.csv"), index_col=0)
    duration_df.fillna(0, inplace=True)
    duration_df.index = pd.to_datetime(duration_df.index)

    n = len(duration_df)
    train_end = int(n * 0.7)
    valid_end = int(n * 0.8)
    calib_end = int(n * 0.9)

    train_df = duration_df.iloc[:train_end]
    test_df = duration_df.iloc[calib_end:]
    test_target_df = test_df.iloc[args.seq_len:]

    historical_quantiles = build_historical_quantiles(train_df, QUANTILES)
    pred_quantiles = make_test_predictions(test_target_df, historical_quantiles)
    labels = test_target_df.to_numpy(dtype=np.float64)
    point_pred = pred_quantiles[:, :, QUANTILES.index(0.5)]

    np.save(os.path.join(args.out_dir, "predict_quantiles.npy"), pred_quantiles)
    np.save(os.path.join(args.out_dir, "predict_point_q50.npy"), point_pred)
    np.save(os.path.join(args.out_dir, "label_list.npy"), labels)

    point_df = pd.DataFrame([metrics_new(point_pred, labels)])
    point_df.to_csv(os.path.join(args.out_dir, "historical_quantiles_point_q50.csv"), index=False)

    interval_rows = []
    for delta in DELTA_LIST:
        li = QUANTILES.index(delta / 2.0)
        ui = QUANTILES.index(1.0 - delta / 2.0)
        lower = pred_quantiles[:, :, li]
        upper = pred_quantiles[:, :, ui]
        np.save(os.path.join(args.out_dir, f"historical_L_delta{delta:.1f}.npy"), lower)
        np.save(os.path.join(args.out_dir, f"historical_U_delta{delta:.1f}.npy"), upper)
        metrics = interval_metrics(labels, lower, upper, point_pred, delta)
        metrics["delta"] = delta
        interval_rows.append(metrics)
        pd.DataFrame([metrics]).to_csv(
            os.path.join(args.out_dir, f"historical_quantiles_interval_delta{delta:.1f}.csv"),
            index=False,
        )

    interval_df = pd.DataFrame(interval_rows)
    interval_df.to_csv(os.path.join(args.out_dir, "historical_quantiles_interval_summary.csv"), index=False)

    risk_df = compute_risk_table(pred_quantiles, labels)
    risk_df.to_csv(os.path.join(args.out_dir, "historical_quantiles_risk_evaluation.csv"), index=False)

    quantile_matrix = np.stack([historical_quantiles[h] for h in range(24)], axis=0)
    np.save(os.path.join(args.out_dir, "hour_of_day_quantiles.npy"), quantile_matrix)

    plot_interval_example(
        labels,
        point_pred,
        pred_quantiles[:, :, 0],
        pred_quantiles[:, :, 6],
        test_target_df.index,
        os.path.join(args.out_dir, "historical_quantiles_example_forecast.png"),
    )

    meta = pd.DataFrame(
        [
            {
                "baseline_name": "hour_of_day_historical_quantiles",
                "task": "duration_forecasting",
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "test_target_rows": len(test_target_df),
                "n_stations": duration_df.shape[1],
                "seq_len_assumed": args.seq_len,
                "quantiles": ",".join(str(q) for q in QUANTILES),
            }
        ]
    )
    meta.to_csv(os.path.join(args.out_dir, "baseline_metadata.csv"), index=False)

    print(f"Saved results to: {args.out_dir}")
    print("Point metrics:")
    print(point_df.to_string(index=False))
    print("Interval summary:")
    print(interval_df.to_string(index=False))


if __name__ == "__main__":
    main()
