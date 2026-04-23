import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def infer_start_hour_from_current_dataset(data_path: str, seq_len: int = 24) -> int:
    duration_path = os.path.join(data_path, "duration.csv")
    time_index = pd.to_datetime(pd.read_csv(duration_path, index_col=0).index)
    n = len(time_index)
    calib_end = int(n * 0.9)
    first_target_idx = calib_end + seq_len
    if first_target_idx >= n:
        raise ValueError("Unable to infer first target hour from current dataset split.")
    return int(time_index[first_target_idx].hour)


def quantile_index(quantiles, q):
    for i, value in enumerate(quantiles):
        if abs(value - q) < 1e-9:
            return i
    raise ValueError(f"Quantile {q} not found in {quantiles}")


def build_hourly_diagnostic(result_dir: str, data_path: str, delta: float, seq_len: int) -> pd.DataFrame:
    labels = np.load(os.path.join(result_dir, "label_list.npy"))
    pred_quantiles = np.load(os.path.join(result_dir, "predict_quantiles.npy"))
    delta_tag = f"{delta:.1f}"
    lower_cal = np.load(os.path.join(result_dir, f"cqr_L_delta{delta_tag}.npy"))
    upper_cal = np.load(os.path.join(result_dir, f"cqr_U_delta{delta_tag}.npy"))

    quantiles = [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]
    li = quantile_index(quantiles, delta / 2.0)
    ui = quantile_index(quantiles, 1.0 - delta / 2.0)
    lower_raw = pred_quantiles[:, :, li]
    upper_raw = pred_quantiles[:, :, ui]

    start_hour = infer_start_hour_from_current_dataset(data_path, seq_len=seq_len)
    hour_labels = (np.arange(labels.shape[0]) + start_hour) % 24

    rows = []
    nominal = 1.0 - delta
    for hour in range(24):
        mask = hour_labels == hour
        raw_picp = float(((labels[mask] >= lower_raw[mask]) & (labels[mask] <= upper_raw[mask])).mean())
        cal_picp = float(((labels[mask] >= lower_cal[mask]) & (labels[mask] <= upper_cal[mask])).mean())
        rows.append(
            {
                "hour": hour,
                "sample_steps": int(mask.sum()),
                "raw_PICP": raw_picp,
                "calibrated_PICP": cal_picp,
                "coverage_gain": cal_picp - raw_picp,
                "nominal_coverage": nominal,
                "gap_to_nominal": cal_picp - nominal,
                "assumed_start_hour": start_hour,
            }
        )

    df = pd.DataFrame(rows)
    df["difficulty_rank"] = df["gap_to_nominal"].rank(method="dense")
    return df.sort_values("hour").reset_index(drop=True)


def plot_hourly_raw_vs_calibrated(df: pd.DataFrame, output_path: str) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    hours = df["hour"].to_numpy()
    raw_picp = df["raw_PICP"].to_numpy()
    cal_picp = df["calibrated_PICP"].to_numpy()
    nominal = float(df["nominal_coverage"].iloc[0])

    ax.plot(hours, raw_picp, color="#9c2f2f", marker="o", linewidth=2.0, markersize=4.5, label="Raw PICP")
    ax.plot(hours, cal_picp, color="#1f4e79", marker="o", linewidth=2.4, markersize=5.0, label="Calibrated PICP")
    ax.axhline(nominal, color="#333333", linestyle="--", linewidth=1.3, label="Nominal coverage")

    hard_hours = df.nsmallest(3, "gap_to_nominal")
    for _, row in hard_hours.iterrows():
        ax.scatter(row["hour"], row["calibrated_PICP"], color="#d17a22", s=42, zorder=4)
        ax.text(
            row["hour"],
            row["calibrated_PICP"] - 0.035,
            f"h={int(row['hour'])}",
            ha="center",
            va="top",
            fontsize=9,
        )

    ax.set_xlim(0, 23)
    ax.set_xticks(np.arange(24))
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Hour of day", fontsize=12)
    ax.set_ylabel("Prediction interval coverage", fontsize=12)
    ax.set_title("Hourly Raw vs Calibrated Coverage for 90% Prediction Intervals", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight diagnostic for difficult hours in time-of-day coverage.")
    parser.add_argument(
        "--result_dir",
        type=str,
        default="canonical_main_results",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/datasets/ST_EVCDP_v2_canonical",
    )
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--seq_len", type=int, default=24)
    args = parser.parse_args()

    df = build_hourly_diagnostic(args.result_dir, args.data_path, args.delta, args.seq_len)
    delta_tag = f"{args.delta:.1f}"
    csv_path = os.path.join(args.result_dir, f"time_of_day_diagnostic_delta{delta_tag}.csv")
    fig_path = os.path.join(args.result_dir, f"time_of_day_diagnostic_delta{delta_tag}.png")

    df.to_csv(csv_path, index=False)
    plot_hourly_raw_vs_calibrated(df, fig_path)

    print(f"Saved diagnostic table: {csv_path}")
    print(f"Saved diagnostic figure: {fig_path}")
    print(df.nsmallest(5, "gap_to_nominal")[["hour", "raw_PICP", "calibrated_PICP", "coverage_gain", "gap_to_nominal", "sample_steps"]].to_string(index=False))


if __name__ == "__main__":
    main()
