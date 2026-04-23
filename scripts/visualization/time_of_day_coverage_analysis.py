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


def compute_hourly_metrics(labels: np.ndarray, lower: np.ndarray, upper: np.ndarray, start_hour: int) -> pd.DataFrame:
    n_steps = labels.shape[0]
    hour_labels = (np.arange(n_steps) + start_hour) % 24

    rows = []
    for hour in range(24):
        mask = hour_labels == hour
        covered = (labels[mask] >= lower[mask]) & (labels[mask] <= upper[mask])
        rows.append(
            {
                "hour": hour,
                "sample_steps": int(mask.sum()),
                "PICP": float(covered.mean()),
                "MPIW": float((upper[mask] - lower[mask]).mean()),
            }
        )

    return pd.DataFrame(rows)


def plot_hourly_metrics(df: pd.DataFrame, output_path: str, nominal_coverage: float = 0.9) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()

    hours = df["hour"].to_numpy()
    picp = df["PICP"].to_numpy()
    mpiw = df["MPIW"].to_numpy()

    picp_color = "#1f4e79"
    mpiw_color = "#c66b3d"

    ax1.plot(hours, picp, color=picp_color, marker="o", linewidth=2.4, markersize=5, label="Hourly PICP")
    ax1.axhline(nominal_coverage, color=picp_color, linestyle="--", linewidth=1.4, alpha=0.8, label="Nominal coverage")

    ax2.plot(hours, mpiw, color=mpiw_color, marker="s", linewidth=2.2, markersize=4.5, label="Hourly MPIW")

    ax1.set_xlim(0, 23)
    ax1.set_xticks(np.arange(24))
    ax1.set_xlabel("Hour of day", fontsize=12)
    ax1.set_ylabel("PICP", color=picp_color, fontsize=12)
    ax2.set_ylabel("MPIW", color=mpiw_color, fontsize=12)
    ax1.set_title("Hourly Reliability and Interval Width for Calibrated 90% Prediction Intervals", fontsize=14, fontweight="bold")

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.tick_params(axis="y", colors=picp_color)
    ax2.tick_params(axis="y", colors=mpiw_color)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-of-day coverage analysis from canonical saved interval outputs.")
    parser.add_argument(
        "--result_dir",
        type=str,
        default="canonical_main_results",
        help="Canonical result folder containing label_list.npy and cqr interval arrays.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/datasets/ST_EVCDP_v2_canonical",
        help="Dataset folder containing duration.csv for inferring the first target hour.",
    )
    parser.add_argument("--delta", type=float, default=0.1, help="Interval delta to analyze.")
    parser.add_argument("--seq_len", type=int, default=24, help="Encoder sequence length used by the model.")
    args = parser.parse_args()

    delta_tag = f"{args.delta:.1f}"
    labels = np.load(os.path.join(args.result_dir, "label_list.npy"))
    lower = np.load(os.path.join(args.result_dir, f"cqr_L_delta{delta_tag}.npy"))
    upper = np.load(os.path.join(args.result_dir, f"cqr_U_delta{delta_tag}.npy"))

    start_hour = infer_start_hour_from_current_dataset(args.data_path, seq_len=args.seq_len)
    hourly_df = compute_hourly_metrics(labels, lower, upper, start_hour=start_hour)
    hourly_df["nominal_coverage"] = 1.0 - args.delta
    hourly_df["assumed_start_hour"] = start_hour
    hourly_df["time_alignment_note"] = (
        "Hours are inferred from contiguous hourly saved predictions anchored to the current dataset split start hour."
    )

    csv_path = os.path.join(args.result_dir, f"time_of_day_coverage_delta{delta_tag}.csv")
    fig_path = os.path.join(args.result_dir, f"time_of_day_coverage_delta{delta_tag}.png")
    hourly_df.to_csv(csv_path, index=False)
    plot_hourly_metrics(hourly_df, fig_path, nominal_coverage=1.0 - args.delta)

    print(f"Saved hourly metrics: {csv_path}")
    print(f"Saved figure: {fig_path}")
    print(hourly_df[["hour", "PICP", "MPIW"]].to_string(index=False))


if __name__ == "__main__":
    main()
