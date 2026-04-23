# assigned_heatmap_comparison.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import argparse
from datetime import datetime, timedelta
import os
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Generate heatmaps comparing assigned charging volumes with and without prediction.")
    parser.add_argument("--start_time", required=True, help="Start time in YYYY-MM-DD HH:MM:SS format")
    parser.add_argument("--end_time", required=True, help="End time in YYYY-MM-DD HH:MM:SS format")
    parser.add_argument("--num_stations", type=int, required=True, help="Number of stations to randomly sample")
    parser.add_argument("--output_dir", required=True, help="Output directory for heatmaps")
    parser.add_argument("--with_pred_file", default="data/greedy_algorithm/with_pred/ev_requests_with_assignments.csv", help="File for assignments with prediction")
    parser.add_argument("--no_pred_file", default="data/greedy_algorithm/no_pred/ev_requests_with_assignments.csv", help="File for assignments without prediction")
    return parser.parse_args()

def process_assignments(assign_df, start_dt, end_dt):
    assign_df["request_time"] = pd.to_datetime(assign_df["request_time"])
    assign_df["end_time"] = pd.to_datetime(assign_df["end_time"])

    # Filter to requests within time range (based on request_time)
    assign_filtered = assign_df[(assign_df["request_time"] >= start_dt) & (assign_df["request_time"] <= end_dt)]

    # Only fcs
    fcs_df = assign_filtered[assign_filtered["charging_entity_name"].str.startswith("fcs_")]

    # Extract station id
    fcs_df["station_id"] = fcs_df["charging_entity_name"].apply(lambda x: x.split("_")[1])

    # Calculate charging_start
    fcs_df["charging_time_min"] = fcs_df["charging_time"]
    fcs_df["charging_start"] = fcs_df["end_time"] - pd.to_timedelta(fcs_df["charging_time_min"], unit="min")

    # Create dict for station-hour energy
    station_hour_energy = defaultdict(lambda: defaultdict(float))

    for _, row in fcs_df.iterrows():
        station = row["station_id"]
        start = row["charging_start"]
        end = row["end_time"]
        total_energy = row["energy_needed"]
        total_min = (end - start).total_seconds() / 60
        if total_min == 0:
            continue

        current = start.floor("H")
        while current < end:
            next_hour = current + timedelta(hours=1)
            interval_end = min(next_hour, end)
            interval_min = (interval_end - max(current, start)).total_seconds() / 60
            fraction = interval_min / total_min
            energy_portion = total_energy * fraction
            station_hour_energy[station][current] += energy_portion
            current = next_hour

    return station_hour_energy, fcs_df

def main():
    args = parse_args()
    start_dt = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(args.end_time, "%Y-%m-%d %H:%M:%S")

    # Load and process with_pred data
    with_pred_df = pd.read_csv(args.with_pred_file)
    with_pred_energy, with_pred_fcs = process_assignments(with_pred_df, start_dt, end_dt)

    # Load and process no_pred data
    no_pred_df = pd.read_csv(args.no_pred_file)
    no_pred_energy, no_pred_fcs = process_assignments(no_pred_df, start_dt, end_dt)

    # Get all unique times (use with_pred as reference, assume same time range)
    times = pd.date_range(start=start_dt, end=end_dt, freq="H")
    times_sorted = sorted(times)

    # Get unique stations from both
    with_pred_stations = set(with_pred_energy.keys())
    no_pred_stations = set(no_pred_energy.keys())

    # Common stations
    common_stations = with_pred_stations.intersection(no_pred_stations)

    if len(common_stations) < args.num_stations:
        print(f"Only {len(common_stations)} common stations available, sampling all.")
        sampled_stations = list(common_stations)
    else:
        sampled_stations = random.sample(list(common_stations), args.num_stations)

    # Sort numerically
    sampled_stations = sorted(sampled_stations, key=int)

    # Prepare dataframes
    def create_dataframe(energy_dict, times, stations):
        df = pd.DataFrame(index=times, columns=stations, data=0.0)
        for station in stations:
            for t in times:
                df.at[t, station] = energy_dict[station].get(t, 0.0)
        return df

    with_pred_data = create_dataframe(with_pred_energy, times_sorted, sampled_stations)
    no_pred_data = create_dataframe(no_pred_energy, times_sorted, sampled_stations)

    # Generate heatmaps
    os.makedirs(args.output_dir, exist_ok=True)

    def plot_heatmap(df, title, filename):
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.T, cmap="YlGnBu", cbar_kws={"label": "Power (kWh)"})
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Station No.")
        plt.savefig(os.path.join(args.output_dir, filename))
        plt.close()

    plot_heatmap(with_pred_data, "Assigned Charging Volume (With Prediction)", "with_pred_heatmap.png")
    plot_heatmap(no_pred_data, "Assigned Charging Volume (Without Prediction)", "no_pred_heatmap.png")

    print("Heatmaps generated successfully.")

if __name__ == "__main__":
    main()
