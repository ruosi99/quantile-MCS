# station_heatmap.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import argparse
from datetime import datetime, timedelta
import os
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Generate heatmaps for charging station busyness.")
    parser.add_argument("--start_time", required=True, help="Start time in YYYY-MM-DD HH:MM:SS format")
    parser.add_argument("--end_time", required=True, help="End time in YYYY-MM-DD HH:MM:SS format")
    parser.add_argument("--num_stations", type=int, required=True, help="Number of stations to randomly sample")
    parser.add_argument("--output_dir", required=True, help="Output directory for heatmaps")
    return parser.parse_args()

def main():
    args = parse_args()
    start_dt = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(args.end_time, "%Y-%m-%d %H:%M:%S")

    # Load original volume data
    volume_df = pd.read_csv("ST_EVCDP_v2/volume.csv", parse_dates=["time"])
    volume_df.set_index("time", inplace=True)

    # Filter to time range
    volume_filtered = volume_df[(volume_df.index >= start_dt) & (volume_df.index <= end_dt)]

    # Load assigned data
    assign_df = pd.read_csv("data/greedy_algorithm/with_pred/ev_requests_with_assignments.csv")
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

    # Now, create a dict for station-hour energy
    from collections import defaultdict
    station_hour_energy = defaultdict(lambda: defaultdict(float))

    for _, row in fcs_df.iterrows():
        station = row["station_id"]
        start = row["charging_start"]
        end = row["end_time"]
        total_energy = row["energy_needed"]
        total_min = (end - start).total_seconds() / 60
        if total_min == 0:
            continue

        # Get all hours from start to end
        current = start.floor("h")
        while current < end:
            next_hour = current + timedelta(hours=1)
            interval_end = min(next_hour, end)
            interval_min = (interval_end - max(current, start)).total_seconds() / 60
            fraction = interval_min / total_min
            energy_portion = total_energy * fraction
            station_hour_energy[station][current] += energy_portion
            current = next_hour

    # Get all unique times from volume_filtered
    times = volume_filtered.index
    times_sorted = sorted(times)

    # Get all unique stations from fcs
    assigned_stations = set(station_hour_energy.keys())

    # Stations in volume are strings like '1001'
    volume_stations = set(volume_df.columns)

    # Common stations
    common_stations = assigned_stations.intersection(volume_stations)

    if len(common_stations) < args.num_stations:
        print(f"Only {len(common_stations)} common stations available, sampling all.")
        sampled_stations = list(common_stations)
    else:
        sampled_stations = random.sample(list(common_stations), args.num_stations)

    # Sort numerically
    sampled_stations = sorted(sampled_stations, key=int)

    # For original data
    original_data = volume_filtered[sampled_stations]

    # For assigned data
    assigned_data = pd.DataFrame(index=times_sorted, columns=sampled_stations, data=0.0)
    for station in sampled_stations:
        for t in times_sorted:
            assigned_data.at[t, station] = station_hour_energy[station].get(t, 0.0)

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

    plot_heatmap(original_data, "Original Charging Volume", "original_heatmap.png")
    plot_heatmap(assigned_data, "Assigned Charging Volume", "assigned_heatmap.png")

    print("Heatmaps generated successfully.")

if __name__ == "__main__":
    main()
