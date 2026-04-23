import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from geopy.distance import geodesic
from tqdm import tqdm

import utils.model_training.training_utils as fn


def jsonify_data(data):
    return json.dumps(data, ensure_ascii=False)


def add_multiplicative_noise(series, noise_level, seed=42):
    """
    对预测结果加入乘性噪声：
        new = series * (1 + eps)
    eps ~ Uniform(-noise_level, +noise_level)

    参数:
        series: pd.Series
        noise_level: float, e.g., 0.05 / 0.10 / 0.20
    """
    if noise_level <= 0:
        return series  # 无噪声

    np.random.seed(seed)
    eps = np.random.uniform(-noise_level, noise_level, size=len(series))
    noisy_series = series * (1 + eps)

    # 防止出现负预测
    noisy_series = noisy_series.clip(lower=0)

    return noisy_series


def main(noise_level=0.0):
    print("="*60)
    print(f"Running experiment with noise_level = ±{noise_level*100:.1f}%")
    print("="*60)

    # ----------------------------
    # 1. 加载数据
    # ----------------------------
    print("Loading data...")
    ev_requests = pd.read_csv('data/ev_requests/ev_requests_new_v6_with_vot.csv')
    charging_stations = pd.read_csv('data/busiest_cluster/busiest_cluster_stations.csv', index_col=0)
    occ, durations, prc, adj, cap = fn.read_dataset_as_df()

    if not isinstance(cap, pd.Series):
        raise ValueError("cap should be a pandas Series")
    cap = cap.astype(float)

    # ----------------------------
    # 2. 加载模型并预测
    # ----------------------------
    model_path = "data/results/ST_EVCDP_v2/dura_pag_informer_no_pretrain_results/dura_pag_informer_no_pretrain_1_bs8_completed.pt"
    model = torch.load(model_path, map_location='cuda')
    model.eval()

    earliest_time_str = ev_requests['request_time'].min()
    earliest_time = datetime.strptime(earliest_time_str, '%Y-%m-%d %H:%M:%S')
    target_time_str = earliest_time.strftime('%Y-%m-%d %H:%M:%S')
    target_hour = earliest_time.replace(minute=0, second=0, microsecond=0)

    one_hour_before = target_hour - timedelta(hours=1)
    time_intervals = [
        (one_hour_before - timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S')
        for i in range(24)
    ]

    input_durations = durations.loc[time_intervals].astype(float)
    input_price = prc.loc[time_intervals].astype(float)

    durations_tensor = torch.tensor(input_durations.values, dtype=torch.float32).unsqueeze(0).transpose(1, 2).to('cuda')
    price_tensor = torch.tensor(input_price.values, dtype=torch.float32).unsqueeze(0).transpose(1, 2).to('cuda')

    with torch.no_grad():
        pred_normalized = model(durations_tensor, price_tensor)
        pred_normalized = pred_normalized.squeeze().cpu().numpy()

    station_ids = list(input_durations.columns.astype(int))
    cap_for_stations = cap.loc[station_ids].astype(float)

    predicted_total_duration = pred_normalized * cap_for_stations * 8
    predictions_df = pd.DataFrame([predicted_total_duration], index=[target_time_str], columns=station_ids)
    predicted_durations = predictions_df.iloc[0]

    # ----------------------------
    # ⭐ 3. 在生成 MCS 前加入噪声（关键修改点）
    # ----------------------------
    print("\nAdding multiplicative noise to MCS demand base...")
    predicted_durations_noisy = add_multiplicative_noise(predicted_durations, noise_level=noise_level)
    print(f"Noise applied. Example values before/after:")
    print(predicted_durations.head(3))
    print(predicted_durations_noisy.head(3))

    # 后续流程全部使用 noisy 版本
    predicted_durations = predicted_durations_noisy

    # ----------------------------
    # 4. excess demand 计算
    # ----------------------------
    common_stations = charging_stations.index.intersection(predicted_durations.index)
    if len(common_stations) == 0:
        raise ValueError("No common stations between predictions and charging_stations")

    predicted_durations = predicted_durations.loc[common_stations]
    cap_common = cap.loc[common_stations].astype(float)

    excess_demand = (predicted_durations - cap_common).clip(lower=0)

    if excess_demand.sum() == 0:
        print("Warning: No excess demand detected. Allocating MCS uniformly.")
        weights = pd.Series(1, index=excess_demand.index)
    else:
        weights = excess_demand

    # ----------------------------
    # 5. 分配 MCS
    # ----------------------------
    num_mcs = 300
    total_weight = weights.sum()
    mcs_allocation = (weights / total_weight * num_mcs).round().astype(int)

    diff = num_mcs - mcs_allocation.sum()
    while diff != 0:
        if diff > 0:
            idx = weights.idxmin()
            mcs_allocation[idx] += 1
            diff -= 1
        else:
            idx = mcs_allocation[mcs_allocation > 0].idxmax()
            mcs_allocation[idx] -= 1
            diff += 1

    mcs_dict = mcs_allocation.to_dict()

    # ----------------------------
    # 6. 随机生成 MCS 位置
    # ----------------------------
    mcs_positions = []
    mcs_id_counter = 0

    for station_id, num_mcs_here in tqdm(mcs_dict.items()):
        if num_mcs_here <= 0:
            continue

        station_id_int = int(station_id)
        station_lat = charging_stations.loc[station_id_int, 'latitude']
        station_lon = charging_stations.loc[station_id_int, 'longitude']

        for _ in range(num_mcs_here):
            valid = False
            while not valid:
                angle = np.random.uniform(0, 2 * np.pi)
                distance_km = np.random.uniform(0.0, 0.5)
                lat_offset = distance_km / 111.0
                lon_offset = distance_km / (111.0 * np.cos(np.radians(station_lat)))
                new_lat = station_lat + lat_offset * np.cos(angle)
                new_lon = station_lon + lon_offset * np.sin(angle)

                valid = True
                for other_id in charging_stations.index:
                    if other_id == station_id_int:
                        continue
                    other_lat = charging_stations.loc[other_id, 'latitude']
                    other_lon = charging_stations.loc[other_id, 'longitude']
                    dist_to_other = geodesic((new_lat, new_lon), (other_lat, other_lon)).km
                    dist_to_home = geodesic((new_lat, new_lon), (station_lat, station_lon)).km
                    if dist_to_other < dist_to_home:
                        valid = False
                        break

            mcs_positions.append({
                'station_id': f"mcs_{mcs_id_counter}",
                'longitude': new_lon,
                'latitude': new_lat,
                'charging_queue': jsonify_data([]),
                'available': True
            })
            mcs_id_counter += 1

    mcs_df = pd.DataFrame(mcs_positions)
    output_path = f'data/mcs_info/mcs_info_pag_informer_best_noise_{noise_level:.2f}_{num_mcs}.csv'
    mcs_df.to_csv(output_path, index=False)
    print(f"\nGenerated {len(mcs_df)} MCS positions with noise_level={noise_level}. Saved to: {output_path}")


if __name__ == "__main__":
    # 示例：分别运行不同噪声水平
    # main(noise_level=0.00)
    # main(noise_level=0.05)
    # main(noise_level=0.10)
    # main(noise_level=0.20)

    main(noise_level=0.30)