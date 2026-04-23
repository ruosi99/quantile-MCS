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


def main():
    # ----------------------------
    # 1. 加载数据
    # ----------------------------
    print("Loading data...")
    ev_requests = pd.read_csv('data/ev_requests/ev_requests_new_v6_with_vot.csv')
    charging_stations = pd.read_csv('data/busiest_cluster/busiest_cluster_stations.csv', index_col=0)
    occ, durations, prc, adj, cap = fn.read_dataset_as_df()
    
    # 确保 cap 是 Series，且索引为 station_id（字符串）
    if not isinstance(cap, pd.Series):
        raise ValueError("cap should be a pandas Series")
    cap = cap.astype(float)

    # ----------------------------
    # 2. 加载模型并预测
    # ----------------------------
    model_path = "data/results/ST_EVCDP_v2/dura_lstm_on_pretrain_results/dura_lstm_on_pretrain_1_bs16_completed.pt"
    model = torch.load(model_path, map_location='cuda')
    model.eval()

    # 获取预测时间点（最早请求时间）
    earliest_time_str = ev_requests['request_time'].min()
    earliest_time = datetime.strptime(earliest_time_str, '%Y-%m-%d %H:%M:%S')
    target_time_str = earliest_time.strftime('%Y-%m-%d %H:%M:%S')

    # 对齐到整点小时（秒设为00）
    target_hour = earliest_time.replace(minute=0, second=0, microsecond=0)

    # 构建输入时间窗口（前8小时）
    one_hour_before = target_hour - timedelta(hours=1)
    time_intervals = [
        (one_hour_before - timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S')
        for i in range(24)
    ]

    # 提取对应历史数据
    input_durations = durations.loc[time_intervals].astype(float)
    input_price = prc.loc[time_intervals].astype(float)

    # 转为模型输入张量 (batch=1, stations, time=8)
    durations_tensor = torch.tensor(input_durations.values, dtype=torch.float32).unsqueeze(0).transpose(1, 2).to('cuda')
    price_tensor = torch.tensor(input_price.values, dtype=torch.float32).unsqueeze(0).transpose(1, 2).to('cuda')

    # # 预测归一化后的占用时长（即 duration / cap）
    # with torch.no_grad():
    #     pred_normalized = model(durations_tensor, price_tensor)
    #     pred_normalized = pred_normalized.squeeze().cpu().numpy()  # shape: (num_stations,)
    with torch.no_grad():
        # 假设 durations_tensor 和 price_tensor 的 shape 是 (1, ...)
        # 在 batch 维度（dim=0）上复制一份，变为 (2, ...)
        durations_tensor_dup = durations_tensor.repeat(2, *([1] * (durations_tensor.ndim - 1)))
        price_tensor_dup = price_tensor.repeat(2, *([1] * (price_tensor.ndim - 1)))

        pred_normalized = model(durations_tensor_dup, price_tensor_dup)  # shape: (2, num_stations)
        pred_normalized = pred_normalized[0].cpu().numpy()  # 只取第一个，shape: (num_stations,)

    station_ids = list(input_durations.columns.astype(int))

    # ✅ 关键修正：还原为真实总占用时长（小时）
    cap_for_stations = cap.loc[station_ids].astype(float)
    predicted_total_duration = pred_normalized * cap_for_stations * 8  # actual duration = normalized * cap

    predictions_df = pd.DataFrame([predicted_total_duration], index=[target_time_str], columns=station_ids)
    predicted_durations = predictions_df.iloc[0]  # Series: station_id -> total predicted occupied hours

    # ----------------------------
    # 3. 计算供需不匹配程度（用于MCS分配）
    # ----------------------------
    # 只考虑与 charging_stations 共有的站点
    common_stations = charging_stations.index.intersection(predicted_durations.index)
    if len(common_stations) == 0:
        raise ValueError("No common stations between predictions and charging_stations")

    predicted_durations = predicted_durations.loc[common_stations]
    cap_common = cap.loc[common_stations].astype(float)

    # 计算“超额需求”：max(0, 预测占用时长 - 最大供给能力)
    # 最大供给能力 = 充电桩数（cap），因为1小时内最多提供 cap 小时的占用
    excess_demand = predicted_durations - cap_common
    excess_demand = excess_demand.clip(lower=0)  # 只保留正超额

    # 如果所有站点都无超额需求，则均匀分配或不分配（这里设为0）
    if excess_demand.sum() == 0:
        print("Warning: No excess demand detected. Allocating MCS uniformly.")
        weights = pd.Series(1, index=excess_demand.index)
    else:
        weights = excess_demand

    # ----------------------------
    # 4. 分配 MCS 数量（共100个）
    # ----------------------------
    num_mcs = 300
    total_weight = weights.sum()
    mcs_allocation = (weights / total_weight * num_mcs).round().astype(int)

    # 校正总数为 num_mcs
    diff = num_mcs - mcs_allocation.sum()
    while diff != 0:
        if diff > 0:
            # 增加权重最小的站点（或随机）
            idx = weights.idxmin()
            mcs_allocation[idx] += 1
            diff -= 1
        else:
            idx = mcs_allocation[mcs_allocation > 0].idxmax()
            mcs_allocation[idx] -= 1
            diff += 1

    mcs_dict = mcs_allocation.to_dict()
    print("MCS allocation based on excess demand:", mcs_dict)

    # ----------------------------
    # 5. 生成 MCS 随机位置（带归属校验）
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
                # 在0~0.5公里内随机生成偏移
                angle = np.random.uniform(0, 2 * np.pi)
                distance_km = np.random.uniform(0.0, 0.5)
                # 转换为经纬度偏移（近似）
                lat_offset = distance_km / 111.0
                lon_offset = distance_km / (111.0 * np.cos(np.radians(station_lat)))
                new_lat = station_lat + lat_offset * np.cos(angle)
                new_lon = station_lon + lon_offset * np.sin(angle)

                # 校验：确保离归属站最近
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

    # ----------------------------
    # 6. 保存结果
    # ----------------------------
    mcs_df = pd.DataFrame(mcs_positions)
    mcs_df.to_csv(f'data/mcs_info/mcs_info_by_model_lstm_{num_mcs}.csv', index=False)
    print(f"\nGenerated {len(mcs_df)} MCS positions. Saved to data/mcs_info/mcs_info_by_model_lstm_{num_mcs}.csv")


if __name__ == "__main__":
    main()