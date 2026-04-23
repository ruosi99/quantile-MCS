import pandas as pd
import numpy as np
from geopy.distance import geodesic
from datetime import datetime, timedelta
import random
from tqdm import tqdm
from scipy.spatial import distance_matrix

def generate_charging_needs(station_id, num_needs, station_lat, station_lon, radius, total_usage, time_str):
    """
    生成充电需求数据
    :param station_id: 充电站 ID
    :param num_needs: 充电需求数量
    :param station_lon: 充电站经度
    :param station_lat: 充电站纬度
    :param radius: 充电站的第5近邻充电站距离
    :param total_usage: 该时间步总的充电用量
    :param time_str: 充电请求时间
    :return: 充电请求列表
    """
    start_time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    num_needs = max(num_needs, 1)  # 避免 0

    charging_needs = []
    for _ in range(num_needs):
        angle = random.uniform(0, 2 * np.pi)
        distance = np.abs(np.random.normal(radius / 2, radius / 6))  # 确保非负
        distance = min(distance, radius)  # 限制最大半径

        demand_lat = station_lat + (distance / 111) * np.cos(angle)
        demand_lon = station_lon + (distance / (111 * np.cos(np.radians(station_lat)))) * np.sin(angle)

        demand_usage = np.clip(np.random.normal(total_usage / num_needs, total_usage / (num_needs * 2)), 0, None)
        if demand_usage < 10:
            demand_usage = np.random.uniform(20, 50)  # 10-50 kWh 充电需求范围

        demand_time = start_time + timedelta(seconds=random.uniform(0, 3600))

        charging_needs.append({
            'request_time': demand_time.strftime('%Y-%m-%d %H:%M:%S'),
            'station_id': station_id,
            'latitude': demand_lat,
            'longitude': demand_lon,
            'energy_needed': demand_usage
        })

    return charging_needs


# **加载数据**
data_dir = 'data/datasets/ST_EVCDP_v2/'
occupancy_df = pd.read_csv(data_dir + 'occupancy.csv', index_col=0, header=0)
time_usage_df = pd.read_csv(data_dir + 'duration.csv', index_col=0, header=0)

# 读取最繁忙的充电站信息，并筛选 `stations_df`
busiest_stations_df = pd.read_csv('data/busiest_cluster/busiest_cluster_stations.csv')
busiest_station_ids = set(busiest_stations_df["station_id"].astype(int))

# 仅加载最繁忙充电站的 `stations_df`
stations_df = pd.read_csv(data_dir + 'inf.csv', index_col=0, header=0)
stations_df = stations_df[stations_df.index.isin(busiest_station_ids)]  # 只保留最繁忙的充电站

# 取最后 20% 时间戳的数据
threshold = int(len(occupancy_df) * 0.2)
num_rows = 24
occupancy_df = occupancy_df.iloc[-threshold: -threshold + num_rows]
time_usage_df = time_usage_df.iloc[-threshold: -threshold + num_rows]

# 计算充电用量
charging_power = 60
scale_factor = 3
total_usage = time_usage_df * charging_power * scale_factor

# **计算第 5 近的充电站距离**
station_coords = stations_df[['latitude', 'longitude']].values
dist_matrix = distance_matrix(station_coords, station_coords)

# 对每个站点找到第 5 近的充电站距离
k = 20  # 取第 5 近的充电站距离
sorted_distances = np.sort(dist_matrix, axis=1)  # 对每一行（站点）排序
fifth_nearest_distances = np.where(sorted_distances.shape[1] > k, sorted_distances[:, k], sorted_distances[:, -1])  # 取第 k 近的距离

# 充电需求存储
all_charging_needs = []

# **按顺序处理所有充电站**
for index, row in tqdm(occupancy_df.iterrows(), total=occupancy_df.shape[0], desc="Processing timestamps"):
    time_stamp = index

    for station_id, value in row.items():
        station_id = int(station_id)

        # **跳过非最繁忙充电站**
        if station_id not in stations_df.index:
            continue

        station_data = stations_df.loc[station_id]
        # **跳过 NaN**
        if pd.isna(value):
            continue
        num_needs = int(value) * scale_factor
        fifth_distance = fifth_nearest_distances[stations_df.index.get_loc(station_id)]  # 取第 5 近的距离

        # 生成充电需求
        all_charging_needs.extend(
            generate_charging_needs(
                station_id, num_needs, station_data['latitude'], station_data['longitude'],
                fifth_distance, total_usage.loc[time_stamp, str(station_id)], time_stamp
            )
        )

# **保存充电需求数据**
charging_needs_df = pd.DataFrame(all_charging_needs)
charging_needs_df.to_csv(f'data/ev_requests/ev_requests_scale_{scale_factor}.csv', index=False)

print(f"EV 充电需求生成完毕，数据已保存至 'data/ev_requests/ev_requests_scale_{scale_factor}.csv'")
