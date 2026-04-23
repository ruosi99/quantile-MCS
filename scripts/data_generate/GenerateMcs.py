import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset_process import jsonify_data

random_seed = 42
np.random.seed(random_seed)

# Load FCS Info
stations = pd.read_csv("data/busiest_cluster/busiest_cluster_stations.csv")

# MCS Location Initialization
num_mcs = 300
mcs_locations = stations[['longitude', 'latitude']].sample(n=num_mcs, replace=True).values

# 计算经纬度的变化范围
longitude_range = stations['longitude'].max() - stations['longitude'].min()
latitude_range = stations['latitude'].max() - stations['latitude'].min()

# 定义偏移范围并生成随机偏移
scale = 5
longitude_offset = np.random.uniform(-longitude_range * scale, longitude_range * scale, size=mcs_locations.shape[0])
latitude_offset = np.random.uniform(-latitude_range * scale, latitude_range * scale, size=mcs_locations.shape[0])

# 添加偏移
mcs_locations[:, 0] += longitude_offset
mcs_locations[:, 1] += latitude_offset

# 确保偏移后的值在经纬度范围内
# mcs_locations[:, 0] = np.clip(mcs_locations[:, 0], stations['longitude'].min(), stations['longitude'].max())
# mcs_locations[:, 1] = np.clip(mcs_locations[:, 1], stations['latitude'].min(), stations['latitude'].max())

# Create the MCS dataset
"""
统一起见：充电站的充电桩和移动充电机器人的充电头都称作充电桩，数据格式保持一致，你喜欢叫充电头的话也可以，爱你呦！！！

每个充电桩维护一个名为'charging_queue'的充电工作队列，里面每个元素都是ev_requests.csv里面的行号，格式为[1, 3, 5]
可以根据行号访问ev_requests.csv具体的充电需求信息，这么做好处有二
    1、在所有充电头都不空闲时将充电需求直接写入队列记录
    2、最后可以根据队列统计评估指标，如充电时间，等待时间等，甚至可以还原出原始数据里面占用、消耗电量等信息

'available'表示充电桩的可用状态，可以以任意频率去更新；只要给定一个时间点，根据充电桩的工作队列记录即可以判断出该充电桩是否可用
"""
mcs = pd.DataFrame({
    'station_id': ["mcs_{}".format(i) for i in range(1, num_mcs + 1)],
    'longitude': mcs_locations[:, 0],
    'latitude': mcs_locations[:, 1],
    'charging_queue':jsonify_data([]),
    'available': [True] * num_mcs
})

# Save the EV requests to a CSV file for use in simulation
mcs.to_csv(f"data/mcs_info/mcs_info_{num_mcs}_v{scale}.csv", index=False)
print(f"Generated {len(mcs)} MCS positions. Saved to data/mcs_info/mcs_info_{num_mcs}_v{scale}.csv")