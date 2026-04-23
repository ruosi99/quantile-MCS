import pandas as pd
import numpy as np
from geopy.distance import geodesic
from tqdm import tqdm

# 读取CSV文件
data = pd.read_csv('ST_EVCDP_v2/inf.csv')

# 获取经纬度信息
coords = data[['latitude', 'longitude']].values

# 初始化邻接矩阵
adj_matrix = np.zeros((len(data), len(data)))

# 计算地理距离并生成邻接矩阵
for i in tqdm(range(len(data))):
    distances = []
    for j in range(len(data)):
        if i != j:
            dist = geodesic(coords[i], coords[j]).kilometers
            distances.append((dist, j))
    # 找到最近的三个充电站
    distances.sort(key=lambda x: x[0])
    nearest_indices = [index for _, index in distances[:10]]
    
    # 更新邻接矩阵
    adj_matrix[i, nearest_indices] = 1

# 创建一个DataFrame用于展示邻接矩阵
adj_df = pd.DataFrame(adj_matrix, index=data['station_id'], columns=data['station_id'])

# 保存邻接矩阵为CSV文件
adj_df.to_csv('ST_EVCDP_v2/adjacency_matrix_v2.csv')

print("邻接矩阵已生成并保存为 'adjacency_matrix.csv'")
