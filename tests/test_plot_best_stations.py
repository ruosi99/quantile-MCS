import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import utils.model_training.training_utils as fn

# ======================
# 配置与路径
# ======================
base_path = 'data/results-v0/results/'
out_dir = 'data/results-v0/results/scientific_figure_illustration'
os.makedirs(out_dir, exist_ok=True)

models = {
    # 'fgn': 'occ_fgn_pred_res',
    # 'lstm': 'occ_lstm_pred_res',
    # 'pag': 'occ_pag_pred_res',
    'pag_informer': 'occ_pag_informer_on_pretrain_pred_res'
}

num_best = 5  # 选择表现最好的站点数量

# ======================
# 加载基础数据
# ======================

# 加载 labels（所有模型共享）
labels_path = os.path.join(base_path, models['pag_informer'], 'label_list.npy')
labels = np.load(labels_path)  # shape: (seq_len, num_nodes)
seq_len, num_nodes = labels.shape

# 加载 pile_count 用于反归一化
inf_path = 'data/datasets/ST_EVCDP_v2/inf.csv'
inf_df = pd.read_csv(inf_path)
pile_count = inf_df['pile_count'].values  # shape: (num_nodes,)
cap = pile_count.reshape(1, -1)  # (1, num_nodes)

# 反归一化 labels
scaled_labels = labels * cap  # shape: (seq_len, num_nodes)

# 加载所有模型的预测结果并反归一化
predicts = {}
scaled_predicts = {}
for model, folder in models.items():
    pred_path = os.path.join(base_path, folder, 'predict_list.npy')
    pred = np.load(pred_path)  # shape: (seq_len, num_nodes)
    predicts[model] = pred
    scaled_predicts[model] = pred * cap

# ======================
# 定义 MAPE 计算函数
# ======================
def compute_mape(pred, real, eps=0.01):
    """计算单个站点的 MAPE (%)，忽略接近零的真实值"""
    valid_mask = real > eps
    if not np.any(valid_mask):
        return np.inf
    return np.mean(np.abs((real[valid_mask] - pred[valid_mask]) / real[valid_mask])) * 100

# ======================
# 计算每个模型在每个站点的 MAPE
# ======================
mape_per_model_station = {}
for model, pred in scaled_predicts.items():
    mape_list = [
        compute_mape(pred[:, s], scaled_labels[:, s])
        for s in range(num_nodes)
    ]
    mape_per_model_station[model] = np.array(mape_list)

# ======================
# 选择 pag_informer 表现最优的站点（MAPE 最低且优于其他模型）
# ======================
candidate_stations = []
for s in range(num_nodes):
    pag_mape = mape_per_model_station['pag_informer'][s]
    # 检查是否严格优于所有其他模型
    if all(pag_mape < mape_per_model_station[m][s] for m in models if m != 'pag_informer'):
        candidate_stations.append((s, pag_mape))

# 按 MAPE 升序排序，取前 num_best 个
candidate_stations.sort(key=lambda x: x[1])
best_stations = [s for s, _ in candidate_stations[:num_best]] + [1267, 1268, 1269]
print(f"Best stations (lowest MAPE for pag_informer and better than others): {best_stations}")

# ======================
# 加载时间戳（用于绘图）
# ======================
time_csv_path = 'data/datasets/ST_EVCDP_v2/volume.csv'
df = pd.read_csv(time_csv_path)
times = df.iloc[:, 0]  # 第一列为时间

# 取最后 20% 的时间点（与预测数据对齐）
last_20_pct = int(len(times) * 0.2)
times = times[-last_20_pct:].reset_index(drop=True)

# 确保时间长度与 seq_len 一致
if len(times) != seq_len:
    print(f"Warning: Time length {len(times)} != seq_len {seq_len}. Truncating to match.")
    times = times[:seq_len]

# ======================
# 全局指标计算 & 保存到 Excel
# ======================
metrics_dict = {}
flat_real = scaled_labels.flatten()
columns = ['MSE', 'RMSE', 'MAPE', 'RAE', 'MAE', 'R2', 'MedAE', 'EVS']

for model, pred in scaled_predicts.items():
    flat_pred = pred.flatten()
    metrics_list = fn.metrics_new(flat_pred, flat_real)
    metrics_dict[model] = metrics_list
    print(f"\nMetrics for {model}: {dict(zip(columns, metrics_list))}")

df_metrics = pd.DataFrame.from_dict(metrics_dict, orient='index', columns=columns)
df_metrics.to_excel(os.path.join(out_dir, 'metrics_summary.xlsx'), index_label='Model')
print("Metrics saved to metrics_summary.xlsx")

# ======================
# 绘图：每个 best station 的时间序列对比
# ======================
num_points = min(96 * 7, seq_len)
start_idx = (seq_len - num_points) // 2
end_idx = start_idx + num_points

for station in best_stations:
    plt.figure(figsize=(12, 6))
    plt.title(f'Time Series Predictions for Station {station}')
    
    # 真实值
    plt.plot(times[start_idx:end_idx], scaled_labels[start_idx:end_idx, station],
             label='True', color='black', linewidth=2)
    
    # 各模型预测值
    for model, pred in scaled_predicts.items():
        plt.plot(times[start_idx:end_idx], pred[start_idx:end_idx, station],
                 label=model)
    
    plt.xlabel('Time')
    plt.ylabel('Occupancy (Scaled Back)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
    plt.tight_layout()
    
    plt.savefig(os.path.join(out_dir, f'station_{station}_timeseries_comparison.png'))
    plt.close()

print("Time series plots generated and saved.")