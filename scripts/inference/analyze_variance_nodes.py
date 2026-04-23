import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==========================================
# 1. 数据读取 (你提供的函数)
# ==========================================
def read_dataset_v2():
    data_path = 'data/datasets/ST_EVCDP_v2/'
    inf = pd.read_csv(data_path + 'inf.csv', index_col=None, header=0)
    inf.fillna(0, inplace=True)
    occ = pd.read_csv(data_path + 'occupancy.csv', index_col=0, header=0)
    occ.fillna(0, inplace=True)
    e_price = pd.read_csv(data_path + 'e_price.csv', index_col=0, header=0)
    e_price.fillna(0, inplace=True)
    s_price = pd.read_csv(data_path + 's_price.csv', index_col=0, header=0)
    s_price.fillna(0, inplace=True)
    price = e_price + s_price
    adj = pd.read_csv(data_path + 'adjacency_matrix.csv', index_col=0, header=0)
    adj.fillna(0, inplace=True)
    duration = pd.read_csv(data_path + 'duration.csv', index_col=0, header=0)
    duration.fillna(0, inplace=True)
    cap = np.array(inf['pile_count'], dtype=float).reshape(1, -1)  # parking_capability

    occ = np.array(occ, dtype=float) / cap
    duration = np.array(duration, dtype=float) / cap
    price = np.array(price, dtype=float)
    adj = torch.tensor(np.array(adj, dtype=float), dtype=torch.float32)
    return occ, duration, price, adj, cap

# ==========================================
# 2. 定义简单的 Dataset 类 (用于滑动窗口)
# ==========================================
class TestDataset(Dataset):
    def __init__(self, demand, price, seq_len, pred_len):
        self.demand = torch.tensor(demand, dtype=torch.float32)
        self.price = torch.tensor(price, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.demand) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_end = index + self.seq_len
        r_end = s_end + self.pred_len
        
        # 输入: [seq_len, N]
        x_demand = self.demand[index:s_end, :]
        x_price = self.price[index:s_end, :]
        
        # 标签: [pred_len, N] -> 这里 pred_len=1, 所以是 [1, N], squeeze为 [N]
        label = self.demand[s_end:r_end, :].squeeze(0)
        
        return x_demand, x_price, label

# ==========================================
# 3. 辅助函数：数据划分
# ==========================================
def split_data(data, train_rate=0.7, valid_rate=0.1, calib_rate=0.1, test_rate=0.1):
    total_len = len(data)
    train_end = int(total_len * train_rate)
    valid_end = train_end + int(total_len * valid_rate)
    calib_end = valid_end + int(total_len * calib_rate)
    
    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    calib_data = data[valid_end:calib_end]
    test_data = data[calib_end:]
    return train_data, valid_data, calib_data, test_data

# ==========================================
# 4. 主逻辑
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='Inference and Variance Analysis')
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型文件路径 (.pt)')
    parser.add_argument('--model_name', type=str, default='model', help='模型名称，用于判断输入类型 (若含dura则用duration)')
    parser.add_argument('--output_dir', type=str, default='data/results/variance_analysis', help='结果保存目录')
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
    parser.add_argument('--seq_len', type=int, default=24, help='输入序列长度')
    parser.add_argument('--pred_len', type=int, default=1, help='预测长度')
    parser.add_argument('--use_cuda', type=str, default='true', help='是否使用CUDA')
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda:0" if args.use_cuda.lower() == 'true' and torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------
    # A. 加载数据
    # ---------------------------
    print("正在加载数据...")
    occ, duration, price, adj, cap = read_dataset_v2()
    
    # 根据模型名选择输入序列 (逻辑同原代码)
    if "dura" in args.model_name.lower():
        input_series = duration
        print("使用 Duration 作为输入特征。")
    else:
        input_series = occ
        print("使用 Occupancy 作为输入特征。")

    # 划分数据集 (只需要 train 来算方差，test 来做推理)
    train_data, _, _, test_data = split_data(input_series)
    _, _, _, test_price = split_data(price)
    
    # ---------------------------
    # B. 计算方差并挑选站点
    # ---------------------------
    print("正在计算方差并挑选站点...")
    # 计算训练集每个站点的方差
    node_variance = np.var(train_data, axis=0) # shape: (N,)
    num_nodes = len(node_variance)
    
    # 按方差大小排序，获取索引
    sorted_indices = np.argsort(node_variance)
    
    # 定义分组 (三等分)
    n_per_group = num_nodes // 3
    
    # 低方差组：前 1/3
    low_group_indices = sorted_indices[:n_per_group]
    # 中方差组：中间 1/3
    med_group_indices = sorted_indices[n_per_group : 2*n_per_group]
    # 高方差组：后 1/3
    high_group_indices = sorted_indices[2*n_per_group:]
    
    # 挑选代表性站点 (每组取最中间的那个，代表该组典型特征)
    # 如果你想取该组方差最大/最小的，可以改用 index 0 或 -1
    pick_low = low_group_indices[len(low_group_indices)//2]
    pick_med = med_group_indices[len(med_group_indices)//2]
    pick_high = high_group_indices[len(high_group_indices)//2]
    
    selected_nodes = {
        "low": pick_low,
        "med": pick_med,
        "high": pick_high
    }
    
    print(f"站点挑选结果 (基于训练集方差):")
    for level, idx in selected_nodes.items():
        print(f"  [{level.upper()}] 站点ID: {idx}, 方差值: {node_variance[idx]:.6f}")

    # ---------------------------
    # C. 加载模型
    # ---------------------------
    print(f"正在加载模型: {args.model_path}")
    # 使用 map_location 确保模型加载到正确的设备
    model = torch.load(args.model_path, map_location=device)
    model.eval() # 设置为评估模式
    model.to(device)

    # ---------------------------
    # D. 构建测试 DataLoader
    # ---------------------------
    test_dataset = TestDataset(test_data, test_price, args.seq_len, args.pred_len)
    # 注意：推理时 shuffle 必须为 False，否则时间顺序会乱
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # ---------------------------
    # E. 推理循环
    # ---------------------------
    print("正在推理...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for demand, price, label in tqdm(test_loader, desc='Inference'):
            demand = demand.to(device) # (B, T, N)
            price = price.to(device)   # (B, T, N)
            # label: (B, N)
            
            # 模型前向传播
            pred = model(demand, price)
            
            # 如果模型输出是 (B, N, Q) (分位数)，我们只取中位数 (假设 index 3 是 0.5)
            # 如果模型输出是 (B, N)，直接使用
            if len(pred.shape) == 3:
                pred_point = pred[:, :, 3] # 假设第3个位置是0.5分位数，具体取决于你的quantiles定义
                # 如果不确定，可以使用 predict_q[:, :, quantiles.index(0.5)]
            else:
                pred_point = pred
                
            all_preds.append(pred_point.cpu().numpy())
            all_labels.append(label.numpy())

    # 合并结果
    # shape: (Total_Time, N)
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # 反归一化 (还原为真实值)
    # cap shape: (1, N)
    preds_real = preds * cap
    labels_real = labels * cap

    # ---------------------------
    # F. 保存选中站点的结果
    # ---------------------------
    print("正在保存结果...")
    for level, node_idx in selected_nodes.items():
        # 提取该站点的数据
        pred_series = preds_real[:, node_idx]
        label_series = labels_real[:, node_idx]
        
        # 构建 DataFrame
        df = pd.DataFrame({
            'Time_Step': np.arange(len(pred_series)),
            'Ground_Truth': label_series,
            'Prediction': pred_series,
            'Variance_Level': level,
            'Station_ID': node_idx
        })
        
        # 保存路径
        save_path = os.path.join(args.output_dir, f'station_{level}_id{node_idx}.csv')
        df.to_csv(save_path, index=False)
        print(f"已保存: {save_path}")

    print("全部完成。")

if __name__ == '__main__':
    main()