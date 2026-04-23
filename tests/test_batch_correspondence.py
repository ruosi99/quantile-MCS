# test_batch_correspondence.py
import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# 假设项目根目录
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

import utils.model_training.training_utils as fn

def test_batch_correspondence(data_dir='data/datasets/charged/SZH/', model_name='dura_pag', bs=16, seq_l=24*7, pre_l=1, num_batches_to_check=5):
    # 加载数据
    occ, duration, prc, distance, time_info = fn.read_dataset_charged(data_dir)
    
    # 选择 demand
    demand = duration if 'dura' in model_name else occ
    
    # 划分数据集
    train_demand, _, _ = fn.division(demand, train_rate=0.6, valid_rate=0.2, test_rate=0.2)
    train_price, _, _ = fn.division(prc, train_rate=0.6, valid_rate=0.2, test_rate=0.2)
    train_time_info, _, _ = fn.division_time(time_info, train_rate=0.6, valid_rate=0.2, test_rate=0.2)
    
    # 创建时间窗口
    train_time_windows = fn.create_time_windows(train_time_info, seq_l, pre_l)
    
    # 创建数据集和 loader
    device = torch.device('cpu')
    train_dataset = fn.CreateDataset(train_demand, train_price, seq_l, pre_l, device)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=False, drop_last=True)
    
    # batchify time windows
    train_time_batches = fn.batchify_time_windows(train_time_windows, bs, drop_last=True)
    
    # 检查 batch 数量是否匹配
    assert len(train_time_batches) == len(train_loader), f"Batch count mismatch: time {len(train_time_batches)} vs data {len(train_loader)}"
    
    # 检查几个 batch 的对应关系
    nodes = train_demand.shape[1]
    for j, data in enumerate(train_loader):
        if j >= num_batches_to_check:
            break
        demand_batch, price_batch, label_batch = data  # demand_batch: [bs, nodes, seq_l] ? 根据代码是 [bs, nodes, seq_l]
        ts_batch = train_time_batches[j]  # list of list, len=bs, each len=seq_l
        
        for sample in range(bs):
            start_i = j * bs + sample
            for t in range(seq_l):
                # 检查时间
                original_time = train_time_info[start_i + t]
                batch_time = ts_batch[sample][t]
                assert original_time == batch_time, f"Time mismatch at batch {j}, sample {sample}, t {t}: {original_time} != {batch_time}"
                
                # 检查数据 (随机选一个 node)
                node = np.random.randint(0, nodes)
                original_demand = train_demand[start_i + t, node]
                batch_demand = demand_batch[sample, node, t].item()
                assert np.isclose(original_demand, batch_demand), f"Data mismatch at batch {j}, sample {sample}, t {t}, node {node}: {original_demand} != {batch_demand}"
    
    print("All checks passed! The correspondence between time and data in batches is accurate.")

if __name__ == '__main__':
    test_batch_correspondence()
