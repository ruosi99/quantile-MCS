import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils.model_training.training_utils as fn
from utils.model_training.baselines.modules import Lo, Arima

def main():
    # ==============================
    # 1. 命令行参数解析
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--use_cuda', type=str, required=True, help='是否使用cuda')
    args = parser.parse_args()

    data_dir = args.data_dir
    use_cuda = args.use_cuda
    data_name = data_dir.split('/')[-2]
    out_dir = f"data/results/{data_name}/baselines_results"
    os.makedirs(out_dir, exist_ok=True)

    # ==============================
    # 2. 系统与超参数配置
    # ==============================
    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
    fn.set_seed(seed=2023, flag=True)

    # 超参数（与 train.py 保持一致）
    seq_len = 24
    pred_len = 1

    # ==============================
    # 3. 数据加载与预处理
    # ==============================
    occ, duration, price_raw, distance, time_info = fn.read_dataset_charged(data_dir)
    num_node = distance.shape[0]

    # 选择输入序列（这里假设使用 duration，如 train.py 中的 "dura" 模型；如果 occ，可修改）
    input_series = duration  # 或 occ，根据需要


    # 划分数据集（与 train.py 一致）
    train_demand, valid_demand, test_demand = fn.division(input_series, train_rate=0.6, valid_rate=0.2, test_rate=0.2)

    # 对于 baselines，使用 train + valid 作为拟合数据
    train_valid_demand = np.concatenate([train_demand, valid_demand], axis=0)

    # ==============================
    # 4. baselines 评估
    # ==============================
    baselines = ['lo', 'arima']  # LO 和 ARIMA（假设 LARIMA 为 Arima）

    for baseline_name in baselines:
        print(f"评估 baseline: {baseline_name}")

        if baseline_name == 'lo':
            model = Lo(pre_len=pred_len)
        elif baseline_name == 'arima':
            model = Arima(pred_len=pred_len)
        else:
            raise ValueError(f"未知 baseline: {baseline_name}")

        # 预测（baselines 的 predict 方法输入 train_valid_feat [T_train+valid, N], test_feat [T_test, N]）
        preds = model.predict(train_valid_demand, test_demand)

        # 计算指标（与 train.py 一致）
        metrics = fn.metrics_new(test_pre=preds, test_real=test_demand)

        # 保存结果
        result_df = pd.DataFrame([metrics], columns=['MSE', 'RMSE', 'MAPE', 'RAE', 'MAE', 'R2', 'MedAE', 'EVS'])
        result_df.to_csv(
            os.path.join(out_dir, f'{baseline_name}_{pred_len}.csv'),
            encoding='gbk',
            index=False
        )

        print(f"{baseline_name} 结果已保存至: {out_dir}/{baseline_name}_{pred_len}.csv")

    print("所有 baselines 评估完成。")

if __name__ == '__main__':
    main()