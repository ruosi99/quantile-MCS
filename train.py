import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils.model_training.training_utils as fn
import utils.model_training.models as models
import utils.model_training.learner as learner
from utils.model_training.baselines.config import PredictionModel
import utils.model_training.baselines.modules as baselines_modules
from utils.model_training.loss_functions import QuantileLoss
from utils.model_training.conformal import conformal_cqr_calibrate, apply_cqr_interval, interval_metrics
from scripts.visualization.plot_forecasting_results import main_plot_forecasting_results


def quantile_crossing_metrics(pred_q):
    """
    pred_q: numpy array (T, N, Q)
    Returns summary statistics for quantile crossing severity.
    """
    diffs = np.diff(pred_q, axis=-1)
    crossing_mask = diffs < 0
    crossing_rate = float(crossing_mask.any(axis=-1).mean())
    mean_crossing_magnitude = float(np.maximum(-diffs, 0.0).mean())
    mean_crossed_pairs = float(crossing_mask.sum(axis=-1).mean())
    max_crossing_magnitude = float(np.maximum(-diffs, 0.0).max())
    return {
        "crossing_rate": crossing_rate,
        "mean_crossing_magnitude": mean_crossing_magnitude,
        "mean_crossed_pairs": mean_crossed_pairs,
        "max_crossing_magnitude": max_crossing_magnitude,
    }


def main():
    # ==============================
    # 1. 命令行参数解析
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='模型名称')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--load_method', type=str, required=True, help='模型加载方式（如 models.MyModel()）')
    parser.add_argument('--is_train', type=lambda x: str(x).lower() == 'true', default=True, help='是否训练')
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
    parser.add_argument('--is_pre_train', type=lambda x: str(x).lower() == 'true', default=True, help='是否预训练')
    parser.add_argument('--use_cuda', type=str, required=True, help='是否使用cuda')
    parser.add_argument('--quantiles', type=str, default='0.05,0.1,0.2,0.5,0.8,0.9,0.95', help='分位数列表，逗号分隔')
    args = parser.parse_args()

    model_name = args.model_name
    data_dir = args.data_dir
    load_method = args.load_method
    is_train = args.is_train
    is_pre_train = args.is_pre_train
    use_cuda = args.use_cuda
    data_name = data_dir.split('/')[-2]
    out_dir = f"data/results/{data_name}/{model_name}_results"
    os.makedirs(out_dir, exist_ok=True)
    quantiles = [float(x) for x in args.quantiles.split(',')]

    # ==============================
    # 2. 系统与超参数配置
    # ==============================
    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
    fn.set_seed(seed=2023, flag=True)

    # 超参数
    seq_len = 24
    label_len = min(seq_len // 2, seq_len)
    pred_len = 1
    batch_size = args.batch_size if args.batch_size else 32
    pretrain_epochs = 50
    finetune_epochs = 200
    law_list = np.array([-1.48, -0.74])  # 需求价格弹性（建议不超过5个）
    mode = 'completed'  # 'simplified' 或 'completed'

    # ==============================
    # 3. 数据加载与预处理
    # ==============================
    # occ, duration, price_raw, distance, time_info, cap = fn.read_dataset_charged(data_dir)
    occ, duration, price_raw, distance, cap = fn.read_dataset_v2(data_dir)
    adj_sparse = distance.to_sparse().to(device)

    # 选择输入序列（根据模型名决定用 occupancy 还是 duration）
    input_series = duration if "dura" in model_name else occ

    # 划分数据集
    train_demand, valid_demand, calib_demand, test_demand = fn.division(
        input_series, train_rate=0.7, valid_rate=0.1, calib_rate=0.1, test_rate=0.1
    )
    train_price, valid_price, calib_price, test_price = fn.division(
        price_raw, train_rate=0.7, valid_rate=0.1, calib_rate=0.1, test_rate=0.1
    )

    # 创建数据集与 DataLoader
    train_dataset = fn.CreateDataset(train_demand, train_price, seq_len, pred_len, device)
    calib_dataset = fn.CreateDataset(calib_demand, calib_price, seq_len, pred_len, device)
    valid_dataset = fn.CreateDataset(valid_demand, valid_price, seq_len, pred_len, device)
    test_dataset = fn.CreateDataset(test_demand, test_price, seq_len, pred_len, device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    calib_loader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # ==============================
    # 4. 模型与优化器初始化
    # ==============================
    model = eval(load_method)  # 注意：eval 有安全风险，生产环境建议用注册机制替代
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss() if "quantile" not in model_name else QuantileLoss(quantiles=quantiles)
    loss_fn = loss_fn.to(device)
    best_valid_loss = float('inf')

    # ==============================
    # 5. 训练流程
    # ==============================
    if is_train:
        model.train()
        # 预训练阶段
        if is_pre_train:
            if mode == 'simplified':
                model = learner.fast_learning(
                    law_list, model, model_name, pretrain_epochs, batch_size,
                    train_demand, train_price, seq_len, pred_len, device, distance, label_len
                )
            elif mode == 'completed':
                model = learner.physics_informed_meta_learning(
                    law_list, model, model_name, pretrain_epochs, batch_size,
                    train_demand, train_price, seq_len, pred_len, 
                    device, distance, label_len, out_dir = out_dir, quantiles = quantiles
                )
            else:
                print("警告：未知 mode，跳过预训练。")

        # 微调阶段
        for epoch in tqdm(range(finetune_epochs), desc='Fine-tuning', position=0, leave=True):
            model.train()
            for j, (demand, price, label) in enumerate(
                tqdm(train_loader, desc=f'Epoch {epoch+1}/{finetune_epochs} [Train]', position=1, leave=False)
            ):
                demand, price, label = demand.to(device), price.to(device), label.to(device)
                optimizer.zero_grad()
                pred = model(demand, price)
                
                loss = loss_fn(pred, label)
                loss.backward()
                optimizer.step()

            # 验证阶段
            model.eval()
            total_valid_loss = 0.0
            with torch.no_grad():
                for j, (demand, price, label) in enumerate(
                    tqdm(valid_loader, desc=f'Epoch {epoch+1}/{finetune_epochs} [Valid]', position=2, leave=False)
                ):
                    demand, price, label = demand.to(device), price.to(device), label.to(device)

                    pred = model(demand, price)

                    total_valid_loss += loss_fn(pred, label).item()

            # 保存最佳模型
            if total_valid_loss < best_valid_loss:
                best_valid_loss = total_valid_loss
                checkpoint_path = f'{out_dir}/{model_name}_{pred_len}_bs{batch_size}_{mode}.pt'
                torch.save(model, checkpoint_path)

    # ==============================
    # 6. 加载最佳模型并测试
    # ==============================
    node_variance = np.var(train_demand, axis=0)
    checkpoint_path = f'{out_dir}/{model_name}_{pred_len}_bs{batch_size}_{mode}.pt'
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.eval()

    delta_list = [0.1, 0.2, 0.4]
    
    # ---- 6.1 模型推理 (只执行一次，获取所有分位数预测结果) ----
    predict_list = []
    label_list = []

    print("正在执行模型推理...")
    with torch.no_grad():
        for j, (demand, price, label) in enumerate(
            tqdm(test_loader, total=len(test_loader), desc='Testing Inference', position=0, leave=True)
        ):
            demand, price, label = demand.to(device), price.to(device), label.to(device)
            pred_q = model(demand, price)  # (B,N,Q)
            predict_list.append(pred_q.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

    # 合并结果并还原尺度
    # 注意：这里对 cap 的操作是为了避免影响后续校准函数中的 cap 使用
    cap_scale = cap.reshape(1, -1) if "quantile" not in model_name else cap.reshape(1, -1, 1)
    predict_q = np.concatenate(predict_list, axis=0) * cap_scale   # (T,N,Q)
    label_array = np.concatenate(label_list, axis=0) * cap_scale.reshape(1, -1)   # (T,N)

    # 保存通用的预测结果和真实值
    np.save(os.path.join(out_dir, 'predict_quantiles.npy'), predict_q)
    np.save(os.path.join(out_dir, 'label_list.npy'), label_array)

    crossing_df = pd.DataFrame([quantile_crossing_metrics(predict_q)])
    crossing_df.to_csv(
        os.path.join(out_dir, 'quantile_crossing_metrics.csv'),
        index=False
    )

    # 预先计算点预测指标 (使用中位数 0.5 分位数)
    q50_idx = quantiles.index(0.5)
    point_pred_common = predict_q if "quantile" not in model_name else predict_q[:, :, q50_idx]
    metrics_point = fn.metrics_new(test_pre=point_pred_common, test_real=label_array)
    
    # 保存点预测指标 (只需保存一次)
    result_df = pd.DataFrame([metrics_point], columns=['MSE', 'RMSE', 'MAPE', 'RAE', 'MAE', 'R2', 'MedAE', 'EVS'])
    result_df.to_csv(
        os.path.join(out_dir, f'{model_name}_{pred_len}bs{batch_size}_point_q50.csv'),
        encoding='gbk', index=False
    )
    np.save(os.path.join(out_dir, 'predict_point_q50.npy'), point_pred_common)
    if "quantile" not in model_name or len(quantiles) == 1:
        # 非分位数预测，不用计算区间指标；也不用画区间指标相关的图
        print("非分位数预测，不用计算区间指标")
        return

    # ---- 6.2 循环处理不同的 Delta (校准 + 计算区间指标) ----
    print(f"开始处理 Delta 列表: {delta_list}")
    
    for delta in delta_list:
        print(f"\n--- Processing delta={delta} ---")

        alpha = delta / 2.0
        li = quantiles.index(alpha)
        ui = quantiles.index(1.0 - alpha)
        raw_point = predict_q[:, :, q50_idx]
        raw_L = predict_q[:, :, li]
        raw_U = predict_q[:, :, ui]
        raw_metrics = interval_metrics(label_array, raw_L, raw_U, point_pred=raw_point, delta=delta)
        pd.DataFrame([raw_metrics]).to_csv(
            os.path.join(out_dir, f'{model_name}_{pred_len}bs{batch_size}_raw_interval_delta{delta}.csv'),
            encoding='gbk', index=False
        )
        
        # 1. Conformal Calibration (CQR)
        # 注意：这里传入原始的 cap，因为校准函数内部可能有其自身的处理逻辑
        s_hat, group_ids = conformal_cqr_calibrate(
            model,
            calib_loader,
            quantiles,
            delta=delta,
            device=device,
            capacity=cap,  # 传入原始 cap
            node_variance=node_variance,
            n_groups=6
        )
        print(f"[CQR] delta={delta}, s_hat={s_hat}")
        np.save(os.path.join(out_dir, f'cqr_s_hat_delta{delta}.npy'), np.array([s_hat], dtype=np.float32))

        # 2. Apply calibrated interval
        point_pred, L_cal, U_cal = apply_cqr_interval(
            predict_q,
            quantiles,
            s_hat,
            delta=delta,
            group_ids=group_ids
        )

        # 保存区间边界
        np.save(os.path.join(out_dir, f'cqr_L_delta{delta}.npy'), L_cal)
        np.save(os.path.join(out_dir, f'cqr_U_delta{delta}.npy'), U_cal)

        # 3. 计算区间指标
        metrics_int = interval_metrics(label_array, L_cal, U_cal, point_pred=point_pred, delta=delta)

        # 保存区间指标 (文件名包含 delta)
        interval_df = pd.DataFrame([metrics_int])
        interval_df.to_csv(
            os.path.join(out_dir, f'{model_name}_{pred_len}bs{batch_size}_cqr_interval_delta{delta}.csv'),
            encoding='gbk', index=False
        )
        print(f"Delta {delta} metrics saved: {metrics_int}")

    print("所有 Delta 测试完成。")

    # ==============================
    # 7. 可视化结果
    # ==============================
    main_plot_forecasting_results(out_dir)


if __name__ == '__main__':
    main()
