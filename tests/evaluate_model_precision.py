# evaluate_prediction_accuracy.py

import os
import sys
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import utils.model_training.training_utils as fn


def main():
    print("=== Evaluating Prediction Accuracy for Earliest Time Step ===")

    # ----------------------------
    # 1. 加载必要数据
    # ----------------------------
    ev_requests = pd.read_csv('data/ev_requests/ev_requests_new_v6_with_vot.csv')
    _, durations, prc, _, cap = fn.read_dataset_as_df()

    # 确保 cap 是 Series，索引为 station_id（字符串或 int）
    if not isinstance(cap, pd.Series):
        raise ValueError("cap should be a pandas Series")
    cap = cap.astype(float)

    # ----------------------------
    # 2. 确定 earliest_time（对齐到整点）
    # ----------------------------
    earliest_time_str = ev_requests['request_time'].min()
    earliest_time = datetime.strptime(earliest_time_str, '%Y-%m-%d %H:%M:%S')
    target_hour = earliest_time.replace(minute=0, second=0, microsecond=0)
    target_time_str = target_hour.strftime('%Y-%m-%d %H:%M:%S')

    print(f"Target prediction time: {target_time_str}")

    # 检查该时间是否在 durations 中
    if target_time_str not in durations.index:
        raise ValueError(f"Target time {target_time_str} not found in durations index")

    # ----------------------------
    # 3. 构建输入：前24小时（注意：原脚本用的是24个点，每小时1个）
    # ----------------------------
    input_times = [
        (target_hour - timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S')
        for i in range(1, 25)  # t-1, t-2, ..., t-24
    ]

    # 检查所有输入时间是否都在 durations 和 prc 中
    missing_in_dura = [t for t in input_times if t not in durations.index]
    missing_in_prc = [t for t in input_times if t not in prc.index]
    if missing_in_dura:
        raise ValueError(f"Missing times in durations: {missing_in_dura}")
    if missing_in_prc:
        raise ValueError(f"Missing times in prc: {missing_in_prc}")

    input_durations = durations.loc[input_times].astype(float)
    input_price = prc.loc[input_times].astype(float)

    # ----------------------------
    # 4. 加载模型并预测
    # ----------------------------
    model_path = "data/results/SZH-EVCDP/dura_pag_s2_on_pretrain_results/dura_pag_s2_on_pretrain_1_bs32_completed.pt"
    model = torch.load(model_path, map_location='cpu')  # 改为 cpu 以防无 GPU
    model.eval()

    # 转为张量: (1, num_stations, 24)
    durations_tensor = torch.tensor(input_durations.values, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
    price_tensor = torch.tensor(input_price.values, dtype=torch.float32).unsqueeze(0).transpose(1, 2)

    with torch.no_grad():
        pred_normalized = model(durations_tensor, price_tensor)
        pred_normalized = pred_normalized.squeeze().cpu().numpy()  # (num_stations,)

    station_ids = list(input_durations.columns.astype(int))  # str or int

    # ----------------------------
    # 5. 还原预测值：真实总占用时长（小时）
    # ----------------------------
    # print(station_ids)
    # print(cap.index)
    cap_for_stations = cap.loc[station_ids].astype(float).values
    predicted_total_duration = pred_normalized

    # 注意：此处保留你原逻辑中的 ×5，但请确认是否合理（原脚本如此）

    # ----------------------------
    # 6. 获取真实值（ground truth）
    # ----------------------------
    station_ids = [str(i) for i in station_ids]
    true_values = durations.loc[target_time_str, station_ids].astype(float).values

    # ----------------------------
    # 7. 只保留非 NaN 且 cap > 0 的站点（避免除零或无效值）
    # ----------------------------
    valid_mask = (~np.isnan(true_values)) & (~np.isnan(predicted_total_duration)) & (cap_for_stations > 0) & (true_values > 0)
    y_true = true_values[valid_mask]
    y_pred = predicted_total_duration[valid_mask]

    if len(y_true) == 0:
        raise ValueError("No valid stations for evaluation.")

    print(f"Evaluating on {len(y_true)} valid stations.")

    # ----------------------------
    # 8. 计算精度指标
    # ----------------------------
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE（避免除零）
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100
    accuracy = 100 - mape  # 平均精度（百分比）

    print("\n=== Prediction Accuracy ===")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"Accuracy (1 - MAPE): {accuracy:.2f}%")

    # # 可选：保存结果到文件
    # result = {
    #     "target_time": target_time_str,
    #     "num_stations": len(y_true),
    #     "MAE": mae,
    #     "RMSE": rmse,
    #     "R2": r2,
    #     "MAPE_%": mape,
    #     "Accuracy_%": accuracy
    # }
    # pd.DataFrame([result]).to_csv("prediction_accuracy.csv", index=False)
    # print("\nResults saved to prediction_accuracy.csv")


if __name__ == "__main__":
    main()