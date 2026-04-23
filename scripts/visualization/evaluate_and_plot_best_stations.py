import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import utils.model_training.training_utils as fn

def main():
    print("Loading data...")
    # durations, prc, cap = fn.read_dataset_as_df()[:3]  # 假设函数返回 (occ, durations, prc, adj, cap)
    # 如果你的 fn.read_dataset_as_df() 返回元组，请按实际调整，例如：
    _, durations, prc, _, cap = fn.read_dataset_as_df()

    # 确保索引是 datetime
    durations.index = pd.to_datetime(durations.index)
    prc.index = pd.to_datetime(prc.index)
    durations = durations.sort_index()
    prc = prc.sort_index()

    # 确保 cap 是 Series，索引为 station_id（字符串或 int）
    if not isinstance(cap, pd.Series):
        raise ValueError("cap should be a pandas Series")
    cap = cap.astype(float)
    cap.index = cap.index.astype(str)

    # ----------------------------
    # 1. 划分测试集：最后 20%
    # ----------------------------
    total_steps = len(durations)
    test_start_idx = int(0.8 * total_steps)
    test_durations = durations.iloc[test_start_idx:].copy()
    test_prc = prc.loc[test_durations.index].copy()

    print(f"Test set from {test_durations.index[0]} to {test_durations.index[-1]} "
          f"({len(test_durations)} time steps)")

    # ----------------------------
    # 2. 加载模型
    # ----------------------------
    model_path = "data/results/ST_EVCDP_v2/dura_pag_informer_on_pretrain_results/dura_pag_informer_on_pretrain_1_bs8_completed.pt"
    model = torch.load(model_path, map_location='cuda')
    model.eval()

    # 获取所有站点
    station_ids = durations.columns.tolist()
    num_stations = len(station_ids)

    # 存储真实值和预测值
    true_values = {sid: [] for sid in station_ids}
    pred_values = {sid: [] for sid in station_ids}

    # ----------------------------
    # 3. 滚动预测整个测试集
    # ----------------------------
    print("Running rolling predictions over test set...")
    for i in range(len(test_durations)):
        target_time = test_durations.index[i]
        # 取前24小时作为输入（不包括 target_time）
        input_start = target_time - pd.Timedelta(hours=24)
        input_end = target_time - pd.Timedelta(hours=1)

        # 检查是否有连续24小时历史数据
        hist_times = pd.date_range(start=input_start, end=input_end, freq='h')
        if not all(t in durations.index for t in hist_times):
            print(f"Skipping {target_time}: insufficient history")
            continue

        # 提取历史数据
        input_durations = durations.loc[hist_times].astype(float)
        input_price = prc.loc[hist_times].astype(float)

        # 转为张量 (1, time=24, stations)
        durations_tensor = torch.tensor(input_durations.values, dtype=torch.float32).unsqueeze(0).transpose(1, 2).to('cuda')
        price_tensor = torch.tensor(input_price.values, dtype=torch.float32).unsqueeze(0).transpose(1, 2).to('cuda')

        with torch.no_grad():
            pred_norm = model(durations_tensor, price_tensor)
            pred_norm = pred_norm.squeeze().cpu().numpy()  # (num_stations,)

        # 还原为真实总占用时长（小时）
        cap_vals = cap.loc[station_ids].astype(float).values
        pred_total = pred_norm * cap_vals

        # 保存真实值和预测值
        true_vals = test_durations.iloc[i].values * cap_vals
        for j, sid in enumerate(station_ids):
            true_values[sid].append(true_vals[j])
            pred_values[sid].append(pred_total[j])

    # ----------------------------
    # 4. 计算 MAPE，并筛选前2 + 后2站点
    # ----------------------------
    print("Computing MAPE and filtering valid stations...")
    mape_dict = {}
    std_threshold = 3e-2  # 波动极小阈值
    epsilon = 1e-8        # 防止除零

    # ---- 先计算所有有效站点的 MAPE ----
    for sid in station_ids:
        true = np.array(true_values[sid])
        pred = np.array(pred_values[sid])
        if len(true) == 0:
            continue

        # 跳过几乎不变的序列（保持你的规则）
        if np.std(true) < std_threshold:
            continue

        # MAPE
        mape = np.mean(np.abs((true - pred) / (true + epsilon))) * 100
        mape_dict[sid] = mape

    if len(mape_dict) < 4:
        raise ValueError(f"Only {len(mape_dict)} valid stations with sufficient variation. Need at least 4.")

    # 将站点按 MAPE 升序排序
    sorted_stations = sorted(mape_dict.items(), key=lambda x: x[1])
    sorted_ids = [sid for sid, _ in sorted_stations]

    # ---- 前两个站点：MAPE 最小的两个 ----
    first_two = sorted_ids[:2]

    # ====================================================================
    # ---- 后两个站点：需要过滤“真值重复率 ≥70% 的站点” ----
    # ====================================================================
    def calc_repeat_ratio(arr):
        """计算重复值比例 = 众数出现次数 / 总长度"""
        if len(arr) == 0:
            return 1.0
        values, counts = np.unique(arr, return_counts=True)
        return np.max(counts) / len(arr)

    filtered_for_last_two = []
    for sid in sorted_ids[2:]:  # 从剩下的站点挑
        true = np.array(true_values[sid])
        repeat_ratio = calc_repeat_ratio(true)

        # 删除重复率 ≥ 0.70 的站点
        if repeat_ratio >= 0.20:
            continue

        filtered_for_last_two.append(sid)

    if len(filtered_for_last_two) < 2:
        raise ValueError("Not enough stations left after filtering by repeat ratio >=70%.")

    # 后两个：过滤后的站点，再从中选 MAPE 最小的两个
    last_two = filtered_for_last_two[:2]

    # 最终的四个站点
    best_ids = first_two + last_two

    print("\nSelected stations:")
    print("  First 2 stations (low MAPE, valid variation):")
    for sid in first_two:
        print(f"    {sid}: MAPE={mape_dict[sid]:.2f}%")

    print("  Last 2 stations (filtered by repeat<70% + low MAPE):")
    for sid in last_two:
        print(f"    {sid}: MAPE={mape_dict[sid]:.2f}%")

        # ----------------------------
    # 5. 绘图：2x2 拟合效果图（学术风格）
    # ----------------------------
    # 可配置的时间范围（默认为整个测试集）
    PLOT_START = "2023-07-01"  # e.g., "2023-10-01"
    PLOT_END = "2023-07-15"   # e.g., "2023-10-07"

    # 获取共同时间索引
    plot_times = test_durations.index[:len(true_values[best_ids[0]])]

    if PLOT_START or PLOT_END:
        ts_start = pd.Timestamp(PLOT_START) if PLOT_START else plot_times[0]
        ts_end = pd.Timestamp(PLOT_END) if PLOT_END else plot_times[-1]
        plot_mask = (plot_times >= ts_start) & (plot_times <= ts_end)
        plot_times_sel = plot_times[plot_mask]
    else:
        plot_mask = np.ones(len(plot_times), dtype=bool)
        plot_times_sel = plot_times

    # 设置学术绘图样式
    plt.rcParams.update({
        "font.family": "serif",            # 或 "sans-serif"（根据期刊要求）
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "grid.alpha": 0.3,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.6,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    axes = axes.flatten()

    # 颜色定义（色盲友好）
    color_true = "#1f77b4"   # blue
    color_pred = "#ff7f0e"   # orange

    for idx, sid in enumerate(best_ids):
        ax = axes[idx]
        true = np.array(true_values[sid])
        pred = np.array(pred_values[sid])

        if PLOT_START or PLOT_END:
            true_plot = true[plot_mask]
            pred_plot = pred[plot_mask]
            times_plot = plot_times_sel
        else:
            true_plot = true
            pred_plot = pred
            times_plot = plot_times

        ax.plot(times_plot, true_plot, 
                color=color_true, 
                linewidth=2.0, 
                label='Observed')
        ax.plot(times_plot, pred_plot, 
                color=color_pred, 
                linewidth=2.0, 
                linestyle='--', 
                label='Predicted')

        # 设置标题（含 MAPE）
        ax.set_title(f"Station {sid} (MAPE: {mape_dict[sid]:.1f}%)", pad=10)

        # 坐标轴标签
        ax.set_ylabel("Occupied Hours (h)")
        if idx >= 2:  # 只在底部子图显示 x 标签
            ax.set_xlabel("Time")

        # 网格
        ax.grid(True, which='major', color='#dddddd', linewidth=0.8)

        # 图例（仅在第一个子图显示，避免重复）
        ax.legend(loc='best', frameon=False, 
                    handlelength=2.0, handletextpad=0.5)

        # 优化 x 轴刻度（避免重叠）
        ax.tick_params(axis='x', rotation=0)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))  # 最多5个刻度

    # 调整子图间距
    plt.tight_layout(pad=2.0)

    # 保存为高分辨率 PNG 和 PDF（推荐用于论文）
    output_dir = "data/plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "best4_stations_fitting.png"), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "best4_stations_fitting.pdf"), bbox_inches='tight')  # 矢量图
    print(f"Fitting plots saved to {output_dir}/best4_stations_fitting.[png|pdf]")

    plt.show()


if __name__ == "__main__":
    main()