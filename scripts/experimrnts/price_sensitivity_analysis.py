import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import utils.model_training.training_utils as fn

# --- 学术绘图配置 ---
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 14,
    "font.size": 12,
    "legend.fontsize": 11,
    "legend.frameon": True,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "savefig.dpi": 300,
    "axes.grid": True
})

def main():
    # 1. 定义要对比的模型路径
    # 你可以在这里添加任意多个模型
    model_configs = {
        "LSTM": "data/results/ST_EVCDP_v2/dura_lstm_on_pretrain_results/dura_lstm_on_pretrain_1_bs16_completed.pt",
        "FGN": "data/results/ST_EVCDP_v2/dura_fgn_on_pretrain_results/dura_fgn_on_pretrain_1_bs16_completed.pt", # 示例：添加第二个模型
        "PAG": "data/results/ST_EVCDP_v2/dura_pag_on_pretrain_results/dura_pag_on_pretrain_1_bs16_completed.pt",
        "PAG-Informer": "data/results/ST_EVCDP_v2/dura_pag_informer_on_pretrain_results/dura_pag_informer_on_pretrain_1_bs8_completed.pt"
    }

    # ----------------------------
    # 2. 加载基础数据 (所有模型共用)
    # ----------------------------
    print("Loading base data...")
    ev_requests = pd.read_csv('data/ev_requests/ev_requests_new_v6_with_vot.csv')
    occ, durations, prc, adj, cap = fn.read_dataset_as_df()
    cap = cap.astype(float)

    # 准备输入张量
    earliest_time_str = ev_requests['request_time'].min()
    earliest_time = datetime.strptime(earliest_time_str, '%Y-%m-%d %H:%M:%S')
    target_hour = earliest_time.replace(minute=0, second=0, microsecond=0)
    one_hour_before = target_hour - timedelta(hours=1)
    time_intervals = [(one_hour_before - timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S') for i in range(24)]

    input_durations = durations.loc[time_intervals].astype(float)
    base_price = prc.loc[time_intervals].astype(float)
    station_ids = list(input_durations.columns.astype(int))
    cap_values = cap.loc[station_ids].values

    durations_tensor = torch.tensor(input_durations.values, dtype=torch.float32).unsqueeze(0).transpose(1, 2).to('cuda')
    durations_tensor_dup = durations_tensor.repeat(2, 1, 1)

    # ----------------------------
    # 3. 执行多模型推理实验
    # ----------------------------
    perturbations = np.linspace(0.5, 1.5, 11)  # -50% 到 +50%
    all_results = []

    for model_name, path in model_configs.items():
        print(f"Evaluating Model: {model_name}...")
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skipping.")
            continue
            
        model = torch.load(path, map_location='cuda')
        model.eval()

        # 计算该模型下的 Baseline
        with torch.no_grad():
            base_price_tensor = torch.tensor(base_price.values, dtype=torch.float32).unsqueeze(0).transpose(1, 2).to('cuda')
            base_price_tensor_dup = base_price_tensor.repeat(2, 1, 1)
            base_pred = model(durations_tensor_dup, base_price_tensor_dup)[0].cpu().numpy()
            base_demand_sum = (base_pred * cap_values * 8).sum()

        for p in perturbations:
            with torch.no_grad():
                p_price_tensor = (base_price_tensor * p).to('cuda')
                p_price_tensor_dup = p_price_tensor.repeat(2, 1, 1)
                
                p_pred = model(durations_tensor_dup, p_price_tensor_dup)[0].cpu().numpy()
                p_demand = p_pred * cap_values * 8
                p_demand_sum = p_demand.sum()
                
                # 计算站点级变化比例
                base_demand_stations = base_pred * cap_values * 8
                diff = p_demand - base_demand_stations
                neg_ratio = np.mean(diff < -1e-5)

                all_results.append({
                    'Model': model_name,
                    'Price_Change_Pct': (p - 1) * 100,
                    'Demand_Change_Pct': (p_demand_sum - base_demand_sum) / base_demand_sum * 100,
                    'Negative_Response_Ratio': neg_ratio
                })

    results_df = pd.DataFrame(all_results)

    # ----------------------------
    # 4. 绘制对比图 (需求变化百分比)
    # ----------------------------
    fig, ax1 = plt.subplots(figsize=(9, 6))

    # 使用不同颜色和标记
    markers = ['o', 's', '^', 'D']
    colors = sns.color_palette("Set1", n_colors=len(model_configs))

    for i, (model_name, group) in enumerate(results_df.groupby('Model')):
        # 绘制带置信区间的回归趋势 (在学术图中非常加分)
        sns.regplot(
            x='Price_Change_Pct', 
            y='Demand_Change_Pct', 
            data=group,
            ax=ax1,
            label=model_name,
            color=colors[i],
            marker=markers[i % len(markers)],
            scatter_kws={'s': 50, 'alpha': 0.6},
            line_kws={'linewidth': 2, 'linestyle': '--' if i > 0 else '-'}
        )

    # 辅助线
    ax1.axhline(0, color='black', linewidth=1, alpha=0.5)
    ax1.axvline(0, color='black', linewidth=1, alpha=0.5)

    # 图形修饰
    ax1.set_title('Sensitivity Comparison: Price vs. Predicted Demand', fontsize=16, pad=20)
    ax1.set_xlabel('Price Perturbation (%)', fontsize=14)
    ax1.set_ylabel('Total Demand Change (%)', fontsize=14)
    ax1.legend(title="Models", loc='best')
    ax1.grid(True, which='both', linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig('multi_model_sensitivity_analysis.png')
    print("\nMain plot saved: 'multi_model_sensitivity_analysis.png'")

    # ----------------------------
    # 5. 补充图：负向响应站点比例 (展示模型稳健性)
    # ----------------------------
    plt.figure(figsize=(9, 5))
    sns.lineplot(
        data=results_df, 
        x='Price_Change_Pct', 
        y='Negative_Response_Ratio', 
        hue='Model', 
        style='Model', 
        markers=True, 
        markersize=8
    )
    plt.title('Ratio of Stations with Decreased Demand', fontsize=16)
    plt.ylabel('Negative Response Ratio', fontsize=14)
    plt.xlabel('Price Perturbation (%)', fontsize=14)
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig('multi_model_response_ratio.png')
    
    print("Response ratio plot saved: 'multi_model_response_ratio.png'")
    plt.show()

if __name__ == "__main__":
    main()