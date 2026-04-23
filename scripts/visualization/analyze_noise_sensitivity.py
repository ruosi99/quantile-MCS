import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches

# ============================
# 1. 实验文件路径配置
# ============================
exp_files = {
    "0.00": "data/greedy_algorithm/with_pred_pag_informer_best_noise_0.00_mcs_300/metrics.txt",
    "0.05": "data/greedy_algorithm/with_pred_pag_informer_best_noise_0.05_mcs_300/metrics.txt",
    "0.10": "data/greedy_algorithm/with_pred_pag_informer_best_noise_0.10_mcs_300/metrics.txt",
    "0.15": "data/greedy_algorithm/with_pred_pag_informer_best_noise_0.15_mcs_300/metrics.txt",
    "0.20": "data/greedy_algorithm/with_pred_pag_informer_best_noise_0.20_mcs_300/metrics.txt",
    "0.25": "data/greedy_algorithm/with_pred_pag_informer_best_noise_0.25_mcs_300/metrics.txt",
    "0.30": "data/greedy_algorithm/with_pred_pag_informer_best_noise_0.30_mcs_300/metrics.txt",
}

# ============================
# 2. 数据读取 (保持不变)
# ============================
sorted_noise_levels = sorted(exp_files.keys())
noise_labels = []
waiting_means = []
completion_mcs = []
completion_total = []
waiting_distributions = []

print("正在读取数据...")
box_mean_values = []
for noise_level in sorted_noise_levels:
    txt_path = exp_files[noise_level]
    csv_path = txt_path.replace("metrics.txt", "ev_requests_with_assignments.csv")
    noise_labels.append(str(noise_level))
    
    # 读取 Metrics
    metrics = {}
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        pattern = r"(.*?):\s*([\d\.]+)\s*(分钟|元|%|)?"
        for line, val, unit in re.findall(pattern, content):
            val = float(val)
            if unit == "%": val /= 100.0
            metrics[line.strip()] = val
    
    mean_val = metrics.get("平均等待时间 (总计)", 0)
    waiting_means.append(mean_val)
    completion_mcs.append(metrics.get("充电任务完成率 (MCS)", 0))
    completion_total.append(metrics.get("充电任务完成率 (总计)", 0))
    
    # 读取 CSV (waiting_time列)
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if 'waiting_time' in df.columns:
                waiting_distributions.append(df['waiting_time'].dropna().tolist())
                box_mean_values.append(np.mean(df['waiting_time'].dropna().tolist()))
            else:
                waiting_distributions.append([])
                box_mean_values.append(0)
        except:
            waiting_distributions.append([])
            box_mean_values.append(0)
    else:
        # 容错：无文件时生成模拟数据
        mock = np.random.normal(loc=mean_val if mean_val>0 else 80, scale=20, size=100)
        waiting_distributions.append(np.maximum(mock, 0))

# ============================
# 3. 绘图配色与样式 (升级版)
# ============================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.grid': True,
    'grid.alpha': 0.4,           # 网格稍微明显一点
    'grid.linestyle': '--',
    'font.size': 12              # 全局字体加大
})

# ---- 学术蓝配色方案 ----
color_box_fill = '#DAE8FC'   # 填充：低饱和度淡蓝 (Light Cool Blue)
color_box_edge = '#6C8EBF'   # 边框：学术深蓝灰 (Glacial Blue Dark)
color_mean_line = '#B85450'  # 均值线：低饱和砖红 (Muted Brick Red)
color_optimal = '#D5E8D4'    # 最优区：极淡的鼠尾草绿 (Sage Green Light)
color_optimal_edge = '#82B366' # 最优区边框

output_dir = "data/plots"
if not os.path.exists(output_dir): os.makedirs(output_dir)

# ============================
# 柱线结合
# ============================
# 使用配套的蓝色系
color_bar_mcs = '#10739E'   # 深一点的学术蓝
color_bar_tot = '#B1DDF0'   # 浅一点的学术蓝

fig2, ax_bar = plt.subplots(figsize=(11, 7))
ax_line = ax_bar.twinx()

x = np.arange(len(noise_labels))
width = 0.35

# 柱状图
bar1 = ax_bar.bar(x - width/2, completion_mcs, width, label='Completion Rate (MCS)', color=color_bar_mcs, alpha=0.9, zorder=2)
bar2 = ax_bar.bar(x + width/2, completion_total, width, label='Completion Rate (Total)', color=color_bar_tot, alpha=0.9, zorder=2)

# 折线图
line1 = ax_line.plot(x, waiting_means, color=color_mean_line, marker='D', linewidth=3, markersize=9, label='Avg. Waiting Time', zorder=3)

# 轴设置
ax_bar.set_xlabel('Noise Level', fontweight='bold', fontsize=14)
ax_bar.set_ylabel('Completion Rate', fontweight='bold', fontsize=14)
ax_bar.set_ylim(0, 1.1)
ax_bar.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(noise_labels, fontsize=12)     

ax_line.set_ylabel('Avg. Waiting Time (min)', color=color_mean_line, fontweight='bold', fontsize=14)
ax_line.tick_params(axis='y', labelcolor=color_mean_line, labelsize=12)

# 手动设置 x 轴和 y 轴范围
ax_bar.set_ylim(0.78, 0.92)                       # 左 y 轴（完成率）范围，0% ~ 100%
ax_line.set_ylim(76, 82)                        # 右 y 轴（等待时间）范围，单位：分钟，按需调整

# # 高亮最优区域
# opt_start, opt_end = 5, 6
# ax_bar.axvspan(opt_start - 0.45, opt_end + 0.45, color=color_optimal, alpha=0.5, hatch='//', zorder=0)

# 注释
# ax_line.annotate('Optimal Deployment\n(Sweet Spot)', 
#                  xy=((opt_start + opt_end)/2, waiting_means[5]), 
#                  xytext=((opt_start + opt_end)/2, waiting_means[5] + 40),
#                  color='#2D4B17', fontsize=12, fontweight='bold', ha='center',
#                  arrowprops=dict(arrowstyle='->', color='#2D4B17', lw=2))

# ax_line.annotate('Diminishing Returns', 
#                  xy=(8, waiting_means[8]), 
#                  xytext=(7, waiting_means[8] + 25),
#                  arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2", color='#444'),
#                  fontsize=11, style='italic', color='#444', 
#                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.8))

# 图例
handles1, labels1 = ax_bar.get_legend_handles_labels()
handles2, labels2 = ax_line.get_legend_handles_labels()
patch_opt2 = mpatches.Patch(facecolor=color_optimal, alpha=0.5, hatch='//', label='Optimal Zone')

ax_bar.legend(handles1 + handles2 + [patch_opt2], labels1 + labels2, 
              bbox_to_anchor=(0.45, 1.0), loc='best', frameon=True, shadow=True, fontsize=12) # 图例字号12

# ax_bar.set_title('Cost-Benefit Analysis: Service Quality vs. Fleet Size', fontweight='bold', fontsize=15, pad=15)
ax_bar.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)

fig2.tight_layout()
save_path2 = os.path.join(output_dir, "noise_comp.png")
plt.savefig(save_path2, dpi=400, bbox_inches="tight")
print(f"图2 (学术蓝版) 已保存: {save_path2}")

plt.show()