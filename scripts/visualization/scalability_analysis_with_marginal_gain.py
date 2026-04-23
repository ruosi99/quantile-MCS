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
base_paths = {
    50: "data/greedy_algorithm/with_pred_pag_informer_best_mcs_50/metrics.txt",
    100: "data/greedy_algorithm/with_pred_pag_informer_best_mcs_100/metrics.txt",
    150: "data/greedy_algorithm/with_pred_pag_informer_best_mcs_150/metrics.txt",
    200: "data/greedy_algorithm/with_pred_pag_informer_best_mcs_200/metrics.txt",
    250: "data/greedy_algorithm/with_pred_pag_informer_best_mcs_250/metrics.txt",
    300: "data/greedy_algorithm/with_pred_pag_informer_best_mcs_300/metrics.txt",
    350: "data/greedy_algorithm/with_pred_pag_informer_best_mcs_350/metrics.txt",
    400: "data/greedy_algorithm/with_pred_pag_informer_best_mcs_400/metrics.txt",
    450: "data/greedy_algorithm/with_pred_pag_informer_best_mcs_450/metrics.txt",
    500: "data/greedy_algorithm/with_pred_pag_informer_best_mcs_500/metrics.txt",
}

# ============================
# 2. 数据读取 (保持不变)
# ============================
sorted_mcs_counts = sorted(base_paths.keys())
mcs_labels = []
waiting_means = []
completion_mcs = []
completion_total = []
waiting_distributions = []

print("正在读取数据...")
box_mean_values = []
for mcs_count in sorted_mcs_counts:
    txt_path = base_paths[mcs_count]
    csv_path = txt_path.replace("metrics.txt", "ev_requests_with_assignments.csv")
    mcs_labels.append(str(mcs_count))
    
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
# 图 1: 稳定性分析 (Box Plot - 学术蓝版)
# ============================
fig1, ax1 = plt.subplots(figsize=(11, 7)) # 画布稍微加大

# 设置箱线图样式
boxprops = dict(linestyle='-', linewidth=1.5, color=color_box_edge, facecolor=color_box_fill)
medianprops = dict(linestyle='-', linewidth=0, color=color_box_fill) # 隐藏中位数线，或者设为同色
flierprops = dict(marker='o', markerfacecolor='#B0B0B0', markersize=3, alpha=0.5, markeredgecolor='none')
whiskerprops = dict(linestyle='-', linewidth=1.5, color=color_box_edge)
capprops = dict(linestyle='-', linewidth=1.5, color=color_box_edge)

# 绘制
bp = ax1.boxplot(waiting_distributions, patch_artist=True,
                 labels=mcs_labels,
                 boxprops=boxprops,
                 medianprops=medianprops,
                 flierprops=flierprops,
                 whiskerprops=whiskerprops,
                 capprops=capprops,
                 widths=0.6,          # 箱子稍微宽一点，显得饱满
                 showfliers=False)    # 依然建议隐藏离群点以突出分布主体

# 叠加均值连线 (对比色)
x_indices = np.arange(1, len(mcs_labels) + 1)
print(box_mean_values)
line_mean, = ax1.plot(x_indices, box_mean_values, color=color_mean_line, marker='o', 
                      linestyle='--', linewidth=2.5, markersize=8, label='Mean Wait Time')

# 标注与美化
ax1.set_xlabel('Number of MCS', fontweight='bold', fontsize=14)
ax1.set_ylabel('Waiting Time Distribution (min)', fontweight='bold', fontsize=14)
# ax1.set_title('Stability Analysis: Waiting Time Distribution vs. Fleet Size', fontweight='bold', fontsize=15, pad=15)

# 高亮最优区域 (300-350 MCS)
ax1.axvspan(6 - 0.45, 7 + 0.45, color=color_optimal, alpha=0.5, label='Optimal Zone')

# ---- 关键修改：放大图例 ----
# 创建一个自定义的 Patch 来代表箱体颜色
patch_box = mpatches.Patch(facecolor=color_box_fill, edgecolor=color_box_edge, label='Wait Time Dist. (IQR)')
patch_opt = mpatches.Patch(facecolor=color_optimal, alpha=0.5, label='Optimal Deployment Zone')

ax1.legend(handles=[patch_box, line_mean, patch_opt], 
           loc='upper right', 
           fontsize=13,            # 字号加大
           frameon=True, 
           framealpha=0.9, 
           edgecolor='#cccccc',
           fancybox=True,          # 圆角图例
           shadow=True)            # 图例阴影

fig1.tight_layout()
save_path1 = os.path.join(output_dir, "scalability_box_plot_blue_waiting_time.png")
plt.savefig(save_path1, dpi=400, bbox_inches="tight")
print(f"图1 (学术蓝版) 已保存: {save_path1}")


# ============================
# 图 2: 成本效益分析 (保持之前的逻辑，微调配色以统一风格)
# ============================
# 使用配套的蓝色系
color_bar_mcs = '#10739E'   # 深一点的学术蓝
color_bar_tot = '#B1DDF0'   # 浅一点的学术蓝

fig2, ax_bar = plt.subplots(figsize=(11, 7))
ax_line = ax_bar.twinx()

x = np.arange(len(mcs_labels))
width = 0.35

# 柱状图
bar1 = ax_bar.bar(x - width/2, completion_mcs, width, label='Completion Rate (MCS)', color=color_bar_mcs, alpha=0.9, zorder=2)
bar2 = ax_bar.bar(x + width/2, completion_total, width, label='Completion Rate (Total)', color=color_bar_tot, alpha=0.9, zorder=2)

# 折线图
line1 = ax_line.plot(x, waiting_means, color=color_mean_line, marker='D', linewidth=3, markersize=9, label='Avg. Waiting Time', zorder=3)

# 轴设置
ax_bar.set_xlabel('Number of MCS', fontweight='bold', fontsize=14)
ax_bar.set_ylabel('Completion Rate', fontweight='bold', fontsize=14)
ax_bar.set_ylim(0, 1.1)
ax_bar.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(mcs_labels, fontsize=12)

ax_line.set_ylabel('Avg. Waiting Time (min)', color=color_mean_line, fontweight='bold', fontsize=14)
ax_line.tick_params(axis='y', labelcolor=color_mean_line, labelsize=12)

# 高亮最优区域
opt_start, opt_end = 5, 6
ax_bar.axvspan(opt_start - 0.45, opt_end + 0.45, color=color_optimal, alpha=0.5, hatch='//', zorder=0)

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

ax_bar.legend(handles1 + handles2 + [patch_opt2], labels1 + labels2 + ['Optimal Zone'], 
              bbox_to_anchor=(0.45, 1.0), loc='best', frameon=True, shadow=True, fontsize=12) # 图例字号12

# ax_bar.set_title('Cost-Benefit Analysis: Service Quality vs. Fleet Size', fontweight='bold', fontsize=15, pad=15)
ax_bar.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)

fig2.tight_layout()
save_path2 = os.path.join(output_dir, "scalability_pareto_blue.png")
plt.savefig(save_path2, dpi=400, bbox_inches="tight")
print(f"图2 (学术蓝版) 已保存: {save_path2}")

plt.show()