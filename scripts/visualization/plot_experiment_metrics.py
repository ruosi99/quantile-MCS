import os
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# ============================
# 1. 在这里填写你的实验配置与文件路径
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
# 2. 解析 txt 的函数
# ============================
def parse_metric_file(path):
    """
    从文本文件中解析关键指标。
    返回 {metric_name: value}
    """
    metric_data = {}

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 通用解析：匹配 "xxx: 12.34" 或 "xxx: 89%"
    pattern = r"(.*?):\s*([\d\.]+)\s*(分钟|元|%|)?"
    for line, val, unit in re.findall(pattern, content):
        line = line.strip()

        # 值：转成 float
        value = float(val)

        # 百分比
        if unit == "%":
            value = value / 100.0

        metric_data[line] = value

    return metric_data

# ============================
# 3. 收集所有实验的关键指标
# ============================
exp_names = []
waiting_total = []
completion_mcs = []
completion_total = []

for exp, path in exp_files.items():
    metrics = parse_metric_file(path)

    exp_names.append(exp)

    waiting_total.append(metrics.get("平均等待时间 (总计)", np.nan))
    completion_mcs.append(metrics.get("充电任务完成率 (MCS)", np.nan))
    completion_total.append(metrics.get("充电任务完成率 (总计)", np.nan))

# ============================
# 4. 绘图（学术标准）
# ============================
plt.style.use("seaborn-v0_8-paper")
# 获取matplotlib默认颜色循环
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

fig, ax1 = plt.subplots(figsize=(10, 6))

# ---- 左轴：Average Waiting Time ----
x = np.arange(len(exp_names))
ax1.plot(
    x, waiting_total,
    marker="o", linestyle="-", linewidth=2, color=colors[1],
    label="Average Waiting Time (Total)"
)
ax1.set_ylabel("Average Waiting Time (min)", fontsize=13)
ax1.set_xlabel("Noise Level", fontsize=13)
ax1.tick_params(axis='y')
ax1.set_xticks(x)
ax1.set_xticklabels(exp_names, rotation=0, ha='center')
ax1.set_ylim(76, 82)
# ---- 右轴：Completion Rates ----
ax2 = ax1.twinx()

ax2.plot(
    x, completion_mcs,
    marker="s", linestyle="--", linewidth=2, color=colors[0],
    label="Completion Rate (MCS)"
)
ax2.plot(
    x, completion_total,
    marker="^", linestyle="--", linewidth=2, color=colors[0],
    label="Completion Rate (Total)"
)
ax2.set_ylabel("Completion Rate", fontsize=13)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# ---- 合并图例 ----
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper left",
    fontsize=12
)

# ---- 网格 & 布局 ----
ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
fig.tight_layout()

# 保存高分辨率版本（论文用）
plt.savefig("data/plots/experiment_comparison_model.png", dpi=400, bbox_inches="tight")

plt.show()