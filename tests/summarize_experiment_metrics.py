import os
import re
import pandas as pd
import numpy as np

# ============================
# 1. 实验配置与文件路径
# ============================
exp_files = {
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
# 2. 解析 txt 的函数
# ============================
def parse_metric_file(path):
    """
    从文本文件中解析所有指标。
    返回 {metric_name: value (as float)}
    """
    metric_data = {}

    if not os.path.exists(path):
        print(f"Warning: {path} 不存在，跳过。")
        return metric_data

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 匹配格式： "指标名: 数值 单位"
    # 支持单位：分钟、元、%（其他单位保留原样，但本例中只有这三种）
    pattern = r"^(.*?):\s*([\d\.]+)\s*(分钟|元|%)?\s*$"
    lines = content.strip().split('\n')
    
    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            name = match.group(1).strip()
            value = float(match.group(2))
            unit = match.group(3)

            # 百分比转换为 0~1 之间的小数（可选，也可保留原%值；这里按你之前逻辑转为小数）
            if unit == "%":
                value = value / 100.0

            metric_data[name] = value

    return metric_data

# ============================
# 3. 收集所有实验的完整指标
# ============================
all_results = []

for mcs_num, path in exp_files.items():
    metrics = parse_metric_file(path)
    metrics["MCS Num"] = mcs_num
    all_results.append(metrics)

# 转为 DataFrame
df = pd.DataFrame(all_results)

# 将 "Noise Level" 设为第一列
cols = ["MCS Num"] + [col for col in df.columns if col != "MCS Num"]
df = df[cols]

# 按 Noise Level 排序（确保 0.00, 0.05, ..., 0.30 顺序）
df["MCS Num"] = pd.Categorical(df["MCS Num"], 
                                   categories=sorted(exp_files.keys(), key=float),
                                   ordered=True)
df = df.sort_values("MCS Num").reset_index(drop=True)

# ============================
# 4. 导出为 Excel
# ============================
output_file = "data/scalability_experiment_metrics_summary.xlsx"
df.to_excel(output_file, index=False, sheet_name="All Metrics")

print(f"✅ 所有指标已成功导出到 {output_file}")