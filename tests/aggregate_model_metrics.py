import os
import pandas as pd
from itertools import product
import glob
import torch
import numpy as np

# 配置根路径和预测目标、模型名称列表
base_dir = r"data\results\ST_EVCDP_v2"
targets = ["volume", "dura"]  # 示例，请替换为你实际的预测目标
models = [
    "lstm", "pag", "fgn", "pag_informer",
]  # 示例，请替换为你实际的模型名
pretrain_suffixes = ["on_pretrain"]  # 如果有多个训练方式
versions = ["1"]  # 如果有多个版本


# 构建所有可能的文件名组合
output_excel_path = "data/results/ST_EVCDP_v2/model_comparison_results.xlsx"
with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
    for target in targets:
        rows = []
        for model, pretrain, ver in product(models, pretrain_suffixes, versions):
            # 构建模糊匹配的搜索模式，不指定batch_size
            search_pattern = os.path.join(base_dir, f"{target}_{model}_{pretrain}_results", f"{target}_{model}*.csv")
            matching_files = glob.glob(search_pattern)
            
            # 取第一个匹配到的文件
            if matching_files:
                filepath = matching_files[0]  # 取第一个匹配的文件
                if os.path.exists(filepath):
                    try:
                        df = pd.read_csv(filepath, header=0)
                        if df.shape[0] == 1:
                            metrics = df.iloc[0].to_dict()
                            # 使用实际找到的文件名（去掉.csv后缀）作为model_config
                            actual_filename = os.path.basename(filepath).replace(".csv", "")
                            row = {"model_config": actual_filename, **metrics}
                            rows.append(row)
                    except Exception as e:
                        print(f"Error reading {filepath}: {e}")
            # 如果没有匹配到文件，则跳过（不执行任何操作）
        
        if rows:
            result_df = pd.DataFrame(rows)
            result_df.to_excel(writer, sheet_name=target, index=False)
            print(f"Saved sheet for target: {target} with {len(rows)} models.")
        else:
            print(f"No valid files found for target: {target}")

print(f"All results saved to {output_excel_path}")