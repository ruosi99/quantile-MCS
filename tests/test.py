import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

# 设置全局字体为 SimHei (黑体) 或其他中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei'] 微软雅黑 等
plt.rcParams['axes.unicode_minus'] = False   # 解决负号 '-' 显示为方块的问题

def lognormal_params_from_mean_gini(mean_income, gini):
    """
    根据均值和基尼系数，反推出对数正态分布参数 (mu, sigma)
    """
    def gini_equation(sigma):
        return 2 * norm.cdf(sigma / np.sqrt(2)) - 1 - gini

    # 数值解 σ
    sigma = brentq(gini_equation, 1e-6, 8.0)
    mu = np.log(mean_income) - 0.5 * sigma**2
    return mu, sigma

def lognormal_variance(mu, sigma):
    """计算对数正态分布的方差"""
    return (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)

# ====== 输入不同的收入和基尼系数组 ======
data = [
    {"name": "私家车主", "mean_income": 19, "gini": 0.526},
    {"name": "网约车司机", "mean_income": 13.8, "gini": 0.526},
    {"name": "物流车主", "mean_income": 10.2, "gini": 0.526},
]

# ====== 计算参数 ======
results = []
for d in data:
    mu, sigma = lognormal_params_from_mean_gini(d["mean_income"], d["gini"])
    var = lognormal_variance(mu, sigma)
    std = np.sqrt(var)
    median = np.exp(mu)
    results.append({
        "群体": d["name"],
        "均值": d["mean_income"],
        "基尼系数": d["gini"],
        "mu": mu,
        "sigma": sigma,
        "标准差": std,
        "中位数": median,
        "变异系数": std / d["mean_income"]
    })

# 打印结果表
import pandas as pd
df = pd.DataFrame(results)
print(df.round(2))

# ====== 绘图比较 ======
plt.figure(figsize=(10,5))
x = np.arange(len(df))
width = 0.35

plt.bar(x - width/2, df["均值"], width, label="均值 (Mean)")
plt.bar(x + width/2, df["标准差"], width, label="标准差 (Std)")

plt.xticks(x, df["群体"], rotation=0, fontsize=10)
plt.ylabel("收入 (元)")
plt.title("不同群体收入均值与标准差比较")
plt.legend()
plt.tight_layout()
plt.show()

# ====== 可选：sigma vs gini 散点图 ======
plt.figure(figsize=(6,5))
plt.scatter(df["基尼系数"], df["sigma"], s=80, c='teal')
for i, row in df.iterrows():
    plt.text(row["基尼系数"]+0.005, row["sigma"], row["群体"])
plt.xlabel("基尼系数")
plt.ylabel("对数标准差 σ")
plt.title("基尼系数与对数标准差关系")
plt.grid(alpha=0.3)
plt.show()