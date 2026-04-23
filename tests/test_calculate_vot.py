# 导入必要的库
from tqdm import tqdm
import pandas as pd
import numpy as np

# 定义采样函数
def sample_income_from_mu_sigma(mu, sigma, n=1, seed=None):
    rng = np.random.default_rng(seed)
    incomes = rng.lognormal(mean=mu, sigma=sigma, size=n)
    return incomes[0] if n == 1 else incomes

# 定义参数
categories = ['私家车', '网约车', '物流车']
probs = [0.71, 0.15, 0.14]

params = {
    '私家车': {'mu': 2.43, 'sigma': 1.01},
    '网约车': {'mu': 2.11, 'sigma': 1.01},
    '物流车': {'mu': 1.81, 'sigma': 1.01}
}

# 假设一年工作天数
work_days = 248 * 8

# 读取CSV文件
df = pd.read_csv('data/ev_requests/ev_requests_scale_3.csv')

# 为每个行采样类别和计算VOT
vot_list = []
category_list = []  # 新增列表来存储类别
for _ in tqdm(range(len(df)), total=len(df), desc="计算VOT"):
    category = np.random.choice(categories, p=probs)
    category_list.append(category)  # 记录类别
    mu = params[category]['mu']
    sigma = params[category]['sigma']
    income = sample_income_from_mu_sigma(mu, sigma) * 10000
    vot = income / work_days
    vot_list.append(vot)

# 添加VOT列
df['vot'] = vot_list
df['category'] = category_list  # 添加类别列

# 保存为新文件
df.to_csv('data/ev_requests/ev_requests_scale_3_with_vot.csv', index=False)

print("处理完成，已保存为 data/ev_requests/ev_requests_scale_3_with_vot.csv")
