import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# ================= 学术绘图配置 =================

plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimSun'] # 英文字体使用Times New Roman，中文字体宋体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300  # 高分辨率导出

# ================= 配置区域 =================

ELEC_PRICE_PATH = 'data/datasets/ST_EVCDP_v2/e_price.csv'
SERVICE_FEE_PATH = 'data/datasets/ST_EVCDP_v2/s_price.csv'
DEMAND_PATH = 'data/datasets/ST_EVCDP_v2/duration.csv'
INF_PATH = 'data/datasets/ST_EVCDP_v2/inf.csv'

LAG_PERIODS = 0           # 滞后阶数（1代表滞后1小时）

# 过滤阈值
# 这里的 0.05 代表：平均每小时每根桩的请求数低于 0.05（即单桩20小时才等到1个请求）则剔除
MIN_MEAN_DEMAND_PER_PILE = 0.05
MIN_PRICE_STD = 0.01             # 电价波动标准差低于此值则剔除

# ===========================================

def plot_academic_results(df_results):
# """
# 专门优化用于处理离群点的学术绘图函数
# """
# 1. 计算统计边界（用于缩放坐标轴，而不是删除数据）
    q1 = df_results['coef'].quantile(0.25)
    q3 = df_results['coef'].quantile(0.75)
    iqr = q3 - q1

    # 定义显示范围：通常设定为 [Q1 - 3*IQR, Q3 + 3*IQR] 
    # 这能保证主体箱体清晰，同时只显示少量“较近”的离群点
    lower_limit = q1 - 3.0 * iqr
    upper_limit = q3 + 3.0 * iqr

    # 2. 开始绘图
    fig, ax = plt.subplots(figsize=(7, 8))

    # 使用小提琴图展现分布密度，配合箱线图展现分位数
    # split=True, inner="quartile" 是学术常用配置
    sns.violinplot(y=df_results['coef'], 
                ax=ax, 
                color="#D6EAF8", # 淡蓝色背景
                inner=None,      # 先不画内部线
                linewidth=1.2)

    # 在小提琴图内部叠加上窄窄的箱线图
    sns.boxplot(y=df_results['coef'], 
                ax=ax, 
                width=0.15, 
                color="#2E86C1", # 深蓝色主体
                showmeans=True,
                showfliers=False, # 这里关键：不显示离群点点位，防止拉长坐标轴
                notch=True,
                meanprops={"marker":"D", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"4"},
                boxprops={'zorder': 2},
                whiskerprops={'linewidth': 1.2})

    # 3. 【核心步骤】限制 Y 轴范围，让主体突出
    # 我们只显示到 3倍 IQR 的范围，极端的离群点会被“切断”，但在论文中会注明
    ax.set_ylim(lower_limit, upper_limit)

    # 4. 辅助线与标注
    ax.axhline(0, color='#CB4335', linestyle='--', linewidth=1.5, alpha=0.8) # 零线

    # 标注 N 值和均值
    n_total = len(df_results)
    mean_val = df_results['coef'].mean()
    median_val = df_results['coef'].median()
    neg_ratio = (df_results['coef'] < 0).mean() * 100

    # 在图上添加文本框
    stats_text = (f"Total Stations ($N$): {n_total}\n"
                f"Mean $\\beta$: {mean_val:.4f}\n"
                f"Median $\\beta$: {median_val:.4f}\n"
                f"Negative Corr: {neg_ratio:.1f}%")

    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='silver')
    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    # 5. 美化坐标轴
    ax.set_ylabel(r'Regression Coefficient ($\beta$)', fontsize=14, fontweight='bold')
    ax.set_title('Price Sensitivity Distribution Analysis', fontsize=16, pad=20)
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f'academic_violin_box_plot_{DEMAND_PATH.split("/")[-1].split(".")[0]}.png', bbox_inches='tight')
    plt.show()

def run_final_analysis():
    print("正在加载并合并数据...")
    df_elec = pd.read_csv(ELEC_PRICE_PATH, index_col='time')
    df_service = pd.read_csv(SERVICE_FEE_PATH, index_col='time')
    df_demand = pd.read_csv(DEMAND_PATH, index_col='time')
    df_inf = pd.read_csv(INF_PATH)

    # 1. 计算总费用 (电费 + 服务费)
    df_total_price = df_elec

    # 2. 准备基础信息映射 (Station ID -> Pile Count)
    df_inf['station_id'] = df_inf['station_id'].astype(str)
    pile_map = df_inf.set_index('station_id')['pile_count'].to_dict()

    station_ids = [col for col in df_total_price.columns if col in pile_map]

    results = []
    skipped_no_inf = 0
    skipped_idle = 0
    skipped_static_price = 0

    print(f"开始分析，初步匹配站点数: {len(station_ids)}")

    for sid in tqdm(station_ids):
        try:
            # A. 提取并对齐数据
            # 价格进行滞后处理
            price = df_total_price[sid].shift(LAG_PERIODS)
            raw_demand = df_demand[sid]
            pile_count = pile_map[sid]

            if pile_count <= 0:
                continue

            # B. 【核心步骤】标准化：计算单桩需求量
            # 所有的分析（过滤和回归）都基于这个 normalized_demand
            normalized_demand = raw_demand / pile_count

            # C. 过滤逻辑
            # 1. 剔除太空闲的站 (基于标准化后的需求)
            if normalized_demand.mean() < MIN_MEAN_DEMAND_PER_PILE:
                skipped_idle += 1
                continue
            
            # 2. 剔除电价几乎没有变动的站
            if price.std() < MIN_PRICE_STD:
                skipped_static_price += 1
                continue

            # D. 构建回归模型数据集
            temp_df = pd.DataFrame({
                'X_price': price,
                'y_demand': normalized_demand
            }).dropna()

            if len(temp_df) < 100: 
                continue

            X = temp_df[['X_price']].values
            y = temp_df['y_demand'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            results.append({
                'station_id': sid,
                'coef': model.coef_[0],       # 这里的系数含义：电价每涨1元，单桩请求数的变化量
                'r2': model.score(X, y),
                'pile_count': pile_count,
                'avg_demand_per_pile': y.mean()
            })

        except Exception as e:
            continue

    # 3. 统计与可视化
    df_results = pd.DataFrame(results)

    print(f"\n--- 过滤报告 ---")
    print(f"因无桩数信息被跳过: {len(df_total_price.columns) - len(station_ids)} 个")
    print(f"因【单桩需求太低】被剔除: {skipped_idle} 个")
    print(f"因【电价无变动】被剔除: {skipped_static_price} 个")
    print(f"最终有效分析站点: {len(df_results)} 个")

    if df_results.empty:
        print("错误：筛选后没有剩余站点，请检查阈值设置。")
        return

    # ================= 核心：学术级绘图 =================
    plot_academic_results(df_results)

if __name__ == "__main__":
    run_final_analysis()