import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# ================= 学术绘图配置 =================
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimSun'] 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300  

# ================= 配置区域 =================
ELEC_PRICE_PATH = 'data/datasets/ST_EVCDP_v2/e_price.csv'
SERVICE_FEE_PATH = 'data/datasets/ST_EVCDP_v2/s_price.csv'
# 这里设置两个输入路径
DEMAND_PATHS = {
    "Duration": 'data/datasets/ST_EVCDP_v2/duration.csv',
    "Occupancy": 'data/datasets/ST_EVCDP_v2/occupancy.csv'
}
INF_PATH = 'data/datasets/ST_EVCDP_v2/inf.csv'

LAG_PERIODS = 0           
MIN_MEAN_DEMAND_PER_PILE = 0.05  
MIN_PRICE_STD = 0.01             

# ================= 核心分析函数 =================

def analyze_demand(demand_path, label, df_total_price, pile_map):
    print(f"\n正在执行 [{label}] 的弹性回归分析 (按价格预处理平均值)...")
    df_demand = pd.read_csv(demand_path, index_col='time')
    station_ids = [col for col in df_total_price.columns if col in pile_map and col in df_demand.columns]
    
    results = []
    for sid in tqdm(station_ids, desc=f"Processing {label}"):
        try:
            price = df_total_price[sid]
            raw_demand = df_demand[sid]
            pile_count = pile_map[sid]
            
            if pile_count <= 0: continue
            
            # 1. 基础标准化 (单桩需求)
            norm_demand = raw_demand / pile_count
            
            # --- 新增：按电价聚合预处理 ---
            # 将价格和需求合并，方便按价格分组
            temp_combined = pd.DataFrame({'Price': price, 'Demand': norm_demand})
            
            # 剔除无效数据
            temp_combined = temp_combined.dropna()
            
            # 【核心步骤】：按 Price 分组，计算 Demand 的平均值
            # 这样每个站点的每一个电价点，只对应一个平均需求值
            grouped = temp_combined.groupby('Price')['Demand'].max().reset_index()
            
            # 如果该站点的电价档位太少（例如只有1-2种电价），无法做回归，跳过
            if len(grouped) < 4: continue 
            
            # 2. 基础过滤：电价是否有波动
            if grouped['Price'].std() < MIN_PRICE_STD: continue

            # 3. 计算百分比变化 (Δx/x 和 Δy/y)
            # 注意：为了使 pct_change 有意义，我们先按电价从小到大排序
            grouped = grouped.sort_values('Price')
            
            price_pct = grouped['Price'].pct_change()
            demand_pct = grouped['Demand'].pct_change()
            
            # 合并变化率并清洗
            regression_df = pd.DataFrame({'X': price_pct, 'y': demand_pct}).replace([np.inf, -np.inf], np.nan).dropna()
            
            # 4. 关键过滤：过滤掉极端异常波动
            regression_df = regression_df[
                (regression_df['X'].abs() > 0.001) & 
                (regression_df['X'].abs() < 1.0) &    
                (regression_df['y'].abs() < 5.0)
            ]
            
            # 由于按价格聚合后样本点会变少，这里放宽最小样本量限制（从30减为5，或根据实际情况调整）
            if len(regression_df) < 5: continue 

            # 5. 回归分析
            model = LinearRegression()
            X = regression_df[['X']].values
            y = regression_df['y'].values
            model.fit(X, y)
            
            results.append({
                'station_id': sid,
                'coef': model.coef_[0],  # 系数即为聚合后的需求价格弹性
                'r2': model.score(X, y),
                'type': label
            })
        except Exception as e:
            # print(f"Error processing {sid}: {e}") # 调试用
            continue
            
    return pd.DataFrame(results)

def plot_combined_results(df_combined):
    """
    两类数据合并绘图，并分别根据自身逻辑计算截断范围
    """
    # 1. 计算各自的显示边界 (按类独立计算)
    y_limits = []
    stats_info = {}
    
    for label in df_combined['type'].unique():
        subset = df_combined[df_combined['type'] == label]['coef']
        q1, q3 = subset.quantile(0.25), subset.quantile(0.75)
        iqr = q3 - q1
        y_limits.append(q1 - 3.0 * iqr) # 下界
        y_limits.append(q3 + 3.0 * iqr) # 上界
        
        # --- 修改点 1: 统计信息包含中位数，移除负相关占比 ---
        stats_info[label] = {
            'n': len(subset),
            'mean': subset.mean(),
            'median': subset.median() 
        }

    # 取两类截断边界的并集，确保两类的主体都能在图中完整显示
    global_min = min(y_limits)
    global_max = max(y_limits)

    # 2. 绘图准备
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#4682B4", "#CD5C5C"] # 学术配色：钢蓝色、印度红
    
    # 3. 绘制小提琴图 + 箱线图
    # 小提琴图：展示分布密度
    sns.violinplot(x='type', y='coef', data=df_combined, 
                   palette=colors, inner=None, alpha=0.3, ax=ax)
    
    # 箱线图：展示分位数、中位数线和均值点
    sns.boxplot(x='type', y='coef', data=df_combined, 
                palette=colors, width=0.18, showmeans=True, showfliers=False,
                meanprops={"marker":"D", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"5"},
                medianprops={"color": "orange", "linewidth": 2}, # 强化中位数线显示
                ax=ax)

    # 4. 截断坐标轴
    ax.set_ylim(global_min, global_max)
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)

    # 5. 添加统计标注
    for i, label in enumerate(stats_info.keys()):
        info = stats_info[label]
        # --- 修改点 2: 格式化文本，展示中位数 ---
        text = (f"$N = {info['n']}$\n"
                f"Mean $\\beta: {info['mean']:.4f}$\n"
                f"Median $\\beta: {info['median']:.4f}$")
        
        # 将标注放在每个类别顶部的固定位置 (global_max 的 95% 处)
        ax.text(i, global_max * 0.95, text, ha='center', va='top', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='silver'),
                fontsize=10, linespacing=1.5)

    # 6. 美化
    ax.set_xlabel('Demand Category', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'Regression Coefficient ($\beta$)', fontsize=13, fontweight='bold')
    ax.set_title('Price Sensitivity Distribution Comparison', fontsize=15, pad=20)
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    # 自动根据输入的文件名命名保存（如果是多类，建议起个通用的名字）
    plt.savefig('academic_comparison_plot.png', bbox_inches='tight', dpi=300)
    plt.show()

# ================= 主程序 =================

def run_analysis():
    # 加载共有数据
    df_elec = pd.read_csv(ELEC_PRICE_PATH, index_col='time')
    df_service = pd.read_csv(SERVICE_FEE_PATH, index_col='time')
    df_inf = pd.read_csv(INF_PATH)
    
    df_total_price = df_elec + df_service  # 总价格
    df_inf['station_id'] = df_inf['station_id'].astype(str)
    pile_map = df_inf.set_index('station_id')['pile_count'].to_dict()

    # 运行两类分析
    all_results = []
    for label, path in DEMAND_PATHS.items():
        df_res = analyze_demand(path, label, df_total_price, pile_map)
        all_results.append(df_res)

    # 合并结果集
    df_final = pd.concat(all_results, ignore_index=True)

    if df_final.empty:
        print("未找到有效数据，请检查输入或过滤阈值。")
        return

    # 绘图
    plot_combined_results(df_final)

if __name__ == "__main__":
    run_analysis()