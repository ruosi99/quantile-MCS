import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 0. 全局设置与美化配置
# ===============================

# 尝试设置较好看的绘图风格
try:
    # 新版 matplotlib 推荐写法
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        # 旧版写法
        plt.style.use('seaborn-whitegrid')
    except:
        # 如果都没有，回退到经典美化风格
        plt.style.use('ggplot')

# 定义统一的配色方案 (Hex color codes)
COLORS = {
    'true': '#1f77b4',      # 较深的蓝色
    'pred': '#ff7f0e',      # 橙色
    'band': '#1f77b4',      # 与真值同色系，但配合透明度
    'bar':  '#2ca02c',      # 绿色
    'hist': '#7f7f7f',      # 灰色直方图
    'line': '#d62728'       # 红色辅助线
}

def setup_axis(ax, title, xlabel=None, ylabel=None):
    """通用的坐标轴美化函数"""
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    if xlabel: ax.set_xlabel(xlabel, fontsize=12)
    if ylabel: ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    # 移除顶部和右侧的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ===============================
# 1. 主绘图函数 (时序预测图)
# ===============================

def plot_prediction_with_interval(
        y_true,
        y_pred,
        L,
        U,
        node_id=0,
        save_path=None,
        max_points=200):
    
    # 截取数据
    y_true_plot = y_true[:max_points, node_id]
    y_pred_plot = y_pred[:max_points, node_id]
    L_plot = L[:max_points, node_id]
    U_plot = U[:max_points, node_id]

    t = np.arange(len(y_true_plot))

    plt.figure(figsize=(14, 7)) # 稍微加宽
    ax = plt.gca()

    # 画置信区间 (先画背景)
    plt.fill_between(
        t, L_plot, U_plot,
        color=COLORS['band'],
        alpha=0.2, # 降低透明度让背景不喧宾夺主
        label='Confidence Interval',
        edgecolor=None
    )

    # 画真值
    plt.plot(t, y_true_plot, 
             color=COLORS['true'], 
             linewidth=2.5, 
             label='Ground Truth', 
             alpha=0.9)
    
    # 画预测值
    plt.plot(t, y_pred_plot, 
             color=COLORS['pred'], 
             linestyle='--', 
             linewidth=2, 
             label='Prediction (Median)')

    setup_axis(ax, 
               title=f'Node {node_id} Forecast: Truth vs Prediction',
               xlabel='Time Step', 
               ylabel='Value')
    
    plt.legend(loc='best', frameon=True, fancybox=True, framealpha=0.9)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


# ===============================
# 2. Coverage Plot (覆盖率)
# ===============================

def plot_coverage(L, U, y_true, target=0.9, save_path=None):

    covered = (y_true >= L) & (y_true <= U)
    coverage = covered.mean()

    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # 画柱状图
    bars = plt.bar(['Coverage'], [coverage], 
                   color=COLORS['bar'], 
                   width=0.5, 
                   alpha=0.8,
                   edgecolor='black')
    
    # 画目标线
    plt.axhline(target, color=COLORS['line'], linestyle='--', linewidth=2, label=f'Target ({target})')
    
    # 在柱子上方标注具体数值
    plt.text(0, coverage + 0.02, f'{coverage:.2%}', 
             ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.ylim(0, 1.1) # 留一点头部空间给文字
    setup_axis(ax, title='Prediction Interval Coverage')
    plt.legend(loc='lower right')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


# ===============================
# 3. Interval Width Histogram (区间宽度分布)
# ===============================

def plot_interval_width(L, U, save_path=None):
    
    width = (U - L).reshape(-1)
    
    # --- 改进的核心：去除离群值 ---
    # 计算 1% 和 99% 分位数
    lower_bound = np.percentile(width, 1)
    upper_bound = np.percentile(width, 99)
    
    # 过滤掉 NaN 或 Inf
    width_clean = width[np.isfinite(width)]
    
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # 使用 range 参数限制绘图范围，忽略极值
    n, bins, patches = plt.hist(width_clean, 
                                bins=50, 
                                range=(lower_bound, upper_bound),
                                color=COLORS['hist'], 
                                alpha=0.7, 
                                edgecolor='white', 
                                linewidth=0.5)

    # 标注平均宽度
    mean_width = np.mean(width_clean)
    plt.axvline(mean_width, color=COLORS['line'], linestyle='--', linewidth=2, label=f'Mean: {mean_width:.2f}')

    setup_axis(ax, 
               title='Interval Width Distribution (1%-99% Percentile)',
               xlabel='Width',
               ylabel='Frequency')
    
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


# ===============================
# 4. Error Histogram (误差分布)
# ===============================

def plot_error_distribution(y_true, y_pred, save_path=None):

    error = (y_true - y_pred).reshape(-1)
    
    # --- 改进的核心：去除离群值 ---
    lower_bound = np.percentile(error, 1)
    upper_bound = np.percentile(error, 99)
    
    # 过滤掉 NaN
    error_clean = error[np.isfinite(error)]

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    plt.hist(error_clean, 
             bins=50, 
             range=(lower_bound, upper_bound), # 限制范围
             color=COLORS['pred'], 
             alpha=0.7, 
             edgecolor='white',
             linewidth=0.5,
             density=True) # 使用密度图，让Y轴数值不至于太大

    # 标注0点
    plt.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    
    # 标注均值和标准差
    mu, std = np.mean(error_clean), np.std(error_clean)
    title_str = f'Error Distribution (1%-99%)\nMean: {mu:.3f}, Std: {std:.3f}'
    
    setup_axis(ax, 
               title=title_str,
               xlabel='Error (True - Pred)',
               ylabel='Density')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


# ===============================
# 5. 自动读取结果并绘图
# ===============================

def main_plot_forecasting_results(result_dir):
    
    if not os.path.exists(result_dir):
        print(f"Error: Directory {result_dir} not found.")
        return

    print(f"Loading data from {result_dir}...")
    
    try:
        y_true = np.load(os.path.join(result_dir, 'label_list.npy'))
        y_pred = np.load(os.path.join(result_dir, 'predict_point_q50.npy'))

        # 自动找 interval 文件 (增加容错)
        files = os.listdir(result_dir)
        L_file = next((f for f in files if f.startswith('cqr_L')), None)
        U_file = next((f for f in files if f.startswith('cqr_U')), None)
        
        if not L_file or not U_file:
            print("Error: Could not find cqr_L or cqr_U files.")
            return

        L = np.load(os.path.join(result_dir, L_file))
        U = np.load(os.path.join(result_dir, U_file))

        print(f"Data Loaded. Shape: {y_true.shape}")

        # 1. 预测图
        plot_prediction_with_interval(
            y_true, y_pred, L, U,
            node_id=8,
            save_path=os.path.join(result_dir, 'forecast_plot_node0.png')
        )

        # 2. 覆盖率
        plot_coverage(
            L, U, y_true,
            save_path=os.path.join(result_dir, 'coverage.png')
        )

        # 3. 区间宽度 (直方图)
        plot_interval_width(
            L, U,
            save_path=os.path.join(result_dir, 'interval_width.png')
        )

        # 4. 误差分布 (直方图)
        plot_error_distribution(
            y_true, y_pred,
            save_path=os.path.join(result_dir, 'error_dist.png')
        )

        print("All plots saved successfully.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

# 使用示例（如果需要直接运行脚本测试）
if __name__ == "__main__":
    # 可以通过命令行传参，或者直接指定路径
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dir', type=str, default='./results')
    # args = parser.parse_args()
    # main_plot_forecasting_results(args.dir)
    pass