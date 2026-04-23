import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Set academic plotting style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12  # 稍微调大字体以适应摘要图
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0

def parse_csv_files(allocation_file, station_file):
    """Parse CSV files and extract charging allocation data"""
    df_allocation = pd.read_csv(allocation_file)
    # df_station is loaded but might not be strictly needed for the heatmap matrix logic
    # depending on how station_id is mapped, but keeping it for consistency.
    df_station = pd.read_csv(station_file)
    
    # Convert time columns to datetime
    df_allocation['request_time'] = pd.to_datetime(df_allocation['request_time'])
    df_allocation['end_time'] = pd.to_datetime(df_allocation['end_time'])
    
    return df_allocation, df_station

def separate_station_types(df_allocation):
    """Separate FCS and MCS data from allocation results"""
    # 虽然只画 MCS，但为了保持逻辑完整防止报错，还是保留 FCS 的处理
    df_fcs = df_allocation[df_allocation['charging_entity_name'].str.contains('fcs', case=False, na=False)].copy()
    df_fcs['charging_entity_name'] = df_fcs['charging_entity_name'].str.rsplit('_', n=1, expand=True)[0]
    
    df_mcs = df_allocation[df_allocation['charging_entity_name'].str.contains('mcs', case=False, na=False)].copy()

    # 简单的对齐时间操作
    if not df_mcs.empty and not df_fcs.empty:
        max_end_time = df_mcs['end_time'].max()
        df_fcs = df_fcs[df_fcs['end_time'] <= max_end_time]
    
    return df_fcs, df_mcs

def create_heatmap_matrix(df, station_type='fcs'):
    """Create a matrix for heatmap visualization"""
    if len(df) == 0:
        return None, None, None
    
    unique_stations = sorted(df['charging_entity_name'].unique())
    station_id_map = {station: idx + 1 for idx, station in enumerate(unique_stations)}
    df['station_id'] = df['charging_entity_name'].map(station_id_map)
    
    start_time = df['request_time'].min()
    df['time_hour'] = (df['request_time'] - start_time).dt.total_seconds() / 3600
    df['end_time_hour'] = (df['end_time'] - start_time).dt.total_seconds() / 3600
    
    max_time = df['end_time_hour'].max()
    time_bins = np.arange(0, int(np.ceil(max_time)) + 1, 1)
    
    n_stations = len(unique_stations)
    n_time_bins = len(time_bins) - 1
    matrix = np.zeros((n_stations, n_time_bins))
    
    for _, row in df.iterrows():
        station_idx = row['station_id'] - 1
        start_hour = row['time_hour']
        end_hour = row['end_time_hour']
        energy = row['energy_needed']
        
        for i, t in enumerate(time_bins[:-1]):
            t_next = time_bins[i + 1]
            overlap_start = max(start_hour, t)
            overlap_end = min(end_hour, t_next)
            
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                total_duration = end_hour - start_hour
                if total_duration > 0:
                    energy_in_bin = energy * (overlap_duration / total_duration)
                    matrix[station_idx, i] += energy_in_bin
    
    station_ids = list(range(1, n_stations + 1))
    return matrix, station_ids, time_bins[:-1]

def create_custom_colormap():
    """
    Create a custom colormap to coordinate with the reference image.
    Reference image uses: 
    - Blue (Duration) ~ #5b9bd5
    - Red (Occupancy) ~ #c0504d
    
    We will create a map that transitions: White -> Light Blue -> Reddish
    This mimics a 'coolwarm' or 'RdYlBu_r' style but matches your specific hues.
    """
    # 定义关键颜色节点
    colors = [
        (0.0, '#ffffff'),  # 0值用白色
        (0.1, '#ebf3fb'),  # 极低值用极浅蓝
        (0.4, '#5b9bd5'),  # 中低值用你的参考图蓝色 (Duration color)
        (0.7, '#f4a582'),  # 过渡橙色
        (1.0, '#c0504d')   # 高值用你的参考图红色 (Occupancy color)
    ]
    
    cmap = LinearSegmentedColormap.from_list('custom_coordinated', colors, N=256)
    return cmap

def plot_heatmaps(scenario1_files, scenario2_files, output_file='charging_heatmap_abstract.png'):
    """
    Plot only the MCS heatmaps (b and d) side by side
    """
    # Parse data
    df_alloc1, _ = parse_csv_files(*scenario1_files)
    df_alloc2, _ = parse_csv_files(*scenario2_files)
    
    # Separate types
    _, df_mcs1 = separate_station_types(df_alloc1)
    _, df_mcs2 = separate_station_types(df_alloc2)
    
    # Create matrices (Only MCS)
    matrix_mcs1, stations_mcs1, time_mcs1 = create_heatmap_matrix(df_mcs1, 'mcs')
    matrix_mcs2, stations_mcs2, time_mcs2 = create_heatmap_matrix(df_mcs2, 'mcs')
    
    # Create custom colormap
    cmap = create_custom_colormap()
    
    # Create figure with 1 row, 2 columns (Side by Side)
    # 调整figsize使其更适合作为摘要图（宽扁型）
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.15, wspace=0.25)
    
    # Plot configurations: (Data, Ax, Title)
    # 注意：去掉了原来的 (b) (d) 标签
    plot_configs = [
        (matrix_mcs1, stations_mcs1, time_mcs1, axes[0], 'MCS (No Prediction)'),
        (matrix_mcs2, stations_mcs2, time_mcs2, axes[1], 'MCS (With Prediction)')
    ]
    
    # Calculate global min/max for MCS to ensure consistent coloring
    mcs_matrices = [m for m in [matrix_mcs1, matrix_mcs2] if m is not None]
    vmin_mcs = min([m.min() for m in mcs_matrices]) if mcs_matrices else 0
    vmax_mcs = max([m.max() for m in mcs_matrices]) if mcs_matrices else 1
    
    for matrix, station_ids, time_bins, ax, title in plot_configs:
        if matrix is None:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue
        
        # Plot Heatmap
        im = ax.imshow(matrix, aspect='auto', cmap=cmap, 
                      interpolation='bilinear', origin='lower',
                      vmin=vmin_mcs, vmax=vmax_mcs,
                      extent=[time_bins[0], time_bins[-1] + 1, 0.5, len(station_ids) + 0.5])
        
        # Set Labels
        # 根据要求修改 X 轴标签
        ax.set_xlabel('Time (h)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Station No.', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        
        # X-axis ticks
        max_time = int(time_bins[-1]) + 1
        x_ticks = np.arange(0, max_time + 1, 5)  # Step 5 to reduce clutter
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(int(x)) for x in x_ticks], fontsize=10)
        
        # Y-axis ticks
        n_stations = len(station_ids)
        if n_stations <= 10:
            y_ticks = station_ids
        else:
            step = max(1, n_stations // 8)
            y_ticks = station_ids[::step]
        
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(int(y)) for y in y_ticks], fontsize=10)
        ax.set_ylim(0.5, len(station_ids) + 0.5)

    # Add shared Colorbar
    # 在右侧添加一个共用的 colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.73]) # [left, bottom, width, height]
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Power (kW·h)', fontsize=11, fontweight='bold', rotation=270, labelpad=15)
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Heatmap saved to: {output_file}")
    
    return fig

if __name__ == "__main__":
    # Define file paths
    scenario1_allocation = "data/greedy_algorithm/no_pred_1116_v1/ev_requests_with_assignments.csv"
    scenario1_station = "data/greedy_algorithm/no_pred_1116_v1/cs_with_assignments.csv"
    
    scenario2_allocation = "data/greedy_algorithm/with_pred_pag_informer_best_mcs_300/ev_requests_with_assignments.csv"
    scenario2_station = "data/greedy_algorithm/with_pred_pag_informer_best_mcs_300/cs_with_assignments.csv"
    
    fig = plot_heatmaps(
        scenario1_files=(scenario1_allocation, scenario1_station),
        scenario2_files=(scenario2_allocation, scenario2_station),
        output_file='data/plots/charging_load_heatmap_abstract.png'
    )
    
    plt.show()