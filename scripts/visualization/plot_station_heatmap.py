import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set academic plotting style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0

def parse_csv_files(allocation_file, station_file):
    """
    Parse CSV files and extract charging allocation data
    
    Parameters:
    -----------
    allocation_file: str, path to EV charging request allocation file
    station_file: str, path to charging station service record file
    
    Returns:
    --------
    df_allocation: DataFrame with allocation data
    df_station: DataFrame with station data
    """
    df_allocation = pd.read_csv(allocation_file)
    df_station = pd.read_csv(station_file)
    
    # Convert time columns to datetime
    df_allocation['request_time'] = pd.to_datetime(df_allocation['request_time'])
    df_allocation['end_time'] = pd.to_datetime(df_allocation['end_time'])
    
    return df_allocation, df_station

def separate_station_types(df_allocation):
    """
    Separate FCS and MCS data from allocation results
    
    Parameters:
    -----------
    df_allocation: DataFrame with allocation data
    
    Returns:
    --------
    df_fcs: DataFrame with FCS allocation data
    df_mcs: DataFrame with MCS allocation data
    """
    df_fcs = df_allocation[df_allocation['charging_entity_name'].str.contains('fcs', case=False, na=False)].copy()
    # 新增：对 FCS 提取充电站前缀（如 fcs_2137）
    df_fcs['charging_entity_name'] = df_fcs['charging_entity_name'].str.rsplit('_', n=1, expand=True)[0]
    df_mcs = df_allocation[df_allocation['charging_entity_name'].str.contains('mcs', case=False, na=False)].copy()

    # df_fcs的end_time列不能超过df_mcs的end_time列的最大值
    max_end_time = df_mcs['end_time'].max()
    df_fcs = df_fcs[df_fcs['end_time'] <= max_end_time]
    
    return df_fcs, df_mcs

def create_heatmap_matrix(df, station_type='fcs'):
    """
    Create a matrix for heatmap visualization
    
    Parameters:
    -----------
    df: DataFrame with charging allocation data
    station_type: str, 'fcs' or 'mcs'
    
    Returns:
    --------
    matrix: 2D numpy array for heatmap
    station_ids: list of station IDs
    time_bins: array of time bins (hours)
    """
    if len(df) == 0:
        return None, None, None
    
    # Get unique stations and assign integer IDs
    unique_stations = sorted(df['charging_entity_name'].unique())
    station_id_map = {station: idx + 1 for idx, station in enumerate(unique_stations)}
    df['station_id'] = df['charging_entity_name'].map(station_id_map)
    
    # Calculate time in hours from start
    start_time = df['request_time'].min()
    df['time_hour'] = (df['request_time'] - start_time).dt.total_seconds() / 3600
    df['charging_time_hour'] = df['charging_time'] / 60  # Convert minutes to hours
    df['end_time_hour'] = (df['end_time'] - start_time).dt.total_seconds() / 3600
    
    # Determine time range
    max_time = df['end_time_hour'].max()
    time_bins = np.arange(0, int(np.ceil(max_time)) + 1, 1)  # Hourly bins
    
    # Create matrix
    n_stations = len(unique_stations)
    n_time_bins = len(time_bins) - 1
    matrix = np.zeros((n_stations, n_time_bins))
    
    # Fill matrix with energy consumption
    for _, row in df.iterrows():
        station_idx = row['station_id'] - 1
        start_hour = row['time_hour']
        end_hour = row['end_time_hour']
        energy = row['energy_needed']
        
        # Distribute energy across time bins
        for i, t in enumerate(time_bins[:-1]):
            t_next = time_bins[i + 1]
            # Calculate overlap between charging period and time bin
            overlap_start = max(start_hour, t)
            overlap_end = min(end_hour, t_next)
            
            if overlap_end > overlap_start:
                # Proportion of charging in this bin
                overlap_duration = overlap_end - overlap_start
                total_duration = end_hour - start_hour
                if total_duration > 0:
                    energy_in_bin = energy * (overlap_duration / total_duration)
                    matrix[station_idx, i] += energy_in_bin
    
    # Convert energy (kWh) to power (kW) by dividing by bin width (1 hour)
    matrix = matrix  # Already in kWh per hour, which equals kW·h
    
    station_ids = list(range(1, n_stations + 1))
    
    return matrix, station_ids, time_bins[:-1]

def create_custom_colormap():
    """
    Create a custom colormap similar to the reference image
    """
    # Define colors similar to the reference image: blue -> cyan -> green -> yellow -> orange -> red
    colors = [
        '#440154',  # Dark purple
        '#3b528b',  # Dark blue
        '#21908c',  # Cyan
        '#5dc863',  # Green
        '#fde725',  # Yellow
        '#f58231',  # Orange
        '#e31a1c'   # Red
    ]
    
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    return cmap

def plot_heatmaps(scenario1_files, scenario2_files, output_file='charging_heatmap.png'):
    """
    Plot four heatmaps in a 2x2 grid
    
    Parameters:
    -----------
    scenario1_files: tuple, (allocation_file, station_file) for Scenario I
    scenario2_files: tuple, (allocation_file, station_file) for Scenario II
    output_file: str, path to save the output figure
    """
    # Parse data for both scenarios
    df_alloc1, df_station1 = parse_csv_files(*scenario1_files)
    df_alloc2, df_station2 = parse_csv_files(*scenario2_files)
    
    # Separate FCS and MCS for both scenarios
    df_fcs1, df_mcs1 = separate_station_types(df_alloc1)
    df_fcs2, df_mcs2 = separate_station_types(df_alloc2)
    
    # Create matrices
    matrix_fcs1, stations_fcs1, time_fcs1 = create_heatmap_matrix(df_fcs1, 'fcs')
    matrix_mcs1, stations_mcs1, time_mcs1 = create_heatmap_matrix(df_mcs1, 'mcs')
    matrix_fcs2, stations_fcs2, time_fcs2 = create_heatmap_matrix(df_fcs2, 'fcs')
    matrix_mcs2, stations_mcs2, time_mcs2 = create_heatmap_matrix(df_mcs2, 'mcs')
    
    # Create custom colormap
    cmap = create_custom_colormap()
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.08, wspace=0.3, hspace=0.3)
    
    # Plot parameters
    plot_configs = [
        (matrix_fcs1, stations_fcs1, time_fcs1, axes[0, 0], '(a)', 'FCS (No Prediction)'),
        (matrix_mcs1, stations_mcs1, time_mcs1, axes[1, 0], '(b)', 'MCS (No Prediction)'),
        (matrix_fcs2, stations_fcs2, time_fcs2, axes[0, 1], '(c)', 'FCS (With Prediction)'),
        (matrix_mcs2, stations_mcs2, time_mcs2, axes[1, 1], '(d)', 'MCS (With Prediction)')
    ]
    
    # Find global min/max for consistent color scaling within each type
    fcs_matrices = [m for m in [matrix_fcs1, matrix_fcs2] if m is not None]
    mcs_matrices = [m for m in [matrix_mcs1, matrix_mcs2] if m is not None]
    
    vmin_fcs = min([m.min() for m in fcs_matrices]) if fcs_matrices else 0
    vmax_fcs = max([m.max() for m in fcs_matrices]) if fcs_matrices else 1
    vmin_mcs = min([m.min() for m in mcs_matrices]) if mcs_matrices else 0
    vmax_mcs = max([m.max() for m in mcs_matrices]) if mcs_matrices else 1
    
    for matrix, station_ids, time_bins, ax, label, title in plot_configs:
        if matrix is None:
            ax.text(0.5, 0.5, 'No data available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{label}  {title}', fontsize=11, fontweight='bold', loc='left')
            continue
        
        # Determine vmin and vmax based on station type
        if 'FC' in title:
            vmin, vmax = vmin_fcs, vmax_fcs
        else:
            vmin, vmax = vmin_mcs, vmax_mcs
        
        # Create heatmap
        im = ax.imshow(matrix, aspect='auto', cmap=cmap, 
                      interpolation='bilinear', origin='lower',
                      vmin=vmin, vmax=vmax,
                      extent=[time_bins[0], time_bins[-1] + 1, 0.5, len(station_ids) + 0.5])
        
        # Set labels and title
        ax.set_xlabel('Time (h)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Station No.', fontsize=10, fontweight='bold')
        ax.set_title(f'{label}  {title}', fontsize=11, fontweight='bold', loc='left', pad=10)
        
        # Set ticks
        max_time = int(time_bins[-1]) + 1
        x_ticks = np.arange(0, max_time + 1, 2)  # Every 2 hours
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(int(x)) for x in x_ticks], fontsize=9)
        
        # Y-axis ticks - show representative station IDs
        n_stations = len(station_ids)
        if n_stations <= 10:
            y_ticks = station_ids
        else:
            # Show every few stations to avoid crowding
            step = max(1, n_stations // 8)
            y_ticks = station_ids[::step]
        
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(int(y)) for y in y_ticks], fontsize=9)
        ax.set_ylim(0.5, len(station_ids) + 0.5)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Power (kW·h)', fontsize=9, fontweight='bold', rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=8)
        
        # Format colorbar ticks
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
    
    # Add figure title
    # fig.suptitle('Load Distribution of Each Station', fontsize=13, fontweight='bold', y=0.98)
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Heatmap saved to: {output_file}")
    
    return fig

# Example usage
if __name__ == "__main__":
    # Define file paths for two scenarios
    # Scenario I
    scenario1_allocation = "data/greedy_algorithm/no_pred_1116_v1/ev_requests_with_assignments.csv"
    scenario1_station = "data/greedy_algorithm/no_pred_1116_v1/cs_with_assignments.csv"
    
    # Scenario II
    scenario2_allocation = "data/greedy_algorithm/with_pred_pag_informer_best_mcs_300/ev_requests_with_assignments.csv"
    scenario2_station = "data/greedy_algorithm/with_pred_pag_informer_best_mcs_300/cs_with_assignments.csv"
    
    # Create heatmaps
    fig = plot_heatmaps(
        scenario1_files=(scenario1_allocation, scenario1_station),
        scenario2_files=(scenario2_allocation, scenario2_station),
        output_file='data/plots/charging_load_heatmap.png'
    )
    
    plt.show()
