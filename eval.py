import pandas as pd
from datetime import datetime

# 定义指标计算函数
def calculate_metrics(result_dir, cutoff_time, output_file):
    # 读取数据
    stations = pd.read_csv(f"{result_dir}/cs_with_assignments.csv")
    requests = pd.read_csv(f"{result_dir}/ev_requests_with_assignments.csv")

    # 判断充电站类型
    stations['entity_type'] = stations['name'].apply(lambda x: 'mcs' if 'mcs' in x.lower() else ('fcs' if 'fcs' in x.lower() else 'unknown'))

    # 判断充电请求分配的充电站类型
    requests['entity_type'] = requests['charging_entity_name'].apply(lambda x: 'mcs' if 'mcs' in x.lower() else ('fcs' if 'fcs' in x.lower() else 'unknown'))

    # 计算等待时间
    requests['waiting_time'] = requests['travel_time'] + requests['queue_time'] + requests['charging_time']

    # 计算不同类型的平均等待时间、排队时间和充电时间
    avg_waiting_time_mcs = requests[requests['entity_type'] == 'mcs']['waiting_time'].mean()
    avg_waiting_time_fcs = requests[requests['entity_type'] == 'fcs']['waiting_time'].mean()
    avg_waiting_time_total = requests['waiting_time'].mean()
    
    avg_travel_time_mcs = requests[requests['entity_type'] == 'mcs']['travel_time'].mean()
    avg_travel_time_fcs = requests[requests['entity_type'] == 'fcs']['travel_time'].mean()
    avg_travel_time_total = requests['travel_time'].mean()
    
    avg_queue_time_mcs = requests[requests['entity_type'] == 'mcs']['queue_time'].mean()
    avg_queue_time_fcs = requests[requests['entity_type'] == 'fcs']['queue_time'].mean()
    avg_queue_time_total = requests['queue_time'].mean()

    avg_charging_time_mcs = requests[requests['entity_type'] == 'mcs']['charging_time'].mean()
    avg_charging_time_fcs = requests[requests['entity_type'] == 'fcs']['charging_time'].mean()
    avg_charging_time_total = requests['charging_time'].mean()

    # 计算不同类型充电站的占用率
    avg_occupancy_rate_mcs = stations[stations['entity_type'] == 'mcs']['occupancy_rate'].mean()
    avg_occupancy_rate_fcs = stations[stations['entity_type'] == 'fcs']['occupancy_rate'].mean()
    avg_occupancy_rate_total = stations['occupancy_rate'].mean()

    # 计算充电任务完成率
    cutoff_datetime = datetime.strptime(cutoff_time, "%Y-%m-%d %H:%M:%S")
    completed_requests = requests[pd.to_datetime(requests['end_time']) <= cutoff_datetime]
    completion_rate_mcs = len(completed_requests[completed_requests['entity_type'] == 'mcs']) / len(requests[requests['entity_type'] == 'mcs']) if len(requests[requests['entity_type'] == 'mcs']) > 0 else 0
    completion_rate_fcs = len(completed_requests[completed_requests['entity_type'] == 'fcs']) / len(requests[requests['entity_type'] == 'fcs']) if len(requests[requests['entity_type'] == 'fcs']) > 0 else 0
    completion_rate_total = len(completed_requests) / len(requests)

    # 计算不同类型充电站的平均成本
    avg_cost_mcs = requests[requests['entity_type'] == 'mcs']['cost'].mean()
    avg_cost_fcs = requests[requests['entity_type'] == 'fcs']['cost'].mean()
    avg_cost_total = requests['cost'].mean()

    # 将结果写入文件
    with open(output_file, 'w', encoding = "utf8") as f:
        f.write(f"平均等待时间 (MCS): {avg_waiting_time_mcs:.2f} 分钟\n")
        f.write(f"平均等待时间 (FCS): {avg_waiting_time_fcs:.2f} 分钟\n")
        f.write(f"平均等待时间 (总计): {avg_waiting_time_total:.2f} 分钟\n")
        f.write(f"平均抵达时间 (MCS): {avg_travel_time_mcs:.2f} 分钟\n")
        f.write(f"平均抵达时间 (FCS): {avg_travel_time_fcs:.2f} 分钟\n")
        f.write(f"平均抵达时间 (总计): {avg_travel_time_total:.2f} 分钟\n")
        f.write(f"平均排队时间 (MCS): {avg_queue_time_mcs:.2f} 分钟\n")
        f.write(f"平均排队时间 (FCS): {avg_queue_time_fcs:.2f} 分钟\n")
        f.write(f"平均排队时间 (总计): {avg_queue_time_total:.2f} 分钟\n")
        f.write(f"平均充电时间 (MCS): {avg_charging_time_mcs:.2f} 分钟\n")
        f.write(f"平均充电时间 (FCS): {avg_charging_time_fcs:.2f} 分钟\n")
        f.write(f"平均充电时间 (总计): {avg_charging_time_total:.2f} 分钟\n")
        f.write(f"平均占用率 (MCS): {avg_occupancy_rate_mcs:.2%}\n")
        f.write(f"平均占用率 (FCS): {avg_occupancy_rate_fcs:.2%}\n")
        f.write(f"平均占用率 (总计): {avg_occupancy_rate_total:.2%}\n")
        f.write(f"充电任务完成率 (MCS): {completion_rate_mcs:.2%}\n")
        f.write(f"充电任务完成率 (FCS): {completion_rate_fcs:.2%}\n")
        f.write(f"充电任务完成率 (总计): {completion_rate_total:.2%}\n")
        f.write(f"平均成本 (MCS): {avg_cost_mcs:.2f} 元\n")
        f.write(f"平均成本 (FCS): {avg_cost_fcs:.2f} 元\n")
        f.write(f"平均成本 (总计): {avg_cost_total:.2f} 元\n")
if __name__ == "__main__":
    output_dir = "data/greedy_algorithm/no_pred_1103"
    calculate_metrics(
        output_dir, "2023-06-20 07:02:52", f"{output_dir}/metrics.txt"
    )
