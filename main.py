import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from allocation_algorithm.GreedyAlgorithm import greedy_algorithm
from allocation_algorithm.GeneticAlgorithm import genetic_algorithm
from allocation_algorithm.ReinforcementLearning import reinforcement_learning_algorithm
from allocation_algorithm.DqnReinforcementLearning import reinforcement_learning_dqn_algorithm
from utils.dataset_process import restructure_fixed_charging_stations
from utils.charging_entity import ChargingEntity
from eval import calculate_metrics

import random

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='EV charging allocation script.')
    parser.add_argument('--stations_file', required=True, help='Path to the stations CSV file.')
    parser.add_argument('--mcs_file', required=True, help='Path to the MCS info CSV file.')
    parser.add_argument('--ev_request_file', required=True, help='Path to the EV request CSV file.')
    parser.add_argument('--output_dir', required=True, help='Output directory for results.')
    parser.add_argument('--algorithm', required=True, help='Algorithm name to use for allocation.')
    parser.add_argument('--prob_param', required=False, help='Probability parameter for greedy algorithm.', default=0.7)
    parser.add_argument('--use_mcs', required=False, help='Use MCS.', default=True, type=lambda x: str(x).lower() == 'true')
    parser.add_argument('--gamma', required=False, help='每个未被成功满足的充电请求在目标函数中对应的固定惩罚成本.', default=58)

    args = parser.parse_args()

    # Load station info
    stations = pd.read_csv(args.stations_file)

    # Load MCS info
    mcs = pd.read_csv(args.mcs_file)

    # FCS Location Initialization
    fcs = restructure_fixed_charging_stations(stations)

    # EV Request Initialization
    ev_request = pd.read_csv(args.ev_request_file)
    # ev_request['longitude'], ev_request['latitude'] = ev_request['latitude'], ev_request['longitude']
    ev_request = ev_request[pd.to_datetime(ev_request['request_time']).dt.hour.isin([0, 1, 2, 3, 4, 5, 6])]
    ev_request = ev_request[ev_request["energy_needed"] != 0]
    ev_request = ev_request.sort_values(by='request_time',
        key=lambda x: x.apply(lambda time_str: datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S"))).reset_index(drop=True)

    # Merge FCS and MCS
    empty_df = pd.DataFrame(columns=['station_id', 'longitude', 'latitude', 'charging_queue', 'available'])
    charging_stations = pd.concat([
            mcs[['station_id', 'longitude', 'latitude', 'charging_queue', 'available']] if args.use_mcs else empty_df,
            fcs[['station_id', 'longitude', 'latitude', 'charging_queue', 'available']]
        ]).drop_duplicates().reset_index(drop=True)

    # Filter EV requests
    import re
    # 设置随机数种子
    seed = 42
    numbers_in_df2 = charging_stations['station_id'].apply(lambda x: re.findall(r'\d+', str(x)))
    numbers_in_df2 = [int(num) for sublist in numbers_in_df2 for num in sublist]
    ev_request_supp = ev_request[~ev_request['station_id'].isin(numbers_in_df2)]
    ev_request = ev_request[ev_request['station_id'].isin(numbers_in_df2)]
    print(ev_request)
    # ev_request = pd.concat([ev_request, ev_request_supp.sample(n=10000, random_state=seed)], ignore_index=True)

    # Initialize charging entities
    charging_entities = []
    random.seed(seed)
    for index, row in charging_stations.iterrows():
        if row['station_id'].startswith("mcs"):
            charging_power = 80
            electricity_fee = 2
        elif row['station_id'].startswith("fcs"):
            if random.random() < 0.1712:
                charging_power = 80
                electricity_fee = 2  # 快充
            else:
                charging_power = 27.5
                electricity_fee = 1  # 慢充
        else:
            charging_power = 60  # default for any other cases
            electricity_fee = 1  # 默认值，可以根据需要调整
        entity = ChargingEntity(str(row['station_id']), charging_power, row['latitude'], row['longitude'], electricity_fee)
        charging_entities.append(entity)

    output_dir = "data/{}/{}".format(args.algorithm, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Run algorithm
    if args.algorithm == "greedy_algorithm":
        greedy_algorithm(charging_entities, ev_request, output_dir, float(args.prob_param))
    elif args.algorithm == "genetic_algorithm":
        genetic_algorithm(charging_entities, ev_request, output_dir)
    elif args.algorithm == "reinforcement_learning_algorithm":
        reinforcement_learning_algorithm(charging_entities, ev_request, output_dir)
    elif args.algorithm == "reinforcement_learning_dqn_algorithm":
        reinforcement_learning_dqn_algorithm(charging_entities, ev_request, output_dir)
    else:
        print("Invalid algorithm specified. Please choose a valid algorithm.")

    # # Evaluate results
    calculate_metrics(
        output_dir, "2023-06-20 07:00:00", f"{output_dir}/metrics.txt"
    )
