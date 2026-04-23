import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.spatial import KDTree

def greedy_algorithm(charging_entities, requests, output_dir, prob_param=0.7, gamma=58):
    results = []
    
    # Initialize positions and KDTree
    def build_tree(entities):
        positions = np.array([[e.latitude, e.longitude] for e in entities])
        return KDTree(positions), positions
    
    # Separate MCS and FCS
    mcs_entities = [e for e in charging_entities if e.entity_type == 'mcs']
    fcs_entities = [e for e in charging_entities if e.entity_type == 'fcs']
    
    mcs_tree, mcs_positions = build_tree(mcs_entities) if mcs_entities else (None, None)
    fcs_tree, fcs_positions = build_tree(fcs_entities) if fcs_entities else (None, None)
    
    slow_fcs = [e for e in fcs_entities if e.electricity_fee == 1]
    slow_tree, slow_positions = build_tree(slow_fcs) if slow_fcs else (None, None)

    entity_list = charging_entities[:]  # Copy for indexing
    
    # 使用 tqdm 包装 requests.iterrows() 以显示进度条
    for index, request in tqdm(requests.iterrows(), total=requests.shape[0], desc="Processing requests"):
        req_pos = np.array([request['latitude'], request['longitude']])
        
        prob = np.random.random()

        if prob < prob_param:
            # Find closest available MCS
            closest_mcs = None
            cost_mcs = float('inf')
            if mcs_entities and mcs_tree:
                k = min(50, len(mcs_entities))
                dists_mcs, indices_mcs = mcs_tree.query(req_pos, k=k)
                candidates_mcs = [mcs_entities[i] for i in indices_mcs]
                available_mcs = [e for e in candidates_mcs if e.is_available(request)]
                if available_mcs:
                    closest_mcs, cost_mcs = min(
                        [(e, e.calculate_cost(request)) for e in available_mcs],
                        key=lambda x: x[1]
                    )
                else:
                    closest_mcs, cost_mcs = min(
                        [(e, e.calculate_cost(request)) for e in mcs_entities],
                        key=lambda x: x[1]
                    )

            # Find closest available FCS
            closest_fcs = None
            cost_fcs = float('inf')
            if fcs_entities and fcs_tree:
                k = min(50, len(fcs_entities))
                dists_fcs, indices_fcs = fcs_tree.query(req_pos, k=k)
                candidates_fcs = [fcs_entities[i] for i in indices_fcs]
                available_fcs = [e for e in candidates_fcs if e.is_available(request)]
                if available_fcs:
                    closest_fcs, cost_fcs = min(
                        [(e, e.calculate_cost(request)) for e in available_fcs],
                        key=lambda x: x[1]
                    )
                else:
                    closest_fcs, cost_fcs = min(
                        [(e, e.calculate_cost(request)) for e in fcs_entities],
                        key=lambda x: x[1]
                    )
            
            # 最低成本不小于gamma时，不进行分配
            if min(cost_mcs, cost_fcs) >= gamma:
                selected = None

            # Select based on cost probability
            if closest_mcs and closest_fcs:
                if cost_mcs == 0 and cost_fcs == 0:
                    selected = np.random.choice([closest_mcs, closest_fcs])
                else:
                    total_cost = cost_mcs + cost_fcs
                    p_mcs = cost_fcs / total_cost
                    p_fcs = cost_mcs / total_cost
                    selected = np.random.choice([closest_mcs, closest_fcs], p=[p_mcs, p_fcs])
            elif closest_mcs:
                selected = closest_mcs
            elif closest_fcs:
                selected = closest_fcs
            else:
                selected = None
        else:
            selected = None
            if slow_fcs and slow_tree:
                k = min(50, len(slow_fcs))
                dists_slow, indices_slow = slow_tree.query(req_pos, k=k)
                candidates_slow = [slow_fcs[i] for i in indices_slow]
                available_slow = [e for e in candidates_slow if e.is_available(request)]
                if available_slow:
                    closest_slow, cost_slow = min(
                        [(e, e.calculate_cost(request)) for e in available_slow],
                        key=lambda x: x[1]
                    )
                else:
                    closest_slow, cost_slow = min(
                        [(e, e.calculate_cost(request)) for e in candidates_slow],
                        key=lambda x: x[1]
                    )
                # 最低成本不小于gamma时，不进行分配
                if cost_slow >= gamma:
                    selected = None
                else:
                    selected = closest_slow

        # 处理请求
        if selected is not None:
            result = selected.process_request(request)
            result['request_time'] = request['request_time']
            result['energy_needed'] = request['energy_needed']
            result['request_latitude'] = request['latitude']
            result['request_longitude'] = request['longitude']
            result['request_index'] = index
            result['charging_entity_name'] = selected.name
            result['cost'] = result['time_related_cost'] + result['price_related_cost'] + result['mobility_related_cost']
        else:
            result = {
                'end_time': "2026-04-01 07:00:00",
            }
        results.append(result)
        
        # If the entity is MCS, position may have changed, rebuild tree
        if selected and selected.entity_type == 'mcs':
            mcs_tree, mcs_positions = build_tree(mcs_entities)
    
    # 将结果保存到 CSV 文件
    result_df = pd.DataFrame(results)
    result_df.to_csv("{}/ev_requests_with_assignments.csv".format(output_dir), index=False)
    
    # 保存充电实体的状态信息到文件
    entity_data = []
    for entity in charging_entities:
        entity_info = {
            'name': entity.name,
            'charging_power': entity.charging_power,
            'latitude': entity.latitude,
            'longitude': entity.longitude,
            'occupancy_rate': entity.occupancy_rate,
            'total_occupied_time': entity.total_occupied_time.total_seconds() / 60,  # 转换为分钟
            'total_time': entity.total_time.total_seconds() / 60  # 转换为分钟
        }
        entity_data.append(entity_info)
    
    entity_df = pd.DataFrame(entity_data)
    entity_df.to_csv("{}/cs_with_assignments.csv".format(output_dir), index=False)
