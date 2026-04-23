import random
import pandas as pd
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from datetime import datetime, timedelta

def reinforcement_learning_algorithm(charging_entities, requests, output_dir,
                                     episodes=100, alpha=0.1, gamma=0.9, epsilon=0.1,
                                     random_seed=42):
    """
    使用Q-Learning(包含实体可用时间的状态表示)进行充电调度示例。

    参数:
    -------------------------------------------------------------------------
    charging_entities : list
        每个元素是一个 ChargingEntity 对象，包含:
        - name, latitude, longitude, charging_power等基本信息
        - process_request(request, is_eval=False/True)，其中:
          is_eval=False会更新内部状态(如 expected_available_time)；
          is_eval=True仅返回模拟结果不更新。
    requests : pd.DataFrame
        请求列表，至少含:
        - request_time
        - energy_needed
        - latitude
        - longitude
    output_dir : str
        输出CSV的目录。
    episodes : int
        Q-Learning的训练轮数。
    alpha : float
        学习率。
    gamma : float
        折扣因子。
    epsilon : float
        ϵ-贪心中的探索概率。
    random_seed : int
        随机种子，保证可重复。

    输出:
    -------------------------------------------------------------------------
    无直接返回；会在 output_dir 目录下生成:
    1) ev_requests_with_assignments.csv
    2) cs_with_assignments.csv
    """

    # 1) 固定随机数种子
    random.seed(random_seed)
    np.random.seed(random_seed)

    # 2) 对请求按时间排序，并记住原始索引以便输出对齐
    requests_sorted = requests.copy()
    requests_sorted['original_index'] = requests_sorted.index
    requests_sorted = requests_sorted.sort_values(by='request_time').reset_index(drop=True)
    num_requests = len(requests_sorted)
    num_entities = len(charging_entities)

    # 基准时间：我们把最早的请求时间当做 time=0
    min_request_time = pd.to_datetime(requests_sorted['request_time'].min())

    def time_to_int_minutes(t: datetime) -> int:
        """
        将一个 datetime 转换为相对 min_request_time 的分钟数(整数)。
        """
        delta = t - min_request_time
        return int(round(delta.total_seconds() / 60.0))

    # 3) 定义访问 Q 的函数(使用字典)。键为 (state, action)，值为标量。
    #    state = (r_idx, (avail_0, avail_1, ..., avail_(N-1)))，action = 实体索引
    Q = {}

    def get_Q_value(state, action):
        return Q.get((state, action), 0.0)

    def set_Q_value(state, action, value):
        Q[(state, action)] = value

    # 4) 状态构造辅助
    def build_state(r_idx, local_entities):
        """
        构造状态: (r_idx, (a0, a1, ..., aN-1)),
        其中 ai 是实体 i 的 expected_available_time 与 min_request_time 的差值(分钟).
        """
        # 如果 r_idx >= num_requests，说明没有后续请求，返回特殊终止状态
        if r_idx >= num_requests:
            return ('DONE',)  # 或任意标记终止状态

        # 收集实体的可用时间(相对 min_request_time, 取整数分钟)
        avail_times = []
        for ent in local_entities:
            if ent.expected_available_time is None:
                # 说明还没被占用过, 可看作立即可用(相对基准为0)
                avail_times.append(0)
            else:
                # 转为 int 分钟
                diff = ent.expected_available_time - min_request_time
                avail_minutes = int(round(diff.total_seconds() / 60.0))
                avail_times.append(avail_minutes)

        # 最终状态元组
        return (r_idx, tuple(avail_times))

    # 5) 训练循环: 多个 episode
    for ep in tqdm(range(episodes)):
        # (1) 深拷贝实体，初始化状态
        local_entities = [deepcopy(e) for e in charging_entities]
        for ent in local_entities:
            ent.expected_available_time = None
            ent.start_time = None
            ent.total_occupied_time = timedelta(0)
            ent.total_time = timedelta(0)

        # (2) 从第0个请求开始
        r_idx = 0
        state = build_state(r_idx, local_entities)

        # (3) 依次对请求做分配，直到处理完所有请求
        while True:
            # 如果是终止状态(DONE)就停止
            if state[0] == 'DONE':
                break

            # ϵ-贪心选动作
            if random.random() < epsilon:
                action = random.randint(0, num_entities - 1)
            else:
                # 选使 Q(state, a) 最大的动作
                q_vals = [get_Q_value(state, a) for a in range(num_entities)]
                action = int(np.argmax(q_vals))

            # (4) 进行一次分配: 调用 local_entities[action].process_request(...) 真正更新状态
            req_row = requests_sorted.iloc[r_idx]
            result = local_entities[action].process_request(req_row, is_eval=False)

            # 计算即时奖励
            travel_t = result.get('travel_time', 0)
            queue_t = result.get('queue_time', 0)
            charge_t = result.get('charging_time', 0)
            # 时间越久 => 惩罚越大 => reward越小
            reward = - (travel_t + queue_t + charge_t)

            # (5) 得到下一个状态
            next_r_idx = r_idx + 1
            next_state = build_state(next_r_idx, local_entities)

            if next_state[0] == 'DONE':
                # 没有后续 Q 值
                future_q = 0.0
            else:
                # max(Q(next_state, a'))
                future_q = max(get_Q_value(next_state, a) for a in range(num_entities))

            # (6) Q-Learning 更新
            old_q = get_Q_value(state, action)
            new_q = old_q + alpha * (reward + gamma * future_q - old_q)
            set_Q_value(state, action, new_q)

            # (7) 移动到下一请求
            r_idx = next_r_idx
            state = next_state

    # ----------------------------------------------------------------------------
    # 6) 训练完成后: 用训练得到的 Q 字典，对真实 charging_entities 进行最终分配
    #    (从头开始处理所有请求, 并更新其状态).
    # ----------------------------------------------------------------------------
    for ent in charging_entities:
        ent.expected_available_time = None
        ent.start_time = None
        ent.total_occupied_time = timedelta(0)
        ent.total_time = timedelta(0)

    results = [None] * num_requests

    # 状态初始化
    r_idx = 0
    state = build_state(r_idx, charging_entities)

    while True:
        if state[0] == 'DONE':
            break

        # 对每个请求选最优动作(贪心)
        q_vals = [get_Q_value(state, a) for a in range(num_entities)]
        best_action = int(np.argmax(q_vals))

        req_row = requests_sorted.iloc[r_idx]
        final_result = charging_entities[best_action].process_request(req_row, is_eval=False)

        final_result['request_time'] = req_row['request_time']
        final_result['energy_needed'] = req_row['energy_needed']
        final_result['request_latitude'] = req_row['latitude']
        final_result['request_longitude'] = req_row['longitude']
        final_result['charging_entity_name'] = charging_entities[best_action].name
        final_result['request_index'] = req_row['original_index']

        results[r_idx] = final_result

        # 下一个状态
        r_idx += 1
        state = build_state(r_idx, charging_entities)

    # ----------------------------------------------------------------------------
    # 7) 输出到CSV
    # ----------------------------------------------------------------------------
    result_df = pd.DataFrame(results).sort_values(by='request_index')
    result_df.to_csv(f"{output_dir}/ev_requests_with_assignments.csv", index=False)

    entity_data = []
    for entity in charging_entities:
        info = {
            'name': entity.name,
            'charging_power': entity.charging_power,
            'latitude': entity.latitude,
            'longitude': entity.longitude,
            'occupancy_rate': entity.occupancy_rate,
            'total_occupied_time': entity.total_occupied_time.total_seconds() / 60.0,
            'total_time': entity.total_time.total_seconds() / 60.0
        }
        entity_data.append(info)

    entity_df = pd.DataFrame(entity_data)
    entity_df.to_csv(f"{output_dir}/cs_with_assignments.csv", index=False)
