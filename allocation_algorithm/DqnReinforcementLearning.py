import random
import pandas as pd
import numpy as np
from copy import deepcopy
from datetime import timedelta
from tqdm import trange, tqdm

import torch
import torch.nn as nn
import torch.optim as optim

# ====== 一个简单的全连接网络，用于近似 Q(s) ======
class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNetwork, self).__init__()
        # 这里定义一个简单的2层全连接网络
        hidden_size = 64
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)  # 输出形状: [batch_size, action_dim]

# ====== 简易的经验回放池 ======
class ReplayMemory:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        # transition: (state, action, reward, next_state, done)
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def reinforcement_learning_dqn_algorithm(charging_entities, requests, output_dir,
                                         episodes=100, gamma=0.9, epsilon_start=1.0, 
                                         epsilon_end=0.01, epsilon_decay=0.995,
                                         lr=1e-3, batch_size=64, 
                                         random_seed=42):
    """
    使用 DQN(Deep Q-Network) 进行充电调度示例，状态中纳入“当前请求索引 + 所有实体可用时间”。

    参数:
    ----------------------------------------------------------------------------
    charging_entities : list
        充电实体对象列表，每个实体应包含:
        - name, latitude, longitude, charging_power, 等基本信息
        - process_request(request, is_eval=False/True): dict
          is_eval=False: 实际更新该实体状态(排队时间, expected_available_time等);
          is_eval=True : 只返回模拟分配结果, 不修改实体状态(本示例未使用).
    requests : pd.DataFrame
        请求数据, 包含:
        - request_time, energy_needed, latitude, longitude
    output_dir : str
        输出 csv 文件的目录.
    episodes : int
        训练的总episode数.
    gamma : float
        折扣因子.
    epsilon_start, epsilon_end, epsilon_decay : float
        ϵ-贪心中的初始ϵ, 最小ϵ, 衰减系数.
    lr : float
        神经网络的学习率.
    batch_size : int
        每次采样的batch大小.
    random_seed : int
        随机种子.

    输出:
    ----------------------------------------------------------------------------
    无直接返回，在 output_dir 下生成:
    1) ev_requests_with_assignments.csv
    2) cs_with_assignments.csv
    """

    # ========== 1) 固定随机数种子 ==========
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # ========== 2) 对请求按时间排序，并记录原始索引 ==========
    requests_sorted = requests.copy()
    requests_sorted['original_index'] = requests_sorted.index
    requests_sorted.sort_values(by='request_time', inplace=True)
    requests_sorted.reset_index(drop=True, inplace=True)
    num_requests = len(requests_sorted)
    num_entities = len(charging_entities)

    # 取最早请求时间作为 0 基准
    min_request_time = pd.to_datetime(requests_sorted['request_time'].min())

    def time_to_int_minutes(t):
        return int(round((pd.to_datetime(t) - min_request_time).total_seconds() / 60))

    # ========== 3) 构建 DQN 与优化器 ==========
    # 状态维度: 1 (r_idx) + num_entities (各实体可用时间) 
    # 动作维度: num_entities
    state_dim = 1 + num_entities
    action_dim = num_entities

    policy_net = DQNetwork(state_dim, action_dim)
    target_net = DQNetwork(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())  # 初始化 target_net = policy_net
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayMemory(capacity=10000)

    # ϵ-贪心参数
    epsilon = epsilon_start

    # ========== 4) 工具函数 ==========

    def build_state(r_idx, entity_list):
        """
        将当前请求索引 + 所有实体的 expected_available_time 拼成一个向量(state)。
        r_idx: int, 当前处理的请求索引
        entity_list: 充电实体列表 (带各自 expected_available_time)
        返回: shape = [state_dim] 的 numpy 或 PyTorch tensor
        """
        if r_idx >= num_requests:
            # 表示Episode结束后的特殊状态，这里用 None 表示一下
            return None
        
        # 将 r_idx 直接作为float使用(或做归一化)
        s = [float(r_idx)]

        for ent in entity_list:
            if ent.expected_available_time is None:
                # 说明还没被占用过, 相对于 min_request_time = 0
                s.append(0.0)
            else:
                diff = ent.expected_available_time - min_request_time
                s.append(float(round(diff.total_seconds() / 60.0)))
        return np.array(s, dtype=np.float32)

    def select_action(state_vec):
        """
        ϵ-贪心选择动作
        state_vec: shape=[state_dim], numpy
        """
        if random.random() < epsilon:
            # 随机探索
            return random.randint(0, action_dim - 1)
        else:
            with torch.no_grad():
                # 转成 torch tensor
                st = torch.from_numpy(state_vec).unsqueeze(0)  # shape=[1, state_dim]
                q_values = policy_net(st)  # shape=[1, action_dim]
                action = torch.argmax(q_values, dim=1).item()
            return action

    def optimize_model():
        """
        从经验池中采样一个 batch，进行一次DQN训练
        """
        if len(memory) < batch_size:
            return  # 不足一个batch则跳过

        transitions = memory.sample(batch_size)
        # transitions是一个列表，包含batch_size个 (state, action, reward, next_state, done)

        # 分拆数据
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        for (s, a, r, s_next, done) in transitions:
            state_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_state_batch.append(s_next)
            done_batch.append(done)

        # 转成 Torch 张量
        state_batch = torch.tensor(state_batch, dtype=torch.float32)
        action_batch = torch.tensor(action_batch, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1)

        # next_state 可能为 None(episode结束后), 需要处理
        # 用一个 mask 来区分
        non_final_mask = torch.tensor([s_next is not None for s_next in next_state_batch], dtype=torch.bool)
        non_final_next_states = torch.tensor([s_next for s_next in next_state_batch if s_next is not None],
                                             dtype=torch.float32)

        # 计算Q(s,a)
        q_values = policy_net(state_batch)  # shape=[batch_size, action_dim]
        q_sa = q_values.gather(1, action_batch)  # shape=[batch_size, 1]

        # 计算 target
        # Q target = r + gamma * max_a' Q'(s', a')  (如果非终止)
        next_q = torch.zeros(batch_size, 1)
        with torch.no_grad():
            # 用 target_net 来计算 next state's Q
            next_q_values = target_net(non_final_next_states)  # shape=[x, action_dim]
            max_next_q = torch.max(next_q_values, dim=1)[0].unsqueeze(1)  # shape=[x,1]
            next_q[non_final_mask] = max_next_q

        expected_q_sa = reward_batch + gamma * next_q

        # 计算 loss
        loss = nn.MSELoss()(q_sa, expected_q_sa)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ========== 5) 训练循环 ==========
    update_target_interval = 50  # 每隔多少个回合/step，同步一次target_net
    global_step = 0

    for ep in trange(episodes, desc="DQN Training"):
        # ---- 深拷贝实体，初始化状态 ----
        local_entities = [deepcopy(e) for e in charging_entities]
        for ent in local_entities:
            ent.expected_available_time = None
            ent.start_time = None
            ent.total_occupied_time = timedelta(0)
            ent.total_time = timedelta(0)

        r_idx = 0
        state_vec = build_state(r_idx, local_entities)

        done = False
        while not done:
            # 如果 state_vec=None 或 r_idx>=num_requests，说明episode结束
            if state_vec is None or r_idx >= num_requests:
                done = True
                # 最后一条经验, next_state=None
                # 不过我们已经无法做任何动作了，可以直接 break
                break

            action = select_action(state_vec)

            # 在本地实体上执行 process_request
            req_row = requests_sorted.iloc[r_idx]
            result = local_entities[action].process_request(req_row, is_eval=False)

            travel_t = result.get('travel_time', 0)
            queue_t = result.get('queue_time', 0)
            charge_t = result.get('charging_time', 0)
            # reward = - (等待总时间)
            reward = - (travel_t + queue_t + charge_t)

            next_r_idx = r_idx + 1
            if next_r_idx >= num_requests:
                # 下一个状态 = None，表示结束
                next_state_vec = None
                done = True
            else:
                next_state_vec = build_state(next_r_idx, local_entities)
                # 如果 build_state 返回 None，也表示 Episode结束(极端情况)
                if next_state_vec is None:
                    done = True

            # 存储到回放池
            memory.push((state_vec, action, reward, next_state_vec, done))

            # 转移到下一个
            state_vec = next_state_vec
            r_idx = next_r_idx
            global_step += 1

            # 调用优化
            optimize_model()

            # 定期更新 target_net
            if global_step % update_target_interval == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # 每回合结束后，降低 epsilon
        epsilon = max(epsilon_end, epsilon_decay * epsilon)

    # ========== 6) 训练完后，用学到的 policy_net 进行一次“真实分配”并输出结果 ==========
    # (从头开始，不再深拷贝，而是直接使用 charging_entities 的原对象或者再次清空)
    for ent in charging_entities:
        ent.expected_available_time = None
        ent.start_time = None
        ent.total_occupied_time = timedelta(0)
        ent.total_time = timedelta(0)

    results = [None] * num_requests

    r_idx = 0
    state_vec = build_state(r_idx, charging_entities)

    while True:
        if state_vec is None or r_idx >= num_requests:
            break

        # 贪心选择动作(不再使用随机探索)
        with torch.no_grad():
            st = torch.from_numpy(state_vec).unsqueeze(0)
            q_values = policy_net(st)
            best_action = torch.argmax(q_values, dim=1).item()

        # 真正分配
        req_row = requests_sorted.iloc[r_idx]
        final_result = charging_entities[best_action].process_request(req_row, is_eval=False)

        final_result['request_time'] = req_row['request_time']
        final_result['energy_needed'] = req_row['energy_needed']
        final_result['request_latitude'] = req_row['latitude']
        final_result['request_longitude'] = req_row['longitude']
        final_result['charging_entity_name'] = charging_entities[best_action].name
        final_result['request_index'] = req_row['original_index']

        results[r_idx] = final_result

        r_idx += 1
        state_vec = build_state(r_idx, charging_entities)

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
