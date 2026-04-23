import random
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from datetime import timedelta

def genetic_algorithm(charging_entities, requests, output_dir,
                      population_size=100, generations=100,
                      crossover_rate=0.8, mutation_rate=0.01,
                      elite_size=1, random_seed=42):
    """
    使用遗传算法为充电实体与请求进行分配。

    参数:
    --------
    charging_entities : list
        充电实体对象列表，需包含以下方法和属性：
        - name: str
        - latitude: float
        - longitude: float
        - charging_power: float
        - occupancy_rate: float
        - total_occupied_time: timedelta
        - total_time: timedelta
        - is_available(request): bool  (可选，示例中直接用 process_request 内部的排队逻辑)
        - calculate_distance(request): float (计算实体到请求的地理距离)
        - process_request(request): dict (处理请求并返回结果字典, 会更新实体的状态)
    requests : pandas.DataFrame
        每行表示一个充电请求，需包含以下列：
        - request_time
        - energy_needed
        - latitude
        - longitude
    output_dir : str
        保存结果 CSV 文件的目录路径。
    population_size : int
        初始种群大小。
    generations : int
        演化的代数。
    crossover_rate : float
        交叉概率。
    mutation_rate : float
        变异概率。
    elite_size : int
        精英保留数量，用于保留适应度最高的个体。
    random_seed : int
        随机种子，保证可复现性。

    输出:
    --------
    无直接返回。会在 output_dir 目录下生成两个文件：
    1) ev_requests_with_assignments.csv: 每条请求的分配结果（分配的实体、各类时间等信息）
    2) cs_with_assignments.csv: 充电实体的状态信息
    """

    # ========== 固定随机种子，保证可复现性 ==========
    random.seed(random_seed)
    np.random.seed(random_seed)

    num_requests = requests.shape[0]
    num_entities = len(charging_entities)

    # ========== 工具函数定义 ==========

    def create_individual():
        """
        创建一个个体（解），用长度为 num_requests 的列表表示，
        列表的每个元素是充电实体的索引（0 ~ num_entities-1）。
        """
        return [random.randint(0, num_entities - 1) for _ in range(num_requests)]

    def evaluate_individual(individual):
        """
        评估个体的适应度。
        这里示例一个“总等待时间 + 总行驶距离”的加权组合作为目标，
        你可以根据需求替换为其它形式:
          - total_waiting_time (含 travel_time + queue_time + charging_time)
          - total_distance
        或者只针对某个指标做单目标优化等。

        评价逻辑:
        1. 拷贝一份 charging_entities 的列表，称为 local_entities，
           每个 local_entity 都有自己的初始状态 (expected_available_time=None 等)。
        2. 按 request_time 先后来处理请求:
           - 找到该请求对应的实体(由 individual 决定)
           - 调用 process_request() 来获得实际的 travel_time, queue_time, charging_time 等
        3. 统计总的等待时间 (waiting_time) 和总距离，用以衡量方案好坏。
        4. 注意: 因为我们要最小化“等待+距离”，而遗传算法通常是最大化适应度，
           所以这里返回 -score 作为适应度(越小的 score -> 越大的 -score)。
        """
        # 深拷贝，避免修改原 charging_entities
        local_entities = [deepcopy(e) for e in charging_entities]

        total_waiting = 0.0
        total_distance = 0.0

        # 顺序遍历每个请求(按照时间先后)
        for req_idx in range(num_requests):
            assigned_entity_idx = individual[req_idx]
            entity = local_entities[assigned_entity_idx]

            request_row = requests.iloc[req_idx]
            # 计算距离
            dist = entity.calculate_distance(request_row)
            total_distance += dist

            # 调用 process_request 来获得等待时间 (包括行驶 + 排队 + 充电)
            result = entity.process_request(request_row)
            total_waiting += result['waiting_time']

        # 这里示例: 将“总等待时间 + 总距离”作为 score
        # 也可以只用总等待时间(含travel+queue+charging)或者只用distance，都根据业务需求调整
        score = total_waiting + total_distance

        # 遗传算法里通常是“越大越好”，所以返回 -score
        return -score

    def selection(population, fitness_scores):
        """
        选择操作: 示例采用“精英保留 + 轮盘赌”。
        """
        # 将个体(及其分数)一起按分数从大到小排序
        pop_with_fit = list(zip(population, fitness_scores))
        pop_sorted = sorted(pop_with_fit, key=lambda x: x[1], reverse=True)

        # 保留精英
        new_population = [x[0] for x in pop_sorted[:elite_size]]
        new_scores = [x[1] for x in pop_sorted[:elite_size]]

        # 剩余个体通过轮盘赌产生
        # 先对分数做 shift，让其全为正值
        shifted = [f - min(fitness_scores) + 1e-6 for f in fitness_scores]
        total_fit = sum(shifted)

        for _ in range(len(population) - elite_size):
            pick = random.uniform(0, total_fit)
            current = 0
            for i, f in enumerate(shifted):
                current += f
                if current >= pick:
                    new_population.append(population[i])
                    new_scores.append(fitness_scores[i])
                    break

        return new_population, new_scores

    def crossover(parent1, parent2):
        """
        单点交叉示例。
        """
        if random.random() > crossover_rate:
            return parent1, parent2

        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def mutation(individual):
        """
        变异：以 mutation_rate 的概率，随机改动基因(即某请求分配的实体)。
        """
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[i] = random.randint(0, num_entities - 1)
        return individual

    # ========== 1) 初始化种群 ==========
    population = [create_individual() for _ in range(population_size)]

    # ========== 2) 开始遗传迭代 ==========
    for gen in tqdm(range(generations), desc="Genetic Algorithm"):
        fitness_scores = [evaluate_individual(ind) for ind in population]

        # 选择
        population, fitness_scores = selection(population, fitness_scores)

        # 交叉
        new_population = []
        for i in range(0, population_size, 2):
            parent1 = population[i]
            parent2 = population[i+1] if i+1 < population_size else population[0]
            child1, child2 = crossover(parent1, parent2)
            new_population.append(child1)
            new_population.append(child2)

        # 变异
        mutated_population = [mutation(ind) for ind in new_population]

        # 更新
        population = mutated_population

    # ========== 3) 找到最终最优个体并生成结果输出 ==========
    final_fitness_scores = [evaluate_individual(ind) for ind in population]
    best_idx = np.argmax(final_fitness_scores)  # 因为分数是负值，越大越好
    best_individual = population[best_idx]

    # ========== 4) 用最佳个体做实际调度并输出 CSV ==========
    # 用来存储每条请求处理后的结果
    results = [None] * num_requests

    for req_idx in range(num_requests):
        assigned_entity_idx = best_individual[req_idx]
        entity = charging_entities[assigned_entity_idx]

        request_row = requests.iloc[req_idx]
        result = entity.process_request(request_row)  # 包含travel_time, queue_time, charging_time, waiting_time, end_time等

        # 补充一些信息
        result['request_time'] = request_row['request_time']
        result['energy_needed'] = request_row['energy_needed']
        result['request_latitude'] = request_row['latitude']
        result['request_longitude'] = request_row['longitude']
        result['charging_entity_name'] = entity.name
        result['request_index'] = requests.index[request_row.name]

        results[req_idx] = result

    # 为了与 greedy_algorithm 的输出对齐：我们希望 ev_requests_with_assignments.csv 中的行顺序
    # 与原请求的顺序一致，所以先转成 DataFrame，再按 request_index 排序
    result_df = pd.DataFrame(results).sort_values(by='request_index')
    result_df.to_csv(f"{output_dir}/ev_requests_with_assignments.csv", index=False)

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
    entity_df.to_csv(f"{output_dir}/cs_with_assignments.csv", index=False)
