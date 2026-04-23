# Project Context: Probabilistic EV Charging Demand Forecasting

## 1. 研究问题 (Research Problem)
- **核心痛点**：电动汽车（EV）充电需求具有高波动性、零值聚集（Zero-inflated）和时空依赖性。传统的确定性预测（Deterministic forecasting）只输出点估计，无法量化不确定性。在实际电网调度中，“高估需求（资源浪费）”和“低估需求（电网过载/停电）”的代价是不对称的，缺乏概率区间会导致严重的运营风险。
- **技术难点**：现有的深度分位数回归（Quantile Regression）通常会遇到“分位数交叉（Quantile Crossing，物理不合理）”以及“经验覆盖率未校准（Uncalibrated Coverage，统计不可靠）”的问题。

## 2. 当前方法 (Current Methodology)
我们提出了一个端到端的概率时空预测框架，包含三个核心模块：
1. **Backbone (GAT-Informer)**：结合图注意力网络（GAT）提取空间依赖，结合 Informer 提取长程时间依赖。
2. **Monotonic Quantile Head**：采用增量 Softplus 参数化（Incremental Softplus Parameterization）。基础分位数 $\tau_1$ 保证非负，后续分位数通过累加非负的 Softplus 输出得到，从数学架构上消除了分位数交叉现象。联合 Pinball Loss 进行多偏置训练。
3. **Conformal Calibration (CQR)**：应用共形分位数回归（Conformalized Quantile Regression），使用独立校准集计算非一致性分数（Non-conformity scores），对输出的预测区间进行免分布假设（Distribution-free）的有限样本校准，保证边缘覆盖率满足目标要求。

## 3. 当前已有结果 (Current Results)
- **数据集**：UrbanEV dataset (Shenzhen, China, 6个月小时级数据，1362个充电站)。
- **点预测精度（相比确定性 GAT-Informer）**：
  - MAPE：从 8.46% 降低至 5.44%
  - MAE：从 0.0939 降低至 0.0650
- **区间预测及校准效果（目标 90% 覆盖率, $\delta=0.1$）**：
  - 未校准时：PICP 为 62.21%（严重欠覆盖）。
  - CQR校准后：PICP 提升至 88.51%（逼近名义覆盖率），且平均区间宽度（MPIW）保持不变（0.2483）。
- **敏感性分析**：在 60%, 80%, 90% 不同目标覆盖率下，CQR均能紧密跟踪名义覆盖率。

## 4. 本次投稿目标 (Target of this Submission)
- **会议稿件类型**：6页左右的英文学术会议论文（如 IEEE 类别会议或数据挖掘 Short Paper）。
- **时间线**：距离截稿仅剩 **10天**。
- **叙事核心**：“Problem-Driven System”。不强调庞大的模型架构创新，而是强调**解决实际问题**：如何低成本地将确定性预测升级为无交叉、严格校准的概率预测，并最终服务于“风险感知（Risk-Aware）”的电网调度。

## 5. 当前约束 (Current Constraints)
1. **禁止修改核心网络架构**：没有时间进行大规模的重新训练和调参。
2. **复用现有预测结果**：尽量利用目前模型已经输出的预测值（Predictions）和真实值（Ground Truths）数组。
3. **代码开发重点**：后续代码需求仅限于**写脚本计算新指标（如经济性指标）**、**数据后处理**、**绘制能吸引审稿人的高质量图表**以及**极轻量级的对比基线**。

## 6. 候选扩展方向与代码需求 (Candidate Extensions & Code Requirements)
为了让论文的 "Risk-Aware" 故事闭环，并应对审稿人的质疑，接下来需要完成以下几个代码任务：

- [ ] **Task 1: 经济性/风险感知评估 (Newsvendor Model Evaluation)**
  - **需求**：基于当前的预测结果，引入不对称惩罚成本（配置过多成本 $c_o$ vs. 缺电惩罚成本 $c_u$）。计算使用确定性预测与使用不同分位数策略（激进、中立、保守）的“总运营惩罚成本”。
  - **目的**：用具体的经济指标证明该框架的“风险感知（Risk-Aware）”价值。

- [ ] **Task 2: 时变覆盖率分析 (Time-of-day Coverage Analysis)**
  - **需求**：按一天 24 小时聚合测试集结果，分别计算每个小时的 PICP 和 MPIW。
  - **目的**：画出双 Y 轴折线图，证明模型在平峰期和极具波动的晚高峰期，覆盖率（PICP）都稳定在 90% 附近，且宽度（MPIW）能够自适应变化。

- [ ] **Task 3: 轻量级基线与消融实验 (Baselines & Ablations)**
  - **需求 1**：实现基于历史经验分位数（Historical Quantiles）的 baseline，作为最基础的概率预测对比。
  - **需求 2**：使用纯 MAE (L1 Loss) 训练现有的 GAT-Informer 并测试。
  - **目的**：验证本模型的点精度提升是因为多分离位数的联合正则化，而不是单纯换了鲁棒的损失函数。