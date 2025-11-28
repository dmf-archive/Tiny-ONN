---
title: "PI-Star (PI-*): First-Principles Active Inference Planning"
category: "Algorithm"
status: "🟡 进行中"
priority: "高"
timebox: "2 周"
created: 2025-11-28
updated: 2025-11-28
owner: "Omega"
tags: ["技术预研", "Algorithm", "Active Inference", "Planning", "PI-Star"]
---

# PI-Star (PI-\*): First-Principles Active Inference Planning

## 摘要

**探索目标 (Spike Objective):**
形式化设计 **PI-Star (Predictive Integrity A\*)** 算法，作为一种基于第一性原理（FEP/IPWT）的确定性规划框架，旨在替代传统的 MCTS/RL 方案，解决未知环境中的探索与利用问题。

**重要性 (Why This Matters):**
目前的 RL/MCTS 方法依赖于随机采样（Rollouts）和标量奖励信号，这在哲学上与 Tiny-ONN 的“逻辑不可约性”和“预测编码”核心相悖。PI-Star 旨在通过最小化**观测自由能 (OFE)** 来统一探索（Epistemic）与利用（Pragmatic），为 System 2（规划）提供理论自洽的实现。

**时限 (Timebox):** 2 周

**决策截止日期 (Decision Deadline):** 2025-12-12

## 研究问题 (Research Question(s))

**主要问题 (Primary Question):**
如何将观测自由能 (OFE) 分解为 A\* 搜索的启发式函数 `h(n)`，并在无外部奖励的稀疏环境中利用“虚拟梯度”生成稠密的内在动机信号？

**次要问题 (Secondary Questions):**

- **计算可行性**: 如何在推理阶段（Inference-time）高效计算基于梯度的认知复杂度（Complexity/Surprise），而不进行实际的参数更新？
- **搜索空间**: 在参数流形（Parameter Manifold）诱导的动力学空间中进行 A\* 搜索的拓扑结构如何定义？
- **基准测试**: 如何构建一个最小可行的 GridWorld/Maze 实验，既能验证 PI-Star 的有效性，又能体现其相对于 MCTS 的理论优势？

## 理论设计 (Theoretical Design)

### 核心假设：规划即生成 (Planning as Generation)

PI-Star 将规划视为在生成模型的**信念空间**（Belief Space）上寻找一条**观测自由能 (OFE) 最小**的测地线。不同于 MCTS 的随机模拟，PI-Star 是确定性的、梯度引导的。

### 1. 自由能的 A\* 分解

标准 A\* 算法最小化 `f(n) = g(n) + h(n)`。在 PI-Star 中：

- **`g(n)` (已实现的自由能)**: 从起点到当前节点 `n` 的累积路径成本。

  - `g(n) = Σₜ₌₀ⁿ (D_KL[Q(sₜ)||P(sₜ)] + 𝔼_Q[ln P(oₜ|sₜ)])`
  - _物理意义_: 这一步走了多远（能耗）以及这一步有多“意外”（模型不确定性）。

- **`h(n)` (观测自由能 OFE)**: 从节点 `n` 到目标的预估成本。
  - `h(n) ≈ G(π) = Risk + Ambiguity`
  - **Pragmatic Value (Risk)**: 当前状态与偏好状态（Goal）的距离。`≈ ||s_n - s_goal||²` 或 `D_KL(P(o|s_n) || P(o_pref))`.
  - **Epistemic Value (Ambiguity)**: 预期的信息增益（负项）。在 A\* 中，我们倾向于去往能最大程度减少不确定性的区域。
  - _修正启发式_: `h(n) = Risk - λ · InfoGain`。

### 2. 虚拟梯度 (Fictitious Gradients) 与内在动机

在缺乏外部奖励（Sparse Reward）的环境中，传统的 $h(n)$ 会失效。PI-Star 引入**虚拟梯度**作为内在动机的度量：

`Curiosity(s_{t+1}) ≈ || ∇_θ ℒ(s_{t+1}) ||`

- **关键前提**: 必须存在一个**虚拟损失函数** `ℒ`，其梯度能反映模型在**参数空间**中的敏感度（Fisher Information）。
- **虚拟损失设计**: 采用 **Expected Gradient Length (EGL)** 的思想。
  - 对于一个状态 `s`，模型预测其输出分布 `P(o|s)`。
  - 从分布中采样（或取期望）得到伪标签 `ô ~ P(o|s)`。
  - 计算标准预测损失 `ℒ = -log P(ô|s)`。
  - 这里的 `ℒ` 实际上是在探测：如果我把这个预测当作真实发生的事实，我的世界观（参数 `θ`）需要受到多大的“冲击”？
  - *修正*: 不需要额外的对比/重建头。直接利用主干网络的预测头，计算其对参数的梯度范数。这直接度量了状态 $s$ 在参数流形上的**认知显著性**。
- **原理**: 如果模型对状态 $s_{t+1}$ 感到“惊讶”，则该状态产生的梯度范数会很大。
- **操作**: 在规划模拟步骤中，执行一次 `loss.backward()` 但**不执行** `optimizer.step()`。
- **用途**: 将梯度范数作为 $h(n)$ 中的负项（奖励），引导搜索向“高认知价值”区域探索。

### 3. 算法流程 (Draft)

```python
def PI_Star_Search(start_state, goal_prior):
    open_set = PriorityQueue()
    open_set.push(start_state, priority=0)

    while not open_set.empty():
        current = open_set.pop()

        if is_goal(current):
            return reconstruct_path(current)

        # 展开潜在动作
        for action in action_space:
            # 1. 预测下一状态 (World Model)
            next_state = model.predict(current, action)

            # 2. 计算 G值 (累积自由能)
            risk = distance(next_state, goal_prior)

            # 3. 计算 H值 (观测自由能)
            # 关键：通过虚拟梯度计算认知复杂度
            complexity = compute_fictitious_gradient_norm(next_state)

            # OFE = Risk (Pragmatic) - Curiosity (Epistemic)
            OFE = risk - lambda * complexity

            f_score = g_score[current] + OFE
            open_set.push(next_state, priority=f_score)
```

## 调查计划

### 现有资源分析 (External Resources)

- **pymdp**: Python Active Inference 库。
  - _发现_: 实现了基于 OFE 的规划，但主要针对离散状态空间，且未利用梯度信息。
  - _借鉴_: 其 GridWorld 环境和 OFE 计算公式（Risk + Ambiguity）可作为基准。
- **Contrastive Active Inference**:
  - _借鉴_: 使用对比学习避免像素级重建，降低 OFE 计算成本。

### 实验设计 (Experimental Design)

我们需要一个支持梯度计算的最小化环境。

1. **环境**: `DifferentiableGridWorld` (基于 PyTorch)。
   - 输入: `(x, y)` 坐标的 One-hot 或 连续向量。
   - 动态: 简单的确定性或随机转移。
   - 观测: 状态本身（简化）或状态的噪声映射。
2. **模型**: 小型 MLP 或 Transformer World Model。
   - 预训练或在线学习环境动态。
3. **任务**:
   - **Sparse Reward**: 迷宫导航，只有到达终点才有信号。
   - **Trap**: 局部极小值。
4. **对比组**:
   - Random Walk
   - Standard A\* (仅使用欧氏距离作为 heuristic)
   - MCTS (基于随机 Rollout)
   - **PI-Star** (基于 OFE + 虚拟梯度)

### 成功标准

- [ ] 在 PyTorch 中实现 `compute_fictitious_gradient_norm` 并验证其与“模型惊讶度”的正相关性。
- [ ] PI-Star 在稀疏奖励迷宫中比标准 A\* 更快找到目标（因为内在动机会引导它探索未知区域，而不是在死胡同徘徊）。
- [ ] 证明 PI-Star 路径的 OFE 显著低于基线。

## 决策

### 建议

优先开发 **PyTorch-based GridWorld** 原型，而非直接使用 `pymdp`，因为我们需要访问模型的梯度 (`autograd`) 来计算 PI 指标，而 `pymdp` 是基于 NumPy 的离散实现，难以直接集成梯度动力学。

### 下一步行动

- [ ] 创建 `exp/pi_star_spike/` 目录。
- [ ] 实现一个简单的 `DifferentiableGridWorld` 类。
- [ ] 实现一个简单的 World Model (MLP)。
- [ ] 编写 `fictitious_gradient` 计算原语。

## 状态历史

| 日期       | 状态      | 备注                       |
| :--------- | :-------- | :------------------------- |
| 2025-11-28 | 🟡 进行中 | 文档创建，初步理论框架完成 |

---

_上次更新时间：2025-11-28，由 Omega_
