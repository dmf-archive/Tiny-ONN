# RDS-ACT: 从 Q-Learning 到自监督推测解码的演进

> **状态**: 生产基准 (Production Baseline)
> **核心目标**: 实现全局资源共享的动态递归推理，通过 PLSD 范式最小化推断空间的计算熵。

## 0. 序言：计算复杂度的动态对齐

在深度学习的静态范式中，模型深度 $L$ 是超参数。然而，任务的 Kolmogorov 复杂度 $K(x)$ 是动态的。**Adaptive Computation Time (ACT)** 的核心使命是寻找一个映射 $t = \phi(x)$，使得计算资源与任务难度实现帕累托最优。

本文将深入探讨从 TRM 的 **QLearning-ACT-v1** 到 **PLSD-ACT-v2** 的范式转移。

## 1. QLearning-ACT-v1 (TRM 方案)

### 1.1 工作方式

TRM (Tiny Recursive Model) 将递归步长的决策建模为一个**马尔可夫决策过程 (MDP)**。

- **状态 (State)**: 当前递归步的隐藏表征 $h_t$。
- **动作 (Action)**: $A \in \{Halt, Continue\}$。
- **奖励 (Reward)**: 仅在停止步给出，基于最终预测的准确性。

其核心实现是一个 `q_head`，输出 $Q(h_t, Halt)$ 和 $Q(h_t, Continue)$。训练采用类似 **DQN** 的自举 (Bootstrapping) 逻辑：
`Q(h_t, Continue) \approx \sigma( \max(Q(h_{t+1}, Halt), Q(h_{t+1}, Continue)) )`

### 1.2 历史渊源与局限

- **渊源**: 继承自 Graves (2016) 的原始 ACT，但引入了强化学习中的 Q-learning 思想以增强决策的非线性。
- **局限性**:
    1. **信号稀疏**: 只有在停止步才能获得真实的 Loss 信号，中间步骤处于“盲飞”状态。
    2. **训练不稳定**: Q-learning 的不稳定性在递归架构中被放大，容易陷入“始终早停”或“始终跑满”的平凡解。
    3. **非单调性**: 递归过程中的 Loss 并不总是单调下降，Q-learning 难以捕捉复杂的 Loss 震荡地形。

## 2. PLSD-ACT-v2 (Per-Layer Speculative Decode)

### 2.1 核心理念：稠密监督与推测对齐

**PLSD (每层推测解码)** 放弃了强化学习的试错范式，转向**全量观测的自监督对齐**。其灵感源于大模型的推测解码 (Speculative Decoding)，但将其内化为架构的训练协议。

### 2.2 形式化推导

在训练阶段，我们强制模型展开至最大深度 $T_{max}$，并对每一层进行“虚拟解码”：

1. **生成损失序列**:
   对于输入 $x$ 和标签 $y$，生成序列：
   `\mathcal{L}_{seq} = \{ \ell_1, \ell_2, \dots, \ell_{T_{max}} \}`
   其中 $\ell_t = \text{CrossEntropy}(\text{LMHead}(h_t), y)$。

2. **定义 Oracle 步长 (最优解)**:
   `t^* = \text{argmin}_{t \in [1, T_{max}]} \ell_t`
   $t^*$ 代表了模型在当前参数下，能够达到的**认知饱和点**。

3. **自监督对齐损失**:
   我们训练一个轻量级的 `halt_head` $H(h_t)$，使其拟合一个阶跃函数：
   `\text{Target}_t = \begin{cases} 0 & t < t^* \\ 1 & t \ge t^* \end{cases}`
   `\mathcal{L}_{ACT} = \sum_t \text{BCE}(H(h_t), \text{Target}_t)`

### 2.3 训练范式：Teacher Forcing 的时间扩展

PLSD 最贴近的训练范式是 **Teacher Forcing** 在递归维度上的扩展。我们不再让模型“猜”什么时候停止，而是直接告诉它：“在这一步，你的 Loss 已经降到最低了，请记住这个状态特征”。

## 3. 理论优势对比

| 特性 | QLearning-ACT-v1 | PLSD-ACT-v2 |
| :--- | :--- | :--- |
| **监督密度** | 极稀疏 (仅停止步) | **极稠密 (全路径观测)** |
| **优化目标** | 期望奖励最大化 | **认知饱和点对齐** |
| **计算开销** | 较低 (单次前向) | 较高 (训练需全量展开，推理可早停) |
| **收敛速度** | 慢，需大量探索 | **极快，直接拟合 Loss 地形** |
| **MDL 兼容性** | 弱 (需显式惩罚项) | **强 (天然寻找最短有效路径)** |

## 4. 工程优势指标 (KPI) 预测

为了量化 RDS-ACT 的性能，我们将监控以下指标：

1. **推断增益效率 (IGE)**:
   `IGE = \frac{\ell_1 - \ell_{t^*}}{\text{Avg}(t^*)}`
   衡量单位计算步长带来的平均 Loss 下降。
2. **早停准确率 (ESA)**:
   `ESA = \mathbb{P}(\text{ExitStep} = t^*)`
   衡量推理时的实际停止步与理论最优步的重合度。
3. **计算节省率 (CSR)**:
   `CSR = 1 - \frac{\mathbb{E}[t^*]}{T_{max}}`
   在 ARC 任务中，预期 CSR 应与任务的 Grid 复杂度呈负相关。

## 5. 预测性待办清单 (Predictive Todo)

- [x] **PLSD 基础框架实现**: 完成 `modeling_recursive_dynsiha.py` 中的损失序列捕获逻辑。
- [ ] **动态认知惩罚 (DCP)**: 实现随训练进程动态调节的 $\gamma$ 系数，初期鼓励探索，后期强制节俭。
- [ ] **KV-Cache 递归重用优化**: 针对递归步之间的 Attention 计算进行算子级优化，减少冗余显存读写。
- [ ] **ARC 复杂度相关性验证**: 在 `evaluation.py` 中增加任务复杂度与 $t^*$ 的散点图分析。
- [ ] **谱熵坍缩监控**: 集成 `LoROU` 探针，观测递归过程中隐藏状态的秩变化。
**Ω Researcher 签发**
**日期**: 2026-01-25
