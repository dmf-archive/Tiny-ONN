# PLSD: 从 Q-Learning 到自监督推测解码的演进

> 状态: 先导实验中
> 核心目标: 实现全局资源共享的动态递归推理，通过 PLSD 范式最小化推断空间的计算熵。

## 0. 序言：计算复杂度的动态对齐

在深度学习的静态范式中，模型深度 `L` 是超参数。然而，任务的 Kolmogorov 复杂度 `K(x)` 是动态的。**Adaptive Computation Time (ACT)** 的核心使命是寻找一个映射 `t = ϕ(x)`，使得计算资源与任务难度实现帕累托最优。

本文将深入探讨从 TRM 的 **QLearning-ACT-v1** 到 **PLSD-ACT-v2** 的范式转移，并结合最新的学术成果阐述 PLSD 的独特性。

## 1. QLearning-ACT-v1 (TRM 方案)

### 1.1 工作方式

TRM (Tiny Recursive Model) 将递归步长的决策建模为一个**马尔可夫决策过程 (MDP)**。

- **状态 (State)**: 当前递归步的隐藏表征 `hₜ`。
- **动作 (Action)**: `A ∈ {Halt, Continue}`。
- **奖励 (Reward)**: 仅在停止步给出，基于最终预测的准确性。

其核心实现是一个 [`q_head`](src/models/dynsiha/recursive/modeling_recursive_dynsiha.py)，输出 `Q(hₜ, Halt)` 和 `Q(hₜ, Continue)`。训练采用类似 **DQN** 的自举 (Bootstrapping) 逻辑：
`Q(hₜ, Continue) ≈ σ( max(Q(hₜ₊₁, Halt), Q(hₜ₊₁, Continue)) )`

### 1.2 历史渊源与局限

- **渊源**: 继承自 Graves (2016) 的原始 ACT，但引入了强化学习中的 Q-learning 思想以增强决策的非线性。
- **局限性**:
    1. **信号稀疏**: 只有在停止步才能获得真实的 Loss 信号，中间步骤处于“盲飞”状态。
    2. **训练不稳定**: Q-learning 的不稳定性在递归架构中被放大，容易陷入“始终早停”或“始终跑满”的平凡解。
    3. **非单调性**: 递归过程中的 Loss 并不总是单调下降，Q-learning 难以捕捉复杂的 Loss 震荡地形。

## 2. PLSD-ACT-v2 (Per-Layer Speculative Decode)

### 2.1 核心理念：自监督内化与推测对齐

**PLSD (每层推测解码)** 放弃了强化学习的试错范式，转向**全量观测的自监督对齐**。其灵感源于大模型的推测解码 (Speculative Decoding)，但将其内化为架构的训练协议。

与 DEL [2] 等动态退出技术不同，PLSD 不依赖于推理时的接受率统计，而是通过训练时的 **Oracle 步长对齐** 实现“Teacher Forcing 的时间扩展”。

### 2.2 形式化推导

在训练阶段，我们强制模型展开至最大深度 `T_max`，并对每一层进行“虚拟解码”：

1. **生成损失序列**:
   对于输入 `x` 和标签 `y`，生成序列：
   `ℒ_seq = { ℓ₁, ℓ₂, ..., ℓ_T_max }`
   其中 `ℓₜ = CrossEntropy(LMHead(hₜ), y)`。

2. **定义 Oracle 步长 (最优解)**:
   `t* = argmin_{t ∈ [1, T_max]} ℓₜ`
   `t*` 代表了模型在当前参数下，能够达到的**认知饱和点**。这一概念在 Duo-LLM [1] 中被用于分析 Token 难度，而在 PLSD 中被直接作为监督信号。

3. **自监督对齐损失**:
   我们训练一个轻量级的 [`halt_head`](src/models/dynsiha/recursive/modeling_recursive_dynsiha.py) `H(hₜ)`，使其拟合一个阶跃函数：
   `Targetₜ = 0 if t < t* else 1`
   `ℒ_ACT = Σₜ BCE(H(hₜ), Targetₜ)`

### 2.3 训练范式：Teacher Forcing 的时间扩展

PLSD 最贴近的训练范式是 **Teacher Forcing** 在递归维度上的扩展。我们不再让模型“猜”什么时候停止，而是直接告诉它：“在这一步，你的 Loss 已经降到最低了，请记住这个状态特征”。这种“强制退出”监督的思想在 LYNX [3] 中也有所体现，但 PLSD 通过全局 `argmin` 提供了更强的全局最优性保证。

## 3. 理论优势对比

| 特性 | QLearning-ACT-v1 | PLSD-ACT-v2 | 学术参考 |
| :--- | :--- | :--- | :--- |
| **监督密度** | 极稀疏 (仅停止步) | **极稠密 (全路径观测)** | LYNX [3] |
| **优化目标** | 期望奖励最大化 | **认知饱和点对齐** | Duo-LLM [1] |
| **计算开销** | 较低 (单次前向) | 较高 (训练需全量展开) | LayerSkip [5] |
| **收敛速度** | 慢，需大量探索 | **极快，直接拟合 Loss 地形** | DEL [2] |
| **MDL 兼容性** | 弱 (需显式惩罚项) | **强 (天然寻找最短有效路径)** | SpecExit [4] |

## References

[1] K. Alizadeh et al., "Duo-LLM: A Framework for Studying Adaptive Computation in Large Language Models," *arXiv preprint arXiv:2410.10846*, 2024. [Online]. Available: <https://arxiv.org/abs/2410.10846>
[2] H. E. Zarch et al., "DEL: Context-Aware Dynamic Exit Layer for Efficient Self-Speculative Decoding," *arXiv preprint arXiv:2504.05598*, 2025. [Online]. Available: <https://arxiv.org/abs/2504.05598>
[3] Ö. F. Akgül et al., "LYNX: Learning Dynamic Exits for Confidence-Controlled Reasoning," *arXiv preprint arXiv:2512.05325*, 2025. [Online]. Available: <https://arxiv.org/abs/2512.05325>
[4] R. Yang et al., "SpecExit: Accelerating Large Reasoning Model via Speculative Exit," *arXiv preprint arXiv:2509.24248*, 2025. [Online]. Available: <https://arxiv.org/abs/2509.24248>
[5] M. Elhoushi et al., "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding," *arXiv preprint arXiv:2404.16710*, 2024. [Online]. Available: <https://arxiv.org/abs/2404.16710>
