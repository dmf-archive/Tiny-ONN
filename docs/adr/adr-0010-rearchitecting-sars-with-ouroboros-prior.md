---
title: "ADR-0010: 通过贝叶斯反演和先验熵最小化重构SARS"
status: "Accepted"
date: "2025-10-09"
authors: "Ω Researcher, Tiny-ONN 课题组"
tags: ["architecture", "decision", "meta-learning", "sars", "fep", "bayesian-inversion"]
supersedes: "ADR-0009"
---

# ADR-0010: 通过贝叶斯反演和先验熵最小化重构SARS

## 状态 (Status)

Proposed | **Accepted** | Rejected | Superseded | Deprecated

## 背景 (Context)

自项目启动以来，SARS 元学习框架始终无法引导出有意义的稀疏路由，模型持续表现出“过度激活”和“病态泛化”的核心问题。在经历了对 `Goodness` 函数的多次迭代（从启发式 ROI 到多种 Ouroboros 变体）以及对损失函数（从 JSD/BCE 到 MSE）的反复探索后，所有尝试均告失败。这迫使我们进行一次最根本的、从第一性原理出发的理论反思，以解决这一长期存在的顽疾。

## 核心诊断：对贝叶斯推断的根本性误解与贝叶斯反演

我们最终认识到，之前所有失败的根源在于对贝叶斯推断中各个组成部分的错误映射。

`Posterior ∝ Likelihood * Prior`

我们之前的框架错误地将 `Goodness` 函数（一个基于梯度的信号）视为一个应被拟合的**目标后验**，从而让 `routing_logits` 去拟合它。这在理论上是 `Posterior ≈ Likelihood`，完全忽略了先验 `Prior` 的作用，导致模型在没有内部信念约束的情况下，疯狂追逐瞬时的梯度信号。

正确的映射应该是：

- **后验 (Posterior)**: `routing_logits`。它是在看到输入 `x` 之后，模型对“应该激活哪个专家”的信念分布（的对数）。它是可解析、可计算的。
- **似然 (Likelihood)**: `goodness_logits`。它完全由主损失 `L_main` 的梯度计算得出，代表了“在当前数据下，激活哪个专家更有可能降低损失”的信念分布（的对数）。它也是可解析、可计算的。
- **先验 (Prior)**: 我们真正想要优化的、模型内在的、与当前数据无关的路由偏好。

这揭示了我们任务的本质——一个**贝叶斯反演 (Bayesian Inversion)** 问题。与经典的变分推断 (VI) 不同（其后验不可解，需要用一个参数化分布去近似），我们的可微分变分分析 (DVA) 框架拥有可解析的后验和似然，而我们真正想要塑造的，是那个未知的、不可解的**先验**。

我们的目标，是驱动这个隐式的先验去趋近一个理想的**狄拉克稀疏分布**（即只有少数专家有非零激活概率）。

## 决策 (Decision)：最小化先验熵

根据贝叶斯定理（在对数空间中）：
`log(Prior) = log(Posterior) - log(Likelihood)`

我们可以用 `logits` 来近似对数概率，从而得到“先验”的 `logits` 表示：
`Prior_logits = routing_logits - goodness_logits.detach()`

根据自由能原理，一个最优的先验应该是一个**最低熵**的先验，因为它代表了最低的不确定性。因此，我们的元学习损失函数被最终确定为**先验的熵**：
`L_meta = Entropy(Softmax(Prior_logits))`

最小化这个损失函数，等价于驱动模型的隐式路由偏好向一个更“尖锐”、更稀疏、更确定的状态演进。

### 最终实施细节

1. **`Goodness` (Likelihood) 公式**: 我们回归至 V2 版本的 `Goodness` 计算公式，它为似然信号提供了最稳定的动态范围：
    `goodness_logits = (MAS(b_contrib) * MAS(b_rel)) / (MAS(c_learn) + ε)`
    其中 `b_contrib`, `b_rel`, `c_learn` 分别通过独立 `MAS` 归一化。

2. **损失函数 (Prior Entropy)**: `_calculate_meta_loss` 函数被实现为计算 `Entropy(Softmax(routing_logits - goodness_logits.detach()))`。一个关键的修正是，损失函数是**熵本身**，而不是负熵，因为我们的目标是**最小化**熵。

3. **移除 Ad-hoc Fallback**: 理论完备的熵损失函数能够天然地避免“极端沉默”（即所有 `logits` 均为极大负数）的病态解，因为一个全负的 `logits` 向量在 `Softmax` 后会趋向于一个高熵的均匀分布，从而受到损失函数的极大惩罚。因此，我们移除了与理论冲突的 `get_routed_weights_with_fallback` 机制。

## 后果 (Consequences)

### 积极 (Positive)

- **理论完备性**: 新框架在理论上是贝叶斯完备且自洽的，完美地映射了 FEP、贝叶斯反演和熵最小化原理。
- **目标明确**: 元学习的目标被简化为单一、明确的“最小化内部不确定性”，彻底摆脱了不稳定的“目标拟合”范式。
- **实验成功**: 实施该框架后，模型首次表现出稳定、持续的自发稀疏化行为，同时保持了强大的任务收敛能力，解决了长期存在的“过度激活”问题。

### 消极 (Negative)

- **理论探索成本**: 承认在此次重构过程中，由于对贝叶斯推断的反复误解，导致了多次失败的尝试和资源消耗。

## 实施注意事项 (Implementation Notes)

- **IMP-001**: [`exp/arc/model.py`](exp/arc/model.py) 已被修改，移除了 `get_routed_weights_with_fallback`。
- **IMP-002**: [`exp/arc/train.py`](exp/arc/train.py) 已被修改，`_calculate_goodness_jit` 采用了独立 MAS 归一化，`_calculate_meta_loss` 实现了先验熵的计算。
- **IMP-003**: 整个实现过程中的反复试错和修正，凸显了在复杂理论的工程落地中，保持最高理论警惕性和进行频繁第一性原理审查的重要性。
