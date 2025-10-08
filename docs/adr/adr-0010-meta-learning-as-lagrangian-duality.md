---
title: "ADR-0010: 元学习作为拉格朗日对偶 (Meta-Learning as Lagrangian Duality)"
status: "Accepted"
date: "2025-10-08"
authors: "Ω Researcher, Tiny-ONN 课题组"
tags: ["architecture", "decision", "meta-learning", "lagrangian", "sparsity", "vfe"]
supersedes: "adr-0009-gate-semantic-correction-and-meta-loss-simplification.md"
superseded_by: ""
---

# ADR-0010: 元学习作为拉格朗日对偶 (Meta-Learning as Lagrangian Duality)

## 状态 (Status)

Proposed | **Accepted** | Rejected | Superseded | Deprecated

## 背景 (Context)

在 `Tiny-ONN-ARC` 项目中，所有先前为实现动态概念稀疏性而设计的元学习机制均遭遇了灾难性的失败。这些失败的共同根源在于，我们试图通过启发式方法，从不稳定的、瞬时的**后验梯度信号**（如 `∇_mu_weight`，`∇_output`）中，去构造一个本应是稳定**先验**的理想路由目标分布 `Q`。这种方法在理论上是循环论证（用学习结果定义学习目标），在数值上则表现为无法控制的梯度爆炸与 `main_loss` 崩溃。项目陷入了一个深刻的理论矛盾：我们需要一个 `L_meta` 来替代 ELBO 中的 `KL[q(z|x) || p(z)]` 项，但我们不知道如何稳定地构造其目标 `p(z)`。

## 决策 (Decision)

我们决定彻底废弃所有基于“分布对齐”（如 JSD）和启发式 `Goodness` 函数的元学习框架。我们将动态稀疏路由问题从第一性原理出发，重新形式化为一个**基于拉格朗日力学的约束优化问题**。

新的元学习框架由以下核心组件构成：

1. **理论重构**: “Lean-Max”原则（用最少的激活，实现最大的预测精度）被视为一个约束优化问题：最小化主损失 `L_main(z)`，同时最小化激活数量 `||z||_0`。其拉格朗日函数为 `ℒ(z, λ) = L_main(z) + λ * ||z||_0`。`L_meta` 的新角色是让网络隐式地学会这个 `ℒ`。

2. **目标 `Q` 的重新定义**: 理想的路由目标 `Q` 不再是概率分布，而是**净效用 (Net Utility)** `U`。
    - **边际收益 `B_i`**: 激活神经元 `i` 对 `L_main` 的预期降低量，由其**最终输出贡献**的梯度绝对值定义：`B_i = |∇_{masked_output_i} L_main|`。
    - **边际成本 `C_marginal`**: 激活任何标准计算单元的抽象成本，定义为一个无超参数的常数 `1`。
    - **净效用 `U_i`**: `U_i = B_i - C_marginal`。
    - **最终目标 `Q_target`**: `Q_target = ReLU(U_i)`。`ReLU` 完美实现了“仅在有利可图时激活”的经济学直觉。

3. **损失 `L_meta` 的重新定义**: `L_meta` 从分布对齐损失，转变为一个简单的**回归损失**。
    - **策略 `P`**: 模型当前的路由激活概率，`P(z_i=1|x) = sigmoid(routing_logits_i)`。
    - **损失函数 `L_meta`**: 均方误差 (MSE) 用于让策略 `P` 回归目标效用 `Q`。
        `L_meta = MSE(P, Q_target.detach())`
    - 对 `Q_target` 使用 `.detach()` 彻底切断了理论自指循环。

## 后果 (Consequences)

### 积极 (Positive)

- **POS-001**: **理论完备**: 框架直接源自约束优化的拉格朗日形式，为“Lean-Max”原则提供了坚实的数学基础，彻底解决了理论循环问题。
- **POS-002**: **无超参数稀疏性**: 系统无需任何关于稀疏度的超参数。最优激活数量 `k*` 将作为所有 `U_i > 0` 的神经元的集合，动态、内容感知地涌现。
- **POS-003**: **数值稳定性**: 抛弃了所有复杂的梯度组合，仅使用 `|∇|`, `ReLU`, `MSE` 等数值上极其稳健的操作，并切断了不稳定的梯度反馈回路，预期将从根本上解决 `NaN/Inf` 问题。
- **POS-004**: **高度可解释性**: `L_meta` 的每个组件（边际收益、边际成本、净效用）都有清晰、直观的经济学解释，极大地增强了模型行为的可分析性。

### 消极 (Negative)

- **NEG-001**: **实现依赖**: 该框架的成功，高度依赖于我们能否精确捕获 `masked_output` 的逐令牌梯度，这在 `ADR-0004` 中已有解决方案。
- **NEG-002**: **潜在的梯度消失**: 如果在训练早期，所有神经元的边际收益 `B_i` 都小于 `1`，那么 `Q_target` 将恒为零，可能导致 `L_meta` 无法提供有效的学习信号。这需要通过监控来验证。

## 考虑的备选方案 (Alternatives Considered)

### 方案 A: 启发式 `Goodness` 函数 (历史方案)

- **ALT-001**: **描述 (Description)**: 使用后验梯度（如 `∇_mu`, `∇_output`）的复杂组合来构造一个目标分布 `Q`，并使用 JSD 等分布散度作为 `L_meta`。
- **ALT-002**: **拒绝理由 (Rejection Reason)**: 历史已反复证明，此方案存在不可调和的理论循环和数值不稳定性，是所有训练失败的根源。

### 方案 B: 时间分离的贝叶斯更新

- **ALT-003**: **描述 (Description)**: 使用历史梯度信号的移动平均（EMA）或参数统计量来构造一个时间上稳定的 `Q`。
- **ALT-004**: **拒绝理由 (Rejection Reason)**: 这是一种过于复杂的间接方法，其本质仍是用后验信号调节路由，并未触及问题的核心。它在概念上不如直接让优化器（如 Adam）的动量机制来平滑梯度统计量来得简洁。

### 方案 C: 直接优化自由能代理

- **ALT-005**: **描述 (Description)**: 将 `L_meta` 定义为变分自由能 `F` 的一个可计算代理，例如 `F = NLL + λ_1 * ||z||_1 + λ_2 * ||∇_θ||^2`。
- **ALT-006**: **拒绝理由 (Rejection Reason)**: 直接惩罚稀疏性（如 `||z||_1`）违背了 `ADR-0002` 确立的“稀疏性是内容感知的涌现项”原则。同时，这会引入难以调整的超参数 `λ_i`，与项目的核心目标相悖。

## 实施注意事项 (Implementation Notes)

- **IMP-001**: **核心实现**: 集中在 `exp/arc/train.py` 的 `_calculate_goodness_jit` 函数，需要将其完全重写以实现新的净效用 `Q_target` 计算。
- **IMP-002**: **损失函数变更**: 需要将 `_calculate_jsd_loss` 替换为一个计算 `MSE` 损失的新函数。
- **IMP-003**: **梯度捕获**: 必须确保 `masked_output` 的梯度被正确捕获，以计算边际收益 `B_i`。这可能需要调整 `model.py` 中的 `register_hook` 位置。
- **IMP-004**: **监控**: `Observer` 必须密切监控 `B_i`、`U_i` 和最终 `Q_target` 的分布，以验证理论是否按预期工作，特别是是否存在梯度消失问题。

## 参考文献 (References)

- **REF-001**: `docs/adr/adr-0009-gate-semantic-correction-and-meta-loss-simplification.md` (被此 ADR 取代)
- **REF-002**: `docs/adr/adr-0004-per-token-gradient-extraction-for-meta-learning.md` (梯度捕获机制)
- **REF-003**: `docs/adr/adr-0002-inference-space-conceptual-sparsity.md` (稀疏性理论基础)
