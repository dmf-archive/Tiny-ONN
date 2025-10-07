---
title: "ADR-0006: 通过SiLU激活函数增强概念细胞非线性表达力"
status: "Accepted"
date: "2025-10-07"
authors: "Ω Researcher, Tiny-ONN 课题组"
tags:
  [
    "architecture",
    "decision",
    "meta-learning",
    "sars",
    "jsd",
    "sparsity",
    "activation-function",
    "silu",
    "neuroscience",
  ]
supersedes: ""
superseded_by: ""
---

# ADR-0006: 通过 SiLU 激活函数增强概念细胞非线性表达力

## 状态 (Status)

Proposed | **Accepted** | Rejected | Superseded | Deprecated

## 背景 (Context)

在解决了路由回退机制后，模型的学习效果虽有提升，但路由稀疏性不足的问题依然存在。经过深入的理论研讨，我们认识到问题的根源可能不在于元学习信号本身，而在于**概念细胞（Concept Cell）**内部的计算表达能力。

当前的 `SparseProtoLinear (SPL)` 模块中，`computation_output` 的计算为 `F.linear(x, mu_weight, mu_bias)`，这是一个纯粹的线性变换。对于 ARC 这类需要复杂逻辑推理的任务，一个线性函数可能不足以捕捉到一个抽象规则的**适用前提**或执行复杂的变换。我们需要的不是增加新的组件，而是为现有的、理论正确的 `match-cost` 框架中的计算核心增加**非线性表达能力**。

## 决策 (Decision)

我们决定维持 `match-cost` 的神经科学模型不变，仅在 `computation_output` 的计算路径上引入平滑的非线性激活函数。

具体的修改是在 `spl_forward` 函数中，将 `computation_output` 的输入从原始的 `x` 替换为其非线性变换：

```python
# 修改前 (线性)
computation_output = F.linear(x, mu_weight, mu_bias)

# 修改后 (非线性)
computation_output = F.linear(F.silu(x), mu_weight, mu_bias)
```

### 理论依据

1. **神经可解释性**: 神经元的输出强度（发放率）是其输入信号的平滑、非线性函数，而非简单的线性叠加。使用平滑的激活函数（如 SiLU）比硬性的 ReLU 更能模拟这种生物学特性。
2. **增强表达能力**: 这等价于在不改变路由逻辑的前提下，将每个专家的计算能力从一个线性函数提升为了一个两层 MLP (`Linear(SiLU(Identity(x)))`)。这为概念细胞提供了学习更复杂、非线性变换的能力，从而更好地适应 ARC 任务的逻辑需求。
3. **最小化变更**: 这是一个极小的、外科手术式的改动，遵循奥卡姆剃刀原则。它不增加新的参数或复杂的结构，却可能带来表达能力上的显著提升。
4. **与现代架构对齐**: 调研显示，`SiLU` (Swish) 是 `Qwen2.5`、`Qwen3` 和 `DeepSeekV3` 等最新、最先进的开源架构中的主流选择，证明了其理论上的优越性和工程上的有效性。

### 区分：计算激活 vs. 路由激活

此决策的关键在于功能分离。我们只在**计算路径**上引入平滑的 `SiLU`，而**路由门控**（即 `raw_weights` 的计算）依然维持硬性的 `F.relu` 激活。

- **路由是决策 (Routing is Decision)**: 路由的目标是**选择**。这在神经科学上对应于“全有或无”的发放原则。一个决策应该是明确的、离散的，以产生稀疏的激活模式。硬性的 `ReLU` 完美地模拟了这一点，确保了决策的稀疏性和明确性。
- **计算是变换 (Computation is Transformation)**: 计算的目标是**处理**信息。这对应于神经元“发放率”的强度调制。这个过程是渐变的、非线性的。平滑的 `SiLU` 能够很好地模拟这种强度调制，产生更丰富、信息量更大的输出。

## 后果 (Consequences)

### 积极 (Positive)

- **POS-001**: **提升概念表达能力**: 为概念细胞提供了学习非线性变换的能力，使其能够捕捉更复杂的逻辑规则。
- **POS-002**: **符合生物学原理**: 使用平滑激活函数更好地模拟了神经元的真实发放特性。
- **POS-003**: **与现代架构对齐**: 使我们的设计与当前最先进的模型保持一致，站在了巨人的肩膀上。

### 消极 (Negative)

- **NEG-001**: **潜在的数值不稳定**: 引入非线性可能会使训练动态在初期变得更加复杂，需要仔细监控。
- **NEG-002**: **超参数敏感性**: 激活函数的选择可能会与现有的超参数（如学习率）产生微妙的相互作用，可能需要微调。

## 考虑的备选方案 (Alternatives Considered)

### 方案 A: 使用 GeLU (及其变体)

- **ALT-001**: **描述 (Description)**: 将 `SiLU` 替换为 `GeLU` (特别是其 `tanh` 近似版本 `NewGELU`)。
- **ALT-002**: **拒绝理由 (Rejection Reason)**: 虽然 `GeLU` 同样是一个优秀的候选者，且与随机正则化有深刻的理论联系，但 `SiLU` 已被证明是**最先进架构的共识选择**。在没有明确证据表明 `GeLU` 显著优于 `SiLU` 的情况下，维持当前已被验证有效的 `SiLU` 实现是更稳妥的选择。如果未来 `SiLU` 遇到瓶颈，我们可以再考虑切换。

### 方案 B: 增加网络深度 (例如，SwiGLU)

- **ALT-003**: **描述 (Description)**: 将 `mu_weight` 的计算路径重构为一个更深的 MLP，例如 `SwiGLU`。
- **ALT-004**: **拒绝理由 (Rejection Reason)**: 这会增加模型的复杂性和参数数量，违背了“最小化变更”的原则。当前的方案旨在通过一个极小的改动获得最大的理论收益。

## 实施注意事项 (Implementation Notes)

- **IMP-001**: 实施集中在 `exp/arc/model.py` 的 `spl_forward` 函数中，将 `F.silu(x)` 替换为 `F.gelu(x, approximate="tanh")`。
- **IMP-002**: 需要通过 `observer` 密切监控训练动态，特别是 `main_loss` 和 `PI-Score`，以验证新激活函数是否按预期工作。

## 参考文献 (References)

- **REF-001**: `docs/adr/adr-0005-sars-goodness-gradient-flow-control.md`
- **REF-002**: `docs/rules/DFC-Theory.md`
- **REF-003**: DeepWiki 查询结果：最新开源架构（Qwen2.5, Qwen3, DeepSeekV3）的激活函数选择。
