---
title: "ADR-0009: 基于协同效用模型的元学习重构"
status: "Accepted"
date: "2025-10-08"
authors: "Ω Researcher, Tiny-ONN 课题组"
tags:
  [
    "architecture",
    "decision",
    "meta-learning",
    "goodness-function",
    "vfe",
    "synergistic-utility",
    "ib-jsd",
  ]
supersedes: "ADR-0009 (V1: Gate 语义修正与元学习损失简化)"
superseded_by: ""
---

# ADR-0009: 基于协同效用模型的元学习重构

## 状态 (Status)

Proposed | **Accepted** | Rejected | Superseded | Deprecated

## 背景 (Context)

在 `ADR-0009` 的先前版本（V1）中，我们致力于从 VFE 第一性原理推导出一个理论完备的 `Goodness` 函数。经过最终的理论研讨，我们发现其仍然未能完全捕捉到元学习的核心矛盾：**先验信念**与**后验证据**之间的关系。

V1 版本的 `Goodness` 函数是基于启发式构造的，缺乏坚实的理论基础。我们需要一个理论上更坚实的基础来定义 `Goodness` 分布，以解决“先验 vs. 后验”的核心矛盾。

## 决策 (决策)

我们决定实施一个最终的、理论上最自洽的元学习框架，其核心是**协同效用模型 (Synergistic Utility Model)**。

1. **明确 P 与 Q 的分离**:

    - **先验信念 P (待优化的策略)**: 严格定义为 `routing_logits`。这是元学习梯度下降的直接目标。
    - **后验效用 Q (Goodness / 目标信号)**: 严格定义为由后验梯度信号构成的“证据”。

2. **定义协同效用 `Goodness` 函数 (Q)**: `Goodness` 分数被定义为**先验信念**与**后验信噪比 (Posterior Signal-to-Noise Ratio)** 的协同作用。一个专家只有在同时满足“被信任”、“有效”和“低成本”三个条件时，才被认为是“好的”。

    `Q_goodness = routing_logits ⊙ (abs(∇_masked_output L_main) / (abs(∇_mu_weight L_main) + ε))`

    其中 `⊙` 代表哈达玛积 (element-wise product)，因为它正确地表达了三个独立条件之间的“逻辑与”关系。

3. **确认元学习损失函数**: 采用**独立伯努利 JSD (IB-JSD)** 作为分布散度度量 `D`。最终的元学习损失为:

    `L_meta = D_IB-JSD(sigmoid(routing_logits) || Q_goodness)`

## 后果 (Consequences)

### 积极 (Positive)

- **POS-001**: **理论的最终统一**: 此模型最终完美地统一了 VFE 框架、ROI 直觉和“先验 vs. 后验”的逻辑关系，解决了我们之前所有的理论困惑。
- **POS-002**: **高度选择性**: 协同效用模型具有极高的选择性，预期将强力推动专家功能分化，只奖励那些在所有维度（信念、效益、成本）上都表现出色的计算路径。
- **POS-003**: **可解释性**: 每个组件都有清晰、不可约的理论含义，使得模型行为的分析和调试变得更加容易。

### 消极 (Negative)

- **NEG-001**: **潜在的梯度消失/爆炸风险**: 三个张量的乘积可能导致最终的 `Goodness` 分数值域非常大或非常小，需要通过 `mas_normalize` 等手段进行仔细的数值稳定性控制。
- **NEG-002**: **反馈循环**: 将 `routing_logits` 同时用作 P 和 Q 的一部分，引入了一个直接的反馈循环。虽然理论上这是正确的（信念强化），但在实践中可能导致不稳定的“赢家通吃”或“输家恒输”动态。

## 实施注意事项 (Implementation Notes)

- **IMP-001**: **核心实现**: 集中在 `exp/arc/train.py` 的 `_calculate_goodness_jit` 函数。
- **IMP-002**: **梯度捕获**: 必须确保我们捕获了 `masked_outputs` 的梯度，并能访问到 `mu_weight` 的梯度。
- **IMP-003**: **数值稳定性**: `Goodness` 的计算结果必须经过 `mas_normalize` 处理后再输入到 IB-JSD 损失函数中，以确保其值域在 `[0, 1]` 之间。
- **IMP-004**: **密切监控**: `Observer` 必须监控 `routing_logits`、`output_grad` 和 `mu_grad` 各自的分布，以及最终 `Goodness` 分布的形态，以诊断潜在的反馈循环问题。

## 参考文献 (References)

- **REF-001**: 本 ADR 的所有先前版本和相关讨论记录。
