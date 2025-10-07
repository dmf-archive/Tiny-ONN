---
title: "ADR-0008: 通过Goodness分数调制Adam动量以取代硬梯度掩码"
status: "Proposed"
date: "2025-10-07"
authors: "Ω Researcher, Tiny-ONN 课题组"
tags:
  [
    "architecture",
    "decision",
    "meta-learning",
    "adam",
    "momentum",
    "goodness",
    "gradient-flow",
  ]
supersedes: ""
superseded_by: ""
---

# ADR-0008: 通过 Goodness 分数调制 Adam 动量以取代硬梯度掩码

## 状态 (Status)

**Proposed** | Accepted | Rejected | Superseded | Deprecated

## 背景 (Context)

当前系统通过“硬梯度掩码”（`grad.mul_(goodness_mask)`）实现知识保护，即将 ReLU 过滤后的`Goodness`分数与`mu_weight`梯度逐元素相乘。该机制虽有效阻止了“坏”专家的更新，但本质是一种**二元的、硬性的**干预，可能过度抑制梯度流，导致路由决策僵化，并引入数值不稳定性。

鉴于 Adam 优化器本身具备**自适应学习率**（`m_t / sqrt(v_t)`）机制，能够根据梯度统计量自动调节参数更新幅度，我们提出一种更**软性的、连续的**替代方案：直接用`Goodness`信号**调制 Adam 的内部动量缓冲区**，从而将元学习效用信号深度整合到优化器的“记忆”中。

## 决策 (Decision)

我们决定实施一个**候选方案**，将当前的硬梯度掩码，替换为基于**ReLU 过滤前**的`latent_goodness`信号对**Adam 动量缓冲区**的直接调制。

具体机制如下：

1. **信号源**：使用`all_goodness_logits`（即`synergistic_benefit / learning_cost`），这是 ReLU 过滤前的、包含负值的**完整效用光谱**。
2. **调制对象**：在`optimizer_comp.step()`执行前，访问每个 SPL 模块`mu_weight`和`mu_bias`对应的 Adam 状态，调制其`exp_avg`（一阶动量）和`exp_avg_sq`（二阶动量）。
3. **调制公式**：`buffer *= max(0, mean_batch_seq(goodness_logits))`。取批次和序列维度的均值，得到标量乘子，确保动量缩放是**稳定且连续的**。

此方案将效用信号从“外部梯度修正”转变为“内部优化器状态调节”，赋予 Adam 对专家学习速度的**元认知控制权**。

## 后果 (Consequences)

### 积极 (Positive)

- **POS-001**: **软性干预**：避免硬掩码的数值突变，提供更平滑、更稳定的梯度流。
- **POS-002**: **深度整合**：将`Goodness`信号注入优化器“记忆”，使学习速度本身成为可优化的元参数。
- **POS-003**: **理论优雅**：利用 Adam 现有机制，避免引入新的硬性约束，符合奥卡姆剃刀原则。

### 消极 (Negative)

- **NEG-001**: **实现复杂度**：需要手动操作优化器内部状态，增加代码复杂性和调试难度。
- **NEG-002**: **超参数敏感性**：调制强度（如`max(0, ...)`）可能成为一个需要微调的新超参数。
- **NEG-003**: **风险未知**：这是一种非常规的优化器干预，其长期训练动态和稳定性尚未验证。

## 考虑的备选方案 (Alternatives Considered)

### 方案 A: 维持硬梯度掩码 (现状)

- **ALT-001**: **描述 (Description)**: 继续使用`grad.mul_(goodness_mask)`进行硬性梯度阻断。
- **ALT-002**: **拒绝理由 (Rejection Reason)**: 过于生硬，可能抑制合法的微小更新，导致路由决策失去细粒度调节能力。

### 方案 B: 动态学习率缩放

- **ALT-003**: **描述 (Description)**: 为不同`Goodness`水平的专家分组设置不同的`lr_mult`。
- **ALT-004**: **拒绝理由 (Rejection Reason)**: 需要手动划分`Goodness`区间并调参，不如直接调制动量来得精细和自适应。

### 方案 C: 不干预，完全信任 Adam

- **ALT-005**: **描述 (Description)**: 移除所有梯度干预，让 Adam 的`m_t / sqrt(v_t)`完全自主地调节学习速度。
- **ALT-006**: **拒绝理由 (Rejection Reason)**: Adam 的统计量只能反映**历史梯度**的稳定性，无法感知我们计算出的**前瞻效用**(`Goodness`)，会损失元学习的主动性。

## 实施注意事项 (Implementation Notes)

- **IMP-001**: 实施将集中在`exp/arc/train.py`的`LearningDynamics.compute_and_apply_gradients`方法中，移除硬掩码代码，添加动量调制逻辑。
- **IMP-002**: 必须严格使用**ReLU 前**的`goodness_logits`，以保留完整的效用光谱（包括负值）。
- **IMP-003**: 需要通过`observer`监控**梯度范数**和**有效学习率**(`m_t / sqrt(v_t)`)在不同`Goodness`水平专家上的分布，验证调制是否按预期工作。

## 参考文献 (References)

- **REF-001**: `docs/adr/adr-0005-sars-goodness-gradient-flow-control.md` (当前硬梯度掩码实现)
- **REF-002**: PyTorch DeepWiki 关于手动修改 Adam 内部状态的查询结果
- **REF-003**: `exp/arc/train.py` (拟议的实施位置)
