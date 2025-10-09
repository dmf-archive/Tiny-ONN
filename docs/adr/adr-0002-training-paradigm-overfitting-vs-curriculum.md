---
title: "ADR-0002: 从单任务过拟合转向顺序课程学习的训练范式"
status: "Proposed"
date: "2025-10-09"
authors: "Ω Researcher"
tags: ["architecture", "decision", "training", "meta-learning"]
supersedes: ""
superseded_by: ""
---

# ADR-0002: 从单任务过拟合转向顺序课程学习的训练范式

## 状态 (Status)

**Proposed** | Accepted | Rejected | Superseded | Deprecated

## 背景 (Context)

当前的训练范式 (`P_current`) 基于 `Key-in-Lock` 优化理论（见 `ARC-Theory.md`），对每个任务的单一数据增强视图（view）进行深度过拟合（最多 100 步），旨在为 Surprise-Aware Routing Shaping (SARS) 元学习机制提供一个纯净、稳定的梯度信号。然而，持续的训练日志和评估结果明确表明，这种方法导致了严重的灾难性遗忘。模型在评估的“阶段 1：遗忘检查”中频繁失败，表明它无法保留先前任务中学到的知识，从而阻碍了任何形式的泛化能力。

## 决策 (Decision)

我们将废弃当前的单任务/视图过拟合范式，并采纳一种新的顺序课程学习范式 (`P_proposed`)。具体变更如下：

1. 训练将以 `batch_size=1` 的形式进行。
2. `DataLoader` 将严格按照任务序列长度（从短到长）的顺序提供数据，不进行洗牌（`shuffle=False`）。
3. 移除 `train.py` 中的内部过拟合循环（即 `for step in range(100):`），每个视图只训练一个固定的、较小的步数（例如，5 步）。

此决策旨在将训练目标从“解决单个任务”转变为“从任务序列中推断通用规则”。

## 后果 (Consequences)

### 积极 (Positive)

- **POS-001**: **缓解遗忘 (Mitigate Forgetting)**: 通过持续向模型展示新任务，强制其学习可跨任务泛化的通用规则，而不是对特定任务的细节进行过度记忆，从而直接解决观察到的灾难性遗忘问题。
- **POS-002**: **提升训练吞吐量 (Increase Throughput)**: 极大缩短了在单个任务上花费的时间，使得模型能够在相同时间内接触到更多样化的数据，加速了对整个 ARC 数据集分布的学习。
- **POS-003**: **对齐推理目标 (Align with Inference Goal)**: 训练过程将更接近 ARC 任务的本质——即从范例中快速推断规则，而不是深度学习。

### 消极 (Negative)

- **NEG-001**: **梯度信号不稳定 (Gradient Signal Instability)**: 每个训练步骤都来自一个新任务，这将导致 `L_main` 的梯度向量剧烈波动。这是与 `P_current` 设计初衷最根本的背离。
- **NEG-002**: **元学习风险 (Meta-Learning Risk)**: 高方差的梯度信号可能严重污染 `goodness_logits` 的计算，使其变为噪声。这可能导致 SARS 元学习机制完全失效，无法形成有效的动态路由策略。
- **NEG-003**: **学习深度不足 (Insufficient Learning Depth)**: 对于需要复杂多步推理的任务，单步训练可能不足以让模型收敛到有意义的解，可能只会学到一些表面模式。

## 考虑的备选方案 (Alternatives Considered)

### 维持当前范式

- **ALT-001**: **描述 (Description)**: 继续使用单任务/视图过拟合策略，并尝试调整超参数。
- **ALT-002**: **拒绝理由 (Rejection Reason)**: 大量的实验证据（持续的“遗忘检查”失败）已经表明，该范式在根本上存在缺陷，无法实现泛化，调整超参数不太可能解决此核心问题。

### 梯度缓冲/平均

- **ALT-003**: **描述 (Description)**: 从多个不同的任务中收集梯度，然后取平均值进行一次参数更新。
- **ALT-004**: **拒绝理由 (Rejection Reason)**: 此方案直接违反了 `Key-in-Lock` 理论。ARC 任务的损失地形极不连续，不同任务的梯度方向可能完全相反。平均这些梯度可能会产生一个无效的、指向平坦高损失区域的更新向量，阻碍模型在任何任务上收敛。

## 实施注意事项 (Implementation Notes)

- **IMP-001**: 核心修改将在 `exp/arc/train.py` 的 `_train_epoch` 函数中进行。
- **IMP-002**: 必须移除内部训练循环和 `selected_views` 逻辑。
- **IMP-003**: 成功的关键衡量标准是“阶段 1：遗忘检查”的通过率。次要监控指标包括 `meta_loss` 的稳定性和 `GBS%` 指标，以评估对元学习的影响。

## 参考文献 (References)

- **REF-001**: `.roo/rules/ARC-Theory.md`
- **REF-002**: `.roo/rules/DFC-Theory.md`
