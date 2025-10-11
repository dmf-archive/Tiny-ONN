---
title: "ADR-0009: 采用二值掩码（硬门控）进行动态路由"
status: "Accepted"
date: "2025-10-11"
authors: "Ω Researcher"
tags: ["architecture", "decision", "sars", "spl", "gradient-dynamics"]
supersedes: ""
superseded_by: ""
---

# ADR-0009: 采用二值掩码（硬门控）进行动态路由

## 状态 (Status)

Proposed | **Accepted** | Rejected | Superseded | Deprecated

## 背景 (Context)

在 `Tiny-ONN` 的 `SparseProtoLinear (SPL)` 模块中，动态路由机制最初采用 `F.relu(routing_logits)` 作为门控信号。这种“软门控”机制将路由决策（`routing_logits`）的数值强度与计算路径的输出（`computation_output`）直接相乘。理论推演和实验观察表明，这种设计导致了几个根本性问题：

1. **梯度信号混淆 (Gradient Signal Conflation)**: 主任务梯度（来自 `L_main`）和元学习梯度（旨在优化路由本身）被混合在一起。路由权重的数值直接影响了主任务梯度的量级，使得归因变得不精确。
2. **违反神经生物学原则 (Violation of Neurobiological Principles)**: `Tiny-ONN` 的一个核心设计原则是模拟神经元的“全有或无”发放机制。软门控允许模拟神经元的“连接决策”影响“发放率”，这在理论上是不纯粹的。
3. **对System 1/System 2的模糊划分**: 我们的元学习框架（如 `DFC-Theory` 中所述）将主任务学习视为 System 1（快速、直觉的计算），将路由决策的优化视为 System 2（缓慢、反思的元认知）。软门控模糊了这两个系统之间的界限，因为 System 2 的状态（路由权重）直接泄漏并干扰了 System 1 的计算过程。

因此，需要一种新的门控机制来彻底隔离这两个系统，确保梯度流的纯粹性，并更严格地遵守我们的核心设计原则。

## 决策 (Decision)

我们决定将 `SPL` 模块中的动态路由机制从软门控（`computation_output * F.relu(routing_logits)`）修改为二值掩码的硬门控（`computation_output * (routing_logits > 0).float()`）。

这一决策的核心是将路由机制的功能严格限定为**结构决策**（连接或断开），而将**数值计算**完全交给主计算路径。路由参数 (`proto_weight`, `gate_param`) 的梯度将**完全且仅**来源于元学习损失 (`L_meta`)，而计算参数 (`mu_weight`, `mu_bias`) 的梯度将**完全且仅**来源于主任务损失 (`L_main`)。

由于元学习优化路径不依赖于主任务梯度的反向传播，此硬门控实现不需要 `Straight-Through Estimator (STE)` 或其他梯度代理。

## 后果 (Consequences)

### 积极 (Positive)

- **POS-001**: **梯度解耦 (Gradient Decoupling)**: 实现了主任务梯度与元学习梯度的完全隔离。计算参数的学习目标变得纯粹且明确，而路由参数的学习也基于更清晰的归因信号。
- **POS-002**: **归因精度提升 (Improved Attribution Accuracy)**: SARS 的 `goodness_logits` 信号变得更加精确。它现在衡量的是一个计算路径在被激活时对任务的“纯粹”贡献，而不再被其自身的激活强度所扭曲，这使得元学习能够做出更准确的结构优化决策。
- **POS-003**: **强化专家专业化 (Enhanced Expert Specialization)**: “全有或无”的梯度更新机制对 SPL 神经元施加了更强的专业化压力，有利于形成功能更分化、更稀疏、更稳定的专家网络结构。
- **POS-004**: **理论一致性 (Theoretical Coherence)**: 该决策与 `DFC-Theory` 中关于 System 1/System 2 分离、本地化赫布学习以及神经生物学保真度的核心原则高度一致。

### 消极 (Negative)

- **NEG-001**: **探索减少 (Reduced Exploration)**: 硬门控可能比软门控更早地关闭某些计算路径，可能会减少模型在早期训练阶段的探索能力。然而，元学习机制旨在动态地重新打开必要的路径，有望缓解此问题。
- **NEG-002**: **潜在的训练不稳定性 (Potential Training Instability)**: 离散的结构变化可能比平滑的权重调整引入更大的训练动态变化。需要通过元梯度裁剪等稳定性措施来确保训练过程平稳。

## 考虑的备选方案 (Alternatives Considered)

### 维持软门控 (`F.relu`)

- **ALT-001**: **描述 (Description)**: 维持 `computation_output * F.relu(routing_logits)` 的现有实现。
- **ALT-002**: **拒绝理由 (Rejection Reason)**: 此方案与我们的核心理论相悖，并已被证明会导致梯度混淆和归因不精确的问题，阻碍了模型性能的进一步提升。

### 采用 STE (Straight-Through Estimator)

- **ALT-003**: **描述 (Description)**: 在前向传播中使用硬门控，但在反向传播中通过 STE 传递一个代理梯度，允许主任务梯度流向路由参数。
- **ALT-004**: **拒绝理由 (Rejection Reason)**: 这违背了我们“System 1/System 2 梯度隔离”的核心原则。路由参数的优化应完全由元学习（System 2）负责，而不应受到主任务（System 1）的直接影响。我们的元学习框架使得 STE 变得毫无必要。

## 实施注意事项 (Implementation Notes)

- **IMP-001**: 已在 `exp/arc/model.py` 的 `MoIETransformerBlock` 中完成实现，将 `masked_routing_logits` 替换为 `mask_active`。
- **IMP-002**: 需要持续监控训练日志中的 `GBS%` (Goodness/Badness/Shutdown) 和 `Act%` (Activation Rate) 等指标，以验证新的梯度动力学是否符合理论预期。
- **IMP-003**: 需保持对元学习梯度的裁剪 (`torch.nn.utils.clip_grad_norm_`)，以应对硬门控可能带来的训练动态变化。

## 参考文献 (References)

- **REF-001**: `DFC-Theory.md`
