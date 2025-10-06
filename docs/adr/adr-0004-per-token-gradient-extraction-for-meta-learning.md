---
title: "ADR-0004: 通过逐令牌梯度提取重构元学习信号计算"
status: "Accepted"
date: "2025-10-06"
authors: "Ω Researcher, Tiny-ONN 课题组"
tags: ["architecture", "decision", "meta-learning", "autograd", "technical-debt"]
supersedes: ""
superseded_by: ""
---

# ADR-0004: 通过逐令牌梯度提取重构元学习信号计算

## 状态 (Status)

Proposed | **Accepted** | Rejected | Superseded | Deprecated

## 背景 (Context)

在训练过程中，模型在中间层（L2, L3）出现了病态的稠密激活现象。可视化分析显示，几乎所有专家原型都被元学习机制错误地判定为“Good”，导致路由选择失效。

经过深入的理论分析，根本原因被定位到元学习的“学习成本”（`C_learn`）信号的计算上。当前的实现依赖于 PyTorch `autograd` 引擎计算出的、在批次（Batch）和序列（Sequence）维度上**聚合后**的参数梯度。这种聚合从根本上摧毁了我们所需要的、细粒度的**逐令牌（per-token）**信息。`Goodness` 分数被输入了严重失真的、平均化的信号，使其在处理中层抽象表征时，完全丧失了区分专家优劣的能力，从而引发了整个元学习系统的崩溃。

## 决策 (Decision)

我们将在一次标准的反向传播中，高效、精确地捕获并计算逐令牌的参数梯度，从而为元学习提供高保真度的信号。

该方案的核心机制如下：

1. **钩子注册与数据捕获**: 在前向传播过程中，为每个 `SparseProtoLinear` 模块的输出张量 `computation_output` 注册一个 `tensor.register_hook()`。同时，将该模块的输入张量 `x` 保存到外部列表中。
2. **梯度拦截**: 当 `loss.backward()` 被调用时，注册的钩子将被触发。钩子函数会捕获流经该节点的、未被聚合的、逐令牌的中间梯度 `grad_output` (`∂L/∂Y`)。
3. **安全存储**: 钩子和前向传播中捕获的所有张量（`grad_output` 和 `x`）都将通过 `.clone().detach()` 进行处理，以防止内存泄漏和意外的计算图交互。
4. **核外计算**: 在 `loss.backward()` 完成后，我们在 `LearningDynamics` 模块中，使用捕获到的 `grad_output` 和 `x`，通过 `torch.einsum('bso,bsi->bsoi', grad_output, x)` 手动、精确地计算出逐令牌的参数梯度。

## 后果 (Consequences)

### 积极 (Positive)

- **POS-001**: 从根本上解决了元学习信号失真的问题，使 `Goodness` 分数能够基于高保真度的、逐令牌的信息进行计算。
- **POS-002**: 预期将消除中间层的病态稠密激活现象，恢复健康、有意义的动态稀疏路由。
- **POS-003**: 使工程实现与 `DFC-Theory` 的核心理论原则（特别是关于局部计算、上下文敏感的学习成本）完全对齐。

### 消极 (Negative)

- **NEG-001**: 增加了训练循环的实现复杂度，引入了手动的梯度计算以及通过钩子和外部列表进行的数据传递。
- **NEG-002**: 引入了中等程度的内存开销，因为需要为批次中的每个SPL模块存储克隆后的 `x` 和 `grad_output` 张量。
- **NEG-003**: 此解决方案高度依赖于对 `autograd` 内部机制的深入理解，可能会降低代码对不熟悉该技术的开发者的可维护性。

## 考虑的备选方案 (Alternatives Considered)

### `torch.func.vmap` 方案

- **ALT-001**: **描述 (Description)**: 使用 `vmap` 遍历序列维度，并为每个令牌调用 `torch.autograd.grad` 来获取逐令牌参数梯度。
- **ALT-002**: **拒绝理由 (Rejection Reason)**: 不可接受的性能开销。计算成本将是 `O(S)` 次完整的反向传播，使训练变得不可行。

### 自定义 `torch.autograd.Function` 方案

- **ALT-003**: **描述 (Description)**: 为线性层创建自定义 `autograd.Function`，尝试在 `backward` 方法中手动实现并提取逐令牌梯度。
- **ALT-004**: **拒绝理由 (Rejection Reason)**: PyTorch 的 `autograd.Function` API 被严格设计为只能返回与前向输入相对应的聚合梯度。没有官方支持的、安全的方法可以从 `backward` 方法中传递出额外的张量。

### 仅修正度量函数方案

- **ALT-005**: **描述 (Description)**: 保持使用聚合梯度，但将信息损失严重的 `torch.norm` 度量替换为稍微好一些的，如 `torch.mean(torch.abs(x))`。
- **ALT-006**: **拒绝理由 (Rejection Reason)**: 治标不治本。虽然度量函数可能有所改善，但它仍然作用于已经失去所有逐令牌信息的聚合梯度上，没有解决信息失真的根本原因。

## 实施注意事项 (Implementation Notes)

- **IMP-001**: 实施将主要集中在 `MoIETransformerBlock.forward`（用于注册钩子和保存张量）和 `LearningDynamics.compute_and_apply_gradients`（用于最终的 `einsum` 计算）。
- **IMP-002**: 所有被捕获的张量（`x` 和 `grad_output`）必须使用 `.clone().detach()` 进行处理。
- **IMP-003**: 方案的成功将通过“原型分布”图进行监控。成功的实施应表现为所有层（特别是中间层）恢复稀疏、有选择性的激活模式。

## 参考文献 (References)

- **REF-001**: `docs/rules/DFC-Theory.md`
- **REF-002**: `docs/adr/adr-0003-defer-bernoulli-jsd-implementation.md`
- **REF-003**: 相关理论会议的内部讨论记录。
