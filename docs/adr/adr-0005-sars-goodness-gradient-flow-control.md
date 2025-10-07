---
title: "ADR-0005: 通过主动梯度门控实现SARS Goodness效用驱动的梯度流控制"
status: "Accepted"
date: "2025-10-07"
authors: "Ω Researcher, Tiny-ONN 课题组"
tags:
  [
    "architecture",
    "decision",
    "meta-learning",
    "sars",
    "gradient-gating",
    "autograd",
  ]
supersedes: ""
superseded_by: ""
---

# ADR-0005: 通过主动梯度门控实现 SARS Goodness 效用驱动的梯度流控制

## 状态 (Status)

Proposed | **Accepted** | Rejected | Superseded | Deprecated

## 背景 (Context)

在 `0c57d45` 的重构后，我们的元学习框架暴露出两个严重的理论缺陷：

1. **被动元学习 (Passive Meta-Learning)**: 系统能够计算出专家的 `Goodness` 效用分数，但仅用其更新路由参数。它未能利用此高阶信号主动干预主任务的梯度回传，导致被判定为“坏”的专家依然会接收到梯度更新，这阻碍了功能分化并可能导致灾难性遗忘。
2. **梯度污染 (Gradient Pollution)**: 转向 `main_loss.backward()` 触发钩子来捕获逐 token 梯度，其副作用是主任务损失 `L_main` 的梯度会错误地累加到路由参数上，破坏了我们双优化器框架设计的核心——计算流与元学习流的正交性。

## 决策 (Decision)

我们决定实施一个**主动的、梯度正交的元学习框架**。该决策包含两个核心机制：

1. **梯度正交性恢复**: 为了消除梯度污染，我们采用一个精确的三步流程：
   a. 在调用 `main_loss.backward(retain_graph=True)` 以触发梯度捕获钩子后，立即手动将所有路由参数 (`p`, `g`) 的 `.grad` 属性清零。
   b. 在计算出元学习损失 `L_meta` 后，使用 `torch.autograd.grad` 来计算完全独立的、纯净的元学习梯度。
   c. 将此纯净梯度手动赋值给路由参数的 `.grad` 属性。

2. **主动梯度门控 (Active Gradient Gating)**: 我们将元学习从被动观察者转变为主动干预者。
   a. 使用元学习流计算出的 `Goodness` 分数张量。
   b. 在计算优化器 `optimizer_comp.step()` 执行之前，将 `Goodness` 分数作为一个门控掩码，直接与计算参数 (`mu_weight`, `mu_bias`) 的梯度 (`.grad`) 进行逐元素相乘。

此决策确保了元学习能够基于高保真度的信号，主动、精确地控制哪些专家有资格根据当前任务进行学习，从而直接保护已有知识并强制推动功能分化。

## 后果 (Consequences)

### 积极 (Positive)

- **POS-001**: **理论完备性**: 使元学习框架从被动转为主动，与主动推断 (Active Inference) 的理论原则更加一致。
- **POS-002**: **知识保护**: 直接阻止了对不相关或“坏”专家的梯度更新，从机制上解决了灾难性遗忘问题。
- **POS-003**: **加速功能分化**: 通过奖励“好”专家并惩罚“坏”专家，强力驱动模型形成功能专一、可复用的专家库。
- **POS-004**: **恢复梯度正交性**: 解决了 `0c57d45` 引入的梯度污染问题，使双优化器框架的理论基础重新变得坚实。

### 消极 (Negative)

- **NEG-001**: **增加实现复杂度**: 引入了更复杂的梯度流手动控制逻辑，需要对 PyTorch `autograd` 机制有更深入的理解。
- **NEG-002**: **潜在的数值不稳定**: 梯度与 `Goodness` 分数的直接相乘，如果 `Goodness` 的尺度控制不当，可能会引入数值问题（尽管 `mas_normalize` 缓解了部分风险）。
- **NEG-003**: **调试难度增加**: 梯度不再仅仅由损失函数决定，还受到元学习系统的动态调控，这使得问题的诊断和调试变得更加困难。

## 考虑的备选方案 (Alternatives Considered)

### 方案 A: 被动元学习 (维持现状)

- **ALT-001**: **描述 (Description)**: 维持被动的元学习系统，仅用元学习更新路由参数，不干预主梯度流。
- **ALT-002**: **拒绝理由 (Rejection Reason)**: 理论上不完备。未能解决灾难性遗忘和功能分化缓慢的核心问题，且存在梯度污染的严重 bug。

### 方案 B: 使用正则化进行软性门控

- **ALT-003**: **描述 (Description)**: 不直接门控梯度，而是在主损失函数中添加一项正则化项，该项惩罚 `Goodness` 分数低的专家的参数更新量（例如，`L_reg = Σ || w_i * (1 - Goodness_i) ||²`）。
- **ALT-004**: **拒绝理由 (Rejection Reason)**: 这是一个间接的、软性的约束，而非精确的门控。其效果严重依赖于正则化超参数的调整，难以保证对知识的完全保护。直接门控梯度是更根本、更精确的解决方案。

## 实施注意事项 (Implementation Notes)

- **IMP-001**: 实施集中在 `exp/arc/train.py` 的 `LearningDynamics` 类中，修改 `compute_and_apply_gradients` 方法。
- **IMP-002**: 必须严格保证梯度操作的顺序：`main_loss.backward()` → `router_grad.zero_()` → `autograd.grad(L_meta)` → `router_grad = meta_grad` → `gated_grad = main_grad * goodness` → `optimizer.step()`。
- **IMP-003**: 需要通过 `observer` 密切监控 `mu_weight` 的梯度范数，以验证梯度门控是否按预期工作。

## 参考文献 (References)

- **REF-001**: `docs/adr/adr-0004-per-token-gradient-extraction-for-meta-learning.md`
- **REF-002**: `docs/rules/DFC-Theory.md`
