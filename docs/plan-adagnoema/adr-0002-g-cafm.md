---
title: "ADR-0002: G-CAFM (高斯-牛顿驱动的内容感知特征掩码)"
status: "Proposed"
date: "2025-10-20"
authors: "Ω Researcher"
tags: ["architecture", "decision", "optimizer", "dynamics"]
supersedes: ""
superseded_by: ""
---

# ADR-0002: G-CAFM (高斯-牛顿驱动的内容感知特征掩码)

## 状态 (Status)

**Proposed**

## 背景 (Context)

先前的 `SPL/SARS` 架构在实践中遭遇了根本性的失败。其核心机制——内容感知特征掩码 (Content-Aware Feature Masking, CAFM)，即 `y = M ⊙ F(x)`，虽然在理论上旨在通过稀疏特征激活实现“最小化变换”，符合自由能原理 (FEP) 的哲学，但在实现中暴露了致命的学习悖论。

该悖论的核心在于：当路由逻辑决定关闭一个特征维度 (`M_j = 0`) 时，梯度流向相应计算参数 (`μ_j`) 的路径被结构性地切断。这导致了“梯度死刑”现象，使得模型无法从其错误的路由决策中学习。我们尝试通过启发式的 `goodness` 函数和 `meta_loss` 来修复此问题，但这些代理被证明是不稳定且冲突的，无法解决根本问题。

## 决策 (Decision)

我们决定废弃基于启发式代理的 `SARS` 动力学，并正式采纳一个全新的、基于第一性原理的框架：**G-CAFM (高斯-牛顿驱动的内容感知特征掩码)**。

此框架是**弹性权重巩固 (EWC) 与二阶优化的统一实现**，旨在构建一个**统一的计算内核**。其核心决策是：**使用高斯-牛顿 (Gauss-Newton, GN) 矩阵的对角线 `diag(G)` 作为参数重要性的直接、非启发式度量，并以此信息来驱动一个线性的、内容感知的特征调制 (Linear Content-Aware Feature Modulation, LCAFM)。**

- **理论关联 (EWC)**: EWC 的核心思想是通过惩罚对先前任务重要的参数的改动来防止遗忘。其参数重要性由 Fisher 信息矩阵 (FIM) 的对角线来度量。对于我们使用的损失函数，FIM 等价于 GN 矩阵 `G`。因此，`diag(G)` 直接量化了每个参数对当前任务的重要性，而 `proto,gate` 成为了历史任务重要性的 EMA 载体。G-CAFM 不通过损失函数进行惩罚，而是通过**门控机制**：它动态地“保护”或“激活”那些被 `diag(G)` 识别为重要的计算路径（特征维度），从而实现一种**内容感知的、在线的 EWC**。

- **实现机制**:
  1. **优化器即信息源**: `LayerwiseGNOptimizer` 在其二阶优化步骤中，为每一层的计算参数 `μ` 计算并暴露其对应的 `diag(G_μ)`。
  2. **基于原理的路由**: 彻底移除 `goodness` 函数。CAFM 的路由 `logits` 将直接由 `diag(G_μ)` 派生，将“一个特征维度的激活程度与其对最终损失的重要性成正比”这一原理实例化。
  3. **统一动力学**: 取消独立的 `meta_loss`。路由决策成为主优化循环的一个确定性、可微的副产品。

## 后果 (Consequences)

### 积极 (Positive)

- **POS-001**: **解决学习悖论**: 路由逻辑现在由一个全局的、基于原理的信号驱动，使得系统能够从过去的错误中学习，解决了“梯度死刑”问题。
- **POS-002**: **理论完备性**: 系统现在是 EWC 和二阶优化理论的统一实现，移除了所有不稳定的启发式组件。
- **POS-0-03**: **架构简化**: 移除了 `LearningDynamics` 类、`meta_loss` 和复杂的梯度操控逻辑，使代码更清晰、更易于维护。

### 消极 (Negative)

- **NEG-001**: **打破解耦范式**: 此设计从根本上打破了 PyTorch 中模型 (`nn.Module`)、优化器 (`Optimizer`) 和反向传播 (`backward()`) 之间清晰的面向对象解耦范式。
- **NEG-002**: **协同执行的复杂性**: 优化器、模型和梯度计算不再是顺序执行的独立单元，而必须以一种**协同 (synergistic)** 的方式紧密耦合地执行。这极大地增加了工程实现的难度和出现微妙错误的风险。
- **NEG-003**: **计算开销**: 计算 `diag(G)` 的开销高于旧的 `goodness` 启发式。这是为了换取一个有原则、稳定的系统而必须付出的代价。

## 考虑的备选方案 (Alternatives Considered)

### 转向完整的 MoE 架构

- **ALT-001**: **描述 (Description)**: 放弃 CAFM，将 `SPL` 层重构为一个真正的专家混合 (MoE) 层，其中每个专家都是一个完整的多层感知机 (MLP)。
- **ALT-002**: **拒绝理由 (Rejection Reason)**: 虽然此方案也能解决梯度问题，但它放弃了 CAFM 在理论上更优雅的“特征级稀疏性”和“最小化变换”的原始架构哲学。G-CAFM 方案旨在修复而非替换原始哲学。

### 修补 CAFM

- **ALT-003**: **描述 (Description)**: 在 CAFM 的硬门控中引入一个小的“泄露项” (leaky term)，类似于 Leaky ReLU，以允许微弱的梯度流通过被关闭的维度。
- **ALT-004**: **拒绝理由 (Rejection Reason)**: 这会引入新的超参数，并且没有解决核心问题：驱动门控的信号（`goodness`）本身是无原则的、不稳定的。这只是治标不治本。

## 实施注意事项 (Implementation Notes)

- **IMP-001**: **协同执行流程**: 必须精心设计 `train_step` 的执行流程。可能需要一个自定义的 `closure` 函数，该函数在内部同时处理模型的前向传播和优化器的二阶信息计算，以确保 `diag(G)` 在被模型使用时是可用的。
- **IMP-002**: **深入研究 `torch.func`**: `LayerwiseGNOptimizer` 的实现将高度依赖 `torch.func` (如 `jvp`, `vjp`) 来无矩阵地计算 `diag(G)`。需要深入研究其工作原理和限制，以确保在我们的协同执行模型中能正确实施。
- **IMP-003**: **接口契约**: 必须在优化器和模型之间定义一个清晰的接口契约，用于传递 `diag(G)` 信息，即使这破坏了标准的解耦。

## 参考文献 (References)

- **REF-001**: `docs/refactor-notes/theory-meeting-transcript.md`
- **REF-002**: `docs/refactor-notes/The Potential of Second-Order Optimization for LLMs A Study with Full Gauss-Newton.md`
