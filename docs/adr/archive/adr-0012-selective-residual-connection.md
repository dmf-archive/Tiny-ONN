---
title: "ADR-0012: 选择性残差连接与最小作用量原则"
status: "Proposed"
date: "2025-10-10"
authors: "林睿, Ω Researcher"
tags: ["architecture", "decision", "residual-connection", "spl", "principle-of-least-action"]
supersedes: "ADR-0011"
superseded_by: ""
---

# ADR-0012: 选择性残差连接与最小作用量原则

## 状态 (Status)

**Proposed** | Accepted | Rejected | Superseded | Deprecated

## 背景 (Context)

在对`SPL`门控机制的持续探索中（见 ADR-0011），我们最初的“直通门”方案被证明是一个理论上的误区。它虽然解决了`ReLU`的信息湮灭问题，但其直通的对象是已被处理的`computation_output`，而非原始输入`x`。这违背了残差连接的核心思想，未能有效保留未经注意的原始信息。更重要的是，我们认识到，将`SPL`视为输入-原型交叉注意力的本质，要求我们对门控机制进行更根本的重构，使其能够精确地对**特征维度 (feature dimensions)** 进行操作，而不仅仅是调制整个输出向量。

## 决策 (Decision)

我们决定废弃所有先前的门控方案，并实施一种新的、理论上更完备的**选择性残差连接 (Selective Residual Connection)** 机制。

该机制将`SPL`模块的输出形式化为两个正交部分的线性组合：

1. **处理部分 (Processed Part)**: 由被激活的神经元（`routing_logits > 0`）计算出的、经过变换的`computation_output`。此部分只在被“注意”的特征维度上产生贡献。
    `processed_part = computation_output * F.relu(routing_logits)`

2. **直通部分 (Pass-Through Part)**: 原始输入`x`中，对应于**未**被激活的神经元的特征维度。这些维度未经任何改变，被直接传递到输出。
    `passthrough_part = x * (1.0 - (routing_logits > 0).float())`

最终输出是这两部分的简单求和：
`output = processed_part + passthrough_part`

此决策将经典的残差连接 (`y = x + F(x)`) 从一种被动的、硬编码的架构特性，转变为一种主动的、由模型根据输入动态决定的、在特征维度级别上操作的计算策略。

## 后果 (Consequences)

### 积极 (Positive)

- **POS-001**: **实现最小作用量原则 (Principle of Least Action)**: 该机制在功能上实现了最小作用量原则。模型被强力激励去学习只对任务相关的输入特征维度进行计算和更新，而让不相关的维度以零成本（恒等变换）通过。这是一种内生的、动态的计算稀疏性。
- **POS-002**: **保留原始信息 (Preservation of Original Information)**: 确保了在多层处理后，原始输入`x`中未经注意的细节信息不会被非线性变换所“冲刷”或丢失，这对于需要精确细节的 ARC 任务至关重要。
- **POS-003**: **理论一致性 (Theoretical Coherence)**: 将残差连接、交叉注意力门控和动态稀疏性统一在一个简洁、自洽的数学框架内，极大地增强了`SPL`模块的理论优雅性。
- **POS-004**: **协同设计验证 (Synergistic Design Validation)**: 该决策的成功，依赖于`train.py`中对`Goodness`计算的同步修正——即`Goodness`分数只在被激活的神经元上计算。这验证了我们“协同、跨文件、系统性设计”方法的有效性。

### 消极 (Negative)

- **NEG-001**: **实现复杂性增加**: 相比简单的乘法门控，向量化的正交融合逻辑在概念上和实现上都更复杂。
- **NEG-002**: **对元学习的依赖**: 该机制的有效性高度依赖于`SARS`元学习能够成功地学习到有意义的路由策略。如果路由失败，模型可能会退化为简单的残差网络或完全不学习。

## 考虑的备选方案 (Alternatives Considered)

### “直通门”方案 (Pass-Through Gate)

- **ALT-001**: **描述 (Description)**: 在 ADR-0011 中提出的方案，对未激活的神经元直通其`computation_output`。
- **ALT-002**: **拒绝理由 (Rejection Reason)**: 这是一个理论错误。它直通的是已被处理的信息，而非原始输入`x`，未能实现真正的残差连接，无法有效保留原始上下文。

### 简单的插值/相加

- **ALT-003**: **描述 (Description)**: 将处理后的输出与原始输入`x`进行简单的相加或插值，例如 `output = x + computation_output * F.relu(routing_logits)`。
- **ALT-004**: **拒绝理由 (Rejection Reason)**: 这种方法是非正交的。激活神经元计算出的结果会与该维度上原始的输入`x`信息相混合，导致信息干扰。我们的正交方案确保了在任何一个特征维度上，信息源要么是处理过的，要么是原始的，二者择一，从而保证了信号的纯净性。

## 实施注意事项 (Implementation Notes)

- **IMP-001**: 核心实现位于`exp/arc/model.py`的`MoIETransformerBlock.forward`方法中，已通过原子重构完成。
- **IMP-002**: **关键协同**: `exp/arc/train.py`中的`_calculate_goodness_jit`函数必须同步修改，确保`goodness_logits`在计算后被激活掩码 (`grad_mask`) 过滤，以将直通神经元视为“关闭 (Shutdown)”。此协同修改已完成。
- **IMP-003**: 训练的成功启动和`GBS%`指标恢复动态，已初步验证了该系统性设计的正确性。

## 参考文献 (References)

- **REF-001**: `docs/adr/adr-0011-pass-through-gate-neurobiology.md` (被此 ADR 取代)
- **REF-002**: `exp/arc/model.py`, `exp/arc/train.py`
