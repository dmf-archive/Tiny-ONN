---
title: "ADR-0001: Simplify Transformer Architecture by Removing MoIE FFN"
status: "Proposed"
date: "2025-10-05"
authors: "Ω Researcher, Tiny-ONN 课题组"
tags: ["architecture", "decision", "transformer", "sparsity", "spl", "moie"]
supersedes: ""
superseded_by: ""
---

# ADR-0001: Simplify Transformer Architecture by Removing MoIE FFN

## 状态 (Status)

Proposed | **Accepted** | Rejected | Superseded | Deprecated

## 背景 (Context)

`Tiny-ONN` 模型的核心 `MoIETransformerBlock` 在 GPU 平台上表现出严重的性能问题。尽管模型在运行时达到了高水平的激活稀疏度（接近 1%），但其端到端延迟与完全稠密的计算没有显著差异。这引发了一项从第一性原理层面分析此问题根源的研究，其最终结论是必须对现有架构进行根本性简化。本 ADR 记录了该分析过程与最终决策。

## 形式化分析 (Formal Analysis)

我们的分析始于性能瓶颈，最终揭示了模型核心构建块 `SparseProtoLinear (SPL)` 的真实计算特性，这是做出最终决策的关键理论基础。

### 1. SPL 计算开销解构

`SPL` 模块的性能瓶颈并非源于我们试图稀疏化的主计算路径，而是源于为决定如何稀疏化而引入的、开销巨大的路由路径。

对于单个输入向量 $x \in \mathbb{R}^{D_{in}}$，`SPL` 模块（权重 $W_{\mu}, W_{p}, W_{g} \in \mathbb{R}^{D_{out} \times D_{in}}$）执行以下操作：

1. **主计算路径 (Computation Path)**:
   $Y_{comp} = x W_{\mu}^T + b_{\mu}$

   - **类型**: 稠密矩阵-向量乘法。
   - **FLOPS**: $O(D_{in} \cdot D_{out})$

2. **路由计算路径 (Routing Path)**:
   a. **原型匹配 (Prototype Matching)**:
   $V_{match} = (x W_{p}^T) / \sqrt{D_{in}}$
   **\*类型**: 缩放点积（稠密矩阵-向量乘法）。 \* **FLOPS**: $O(D_{in} \cdot D_{out})$
   b. **门控成本 (Gating Cost)**:
   $L_{gate} = x W_{g}^T$
   **\*类型**: 稠密矩阵-向量乘法。 \* **FLOPS**: $O(D_{in} \cdot D_{out})$

**分析**: 总计算开销 $Cost_{total} \approx 3 \cdot O(D_{in} \cdot D_{out})$。而路由开销占比为 $2/3$。这个结论是颠覆性的：我们模型的性能瓶颈是一个完全稠密的、占据绝对主导地位的路由计算，任何针对主计算路径的稀疏优化都注定无效。

### 2. SPL 作为两层 MLP 的计算等价性

一个普遍的误解是将 `SPL` 视为一个线性模块。然而，其内部的数据依赖路由机制使其成为一个强大的非线性单元。

`SPL` 的最终输出由主计算结果与一个动态生成的掩码 $M_{routing}(x)$ 逐元素相乘得到：
$Y_{out} = Y_{comp} \odot M_{routing}(x)$

路由函数 $M_{routing}(x)$ 的精确形式为：
$M_{routing}(x) = \text{mas\_normalize}((x W_{p}^T / \sqrt{D_{in}}) - \text{mas\_normalize}(x W_{g}^T))$
其中 `mas_normalize` 的行为类似 `ReLU` 激活函数。

**等价性论证**:
一个 `ReLU(SPL(x))` 单元的计算图如下：

1. **第一层（并行线性层）**: 输入 $x$ 被并行馈送到三个不同的线性层 ($W_{\mu}, W_{p}, W_{g}$)。
2. **第一层非线性 (路由激活)**: 路由分支的结果通过 `mas_normalize` (ReLU-like) 和逐元素减法组合，生成路由权重 $M_{routing}(x)$。
3. **Hadamard 积 (门控非线性)**: 主计算路径的结果与路由权重相乘 ($Y_{comp} \odot M_{routing}(x)$)。这是一个强非线性操作，因为权重 $M_{routing}$ 是输入 $x$ 的函数。
4. **第二层非线性 (输出激活)**: 步骤 3 的结果被送入外部的 `ReLU` 激活函数。

**结论**: `ReLU(SPL(x))` 单元包含两组核心的非线性激活（内部路由激活和外部输出激活），其计算深度和表达能力可合理地等价于一个**两层 MLP**。

### 3. MoIE FFN 与 DynSIHA 的真实计算深度

基于上述结论，我们可以推断出模型关键组件的真实计算深度：

- **`MoIE` FFN**: 由 `SPL_2(ReLU(SPL_1(x)))` 构成，其计算深度和非线性复杂性等效于一个**四层 MLP**。
- **`DynSIHA`**: 由四个并行的 `SPL` 模块（用于生成 Q, K, V, O）和一个标准的自注意力模块构成。它不仅仅是信息混合，其 Q, K, V 的生成过程本身就是强大的非线性特征变换。其复杂性至少等价于一个**八层 MLP**。

### 4. DynSIHA 的双重注意力本质与贝叶斯诠释

`DynSIHA` 的计算过程可以被重新阐释为一个“双重注意力”机制：

1. **第一重 (垂直) 注意力：输入-原型 注意力**: 在 `SPL` 内部，输入 token $x$ (Query) 与所有神经元原型 $W_p$ (Key) 进行交互，以决定对应的计算核 $W_{\mu}$ (Value) 的激活权重。这是一个隐式的、决定“关注”哪些内部概念的注意力过程。
2. **第二重 (水平) 注意力：上下文 Token 间注意力**: 经过第一重注意力“提炼”后的稀疏 $q, k, v$ 向量，再进入标准的自注意力模块，在序列的不同位置间进行信息整合。

这个过程依然等价于一个**贝叶斯更新**：

- **先验 (Prior)**: 原型库 ($W_p, W_{\mu}$) 是模型关于世界概念和操作的先验知识。
- **似然 (Likelihood)**: 第一重注意力计算了输入数据在给定这些先验下的可能性，生成了稀疏的 $q, k, v$ 作为似然的体现。
- **后验 (Posterior)**: 第二重注意力整合上下文信息，最终输出的 `hidden_state` 是对 token 表示的后验更新。

## 决策 (Decision)

基于以上形式化分析，我们提出“`MoIE` FFN 过剩假说”：`DynSIHA` 模块强大的非线性变换能力与 `MoIE` FFN 的功能存在严重重叠。

因此，我们决定，下一阶段的核心实验方向是**完全移除 `MoIETransformerBlock` 中的 `MoIE` (FFN) 模块**，并评估一个仅由 `DynSIHA` 驱动的简化 Transformer 架构。

**决策理由**:

1. **消除计算冗余**: `DynSIHA` 模块本身已具备强大的非线性特征变换能力。移除功能重叠的 `MoIE` FFN 是遵循奥卡姆剃刀原则 (`CON-201`)，消除潜在计算过剩的最直接方法。
2. **大幅降低计算开销**: `MoIE` FFN 模块占据了 Transformer 块一半以上的计算量。移除它可以直接、显著地降低模型的训练和推理延迟，并减少参数量。
3. **统一计算范式**: 在简化架构中，`DynSIHA` 将协同地、统一地执行信息在序列位置间的路由和信息在特征维度上的变换。这可能迫使模型学习到更高效、更整合的内部表示。

## 后果 (Consequences)

### 积极 (Positive)

- **POS-001**: **性能提升**: 预计模型的训练和推理速度将有数量级的提升。
- **POS-002**: **模型简化**: 降低了代码库的复杂性，提高了可维护性。
- **POS-003**: **参数效率**: 模型总参数量将显著减少，降低了内存占用。
- **POS-004**: **理论统一**: 推动模型架构向一个更简洁、计算范式更统一的方向演进。

### 消极 (Negative)

- **NEG-001**: **表达能力风险**: 移除一个等效四层 MLP 的计算深度，可能会降低模型的总表达能力，存在该模块对解决某些复杂 ARC 任务是必要的风险。
- **NEG-002**: **收敛特性未知**: 简化后模型的收敛动态和最终性能需要通过实验重新验证。
- **NEG-003**: **理论假设依赖**: 该决策的成功，高度依赖于我们关于“`DynSIHA` 的变换能力足够强大”这一核心假说。

## 考虑的备选方案 (Alternatives Considered)

### 方案 A: 硬件/计算范式优化

- **ALT-001**: **描述 (Description)**: 探索使用标准稀疏格式（BSR）、`Top-K` 门控、手写 CUDA 核函数、或切换至 CPU 进行稀疏推理。
- **ALT-002**: **拒绝理由 (Rejection Reason)**: 我们的理论分析证明，这些方案都不可行。标准稀疏格式不适用动态稀疏；`Top-K` 违背理论原则；而由于模型计算范式与 GPU 硬件的根本不匹配，任何软件层面的优化都无法解决不规则内存访问带来的高昂延迟。

### 方案 B: 维持现状，仅修正理论

- **ALT-003**: **描述 (Description)**: 保持现有架构不变，仅修正会议中发现的 SARS 梯度信源不纯粹 的理论缺陷。
- **ALT-004**: **拒绝理由 (Rejection Reason)**: 虽然修正理论是必要的，但它不直接解决路由计算占主导且完全稠密的根本问题。在性能瓶颈明确的情况下，优先进行架构层面的简化是更合理的选择。

## 实施注意事项 (Implementation Notes)

- **IMP-001**: **代码重构**: 需要修改 `MoIETransformerBlock`，移除对 `DynamicInfiniteExpert` 的调用以及所有 FFN 相关的计算路径。
- **IMP-002**: **实验验证**: 必须进行严格的 A/B 测试，比较简化前后的模型在关键 ARC 任务基准上的收敛速度和最终性能。
- **IMP-003**: **监控指标**: 实验期间应密切监控模型的激活稀疏度、`PI-Score` 和各项损失，以评估简化对模型自组织能力的影响。

## 参考文献 (References)

- **REF-001**: `exp/arc/model.py` (当前模型实现)
- **REF-002**: `.roo/rules/DFC-Theory.md` (动态函数合成理论基础)
