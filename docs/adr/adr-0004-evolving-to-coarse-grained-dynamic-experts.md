---
title: "ADR-0004: Evolving from Fine-Grained CAFM to Coarse-Grained Dynamic Experts"
status: "Proposed"
date: "2025-10-21"
authors: "Ω Researcher"
tags: ["architecture", "decision", "sparse-attention", "moe", "dynsmha"]
supersedes: ""
superseded_by: ""
---

# ADR-0004: Evolving from Fine-Grained CAFM to Coarse-Grained Dynamic Experts

## 状态 (Status)

**Proposed**

## 背景 (Context)

当前的 `SparseProtoLinear` (SPL) 架构，它在单个特征维度上实现内容感知特征掩码 (Content-Aware Feature Masking, CAFM)，在实践中暴露了根本性的理论与工程问题。尽管我们尝试通过改进其学习动力学 (`SARS`) 来修复，但模型仍然持续遭遇“模式坍塌”和“梯度死亡”的困境。

经过深入分析，我们识别出其核心病因：

1. **无效的计算单元**: 将单个特征维度作为稀疏化的基本单位，这个“子空间”过小，无法捕捉到解决 ARC 等复杂任务所需的高层语义概念。
2. **结构性的梯度断流**: 基于 `ReLU` 的硬性门控机制在路由决策中关闭了某些计算路径，从而结构性地切断了梯度流。这使得模型无法从其错误的路由决策中学习，我们称之为“梯度死刑”。
3. **失败的根本假设**: 实验（如用恒等变换代替掩码）表明，即使缓解了梯度问题，模式坍塌依旧存在。这强烈暗示，SPL 所基于的“细粒度特征组合”这一核心假设本身可能存在缺陷。

## 决策 (Decision)

我们决定对架构的计算单元进行一次**升维演进**，从细粒度的特征掩码转向粗粒度的专家路由。

1. **提升计算单元**: `SPL` 框架中的核心计算单元 `μ`，将不再是一个特征向量，而是被提升为一个完整的、功能更丰富的**专家模块**（例如，一个 `MLP` 或一个完整的 `Linear` 层）。

2. **回归 DynSMHA + DynMoE**:

   - 注意力模块 (`DynamicInfiniteHeadAttention`) 将演进为 **`DynamicSparseMultiHeadAttention` (DynSMHA)**。它将保留端到端可微的基于原型的交叉注意力路由机制，但路由目标从特征维度变为一组离散的、可学习的“专家头”（Expert Heads）。其输出投影 `o_proj` 将使用一个标准的 `Linear` 层。每一个输入子空间将通过交叉注意力和所有潜在注意力头进行匹配，允许在同一层多次复用单个注意力头。
   - 我们将重新在 Transformer 块中引入一个独立的 FFN 层，并将其实现为 **`DynamicMixtureOfExperts` (DynMoE)**。该层同样使用我们升级后的、作用于专家模块的 `SPL/SARS` 动力学。

3. **保留学习框架**: `SARS` 的核心学习动力学（基于 `goodness_logits` 和 `meta_loss`）将被保留，但其作用和评估的对象从单个特征维度转变为整个专家模块。

## 后果 (Consequences)

### 积极 (Positive)

- **POS-001 (解决模式坍塌)**: 通过使用表达能力更强的专家模块（`MLP`），我们为学习提供了更丰富的语义基底，从而降低了所有专家都坍塌到一个单一、通用功能的风险。
- **POS-002 (改善梯度流)**: 梯度现在将流向整个专家模块，而不是被硬性门控切断。这为路由和计算参数提供了更稳定、更有意义的学习信号。
- **POS-003 (架构对齐)**: 新架构（`DynSMHA` + `DynMoE` FFN）在宏观上与经过验证的标准 Transformer 设计更加一致，同时保留了我们独特的动态路由机制，降低了整体架构风险。

### 消极 (Negative)

- **NEG-001 (参数量增加)**: 用完整的 `MLP` 专家替换特征向量，将不可避免地增加模型的总参数量。
- **NEG-002 (实现复杂性)**: `DynSMHA` 的前向传播实现将更具挑战性。它需要为每个头和每个 token 高效地动态选择、应用专家，并最终将结果组装成 `scaled_dot_product_attention` 所需的格式。
- **NEG-003 (假设的演进)**: 此决策标志着我们从“细粒度特征组合”的假设，演进到了“粗粒度专家选择”的假设。

## 考虑的备选方案 (Alternatives Considered)

### 方案 1: 为细粒度 CAFM 引入二阶信号 (G-CAFM)

- **ALT-001 (描述)**: 保留当前的细粒度 SPL 架构，但用一个基于高斯-牛顿矩阵对角线 (`diag(G)`) 的、有原则的二阶信号来取代启发式的 `goodness_logits`。
- **ALT-002 (拒绝理由)**: 此方案未能解决核心批评，即“特征子空间过小”以及“细粒度组合”这一假设本身可能就是有缺陷的。无论学习信号多么完美，在一个有缺陷的基底上进行优化，可能依然无法避免模式坍塌。未来的 `DynSMHA` 依然可以考虑引入二阶优化体系。

### 方案 2: 采用标准的 MoE 实现

- **ALT-003 (描述)**: 完全抛弃 `SPL/SARS` 框架，用一个标准的、使用简单线性层 + `softmax` 进行门控的 MoE 实现来替换相关层。
- **ALT-004 (拒绝理由)**: 这将完全放弃本项目在动态路由机制上的特色——端到端可微的，基于交叉注意力的原型路由。我们选择的路径是一次保留核心思想的演进，而非全盘替换。

## 实施注意事项 (Implementation Notes)

- **IMP-001 (高效批处理)**: `DynSMHA` 中的动态专家选择应通过高效的、可向量化的张量操作（例如，使用“稠密掩码”和 `torch.einsum` 或 `torch.bmm`）来实现，以避免使用低效的 Python 循环。
- **IMP-002 (SDPA 兼容性)**: 动态专家计算的输出必须被仔细地塑形 (reshape) 和组装 (assemble) 成 `(Batch, Heads, SeqLen, HeadDim)` 的格式，以确保与 `torch.nn.functional.scaled_dot_product_attention` 的无缝兼容。
- **IMP-003 (分阶段重构)**: 建议分阶段进行重构：1. 定义并实现新的 `ExpertMLP` 基础模块。2. 实现 `DynamicSparseMultiHeadAttention` 模块。3. 将新模块集成到 `MoIETransformerBlock` 中，并重新引入 `DynMoE` FFN 层。

## 参考文献 (References)

- **REF-001**: `.roo/rules/1-glossary.md` (关于 DynMoE, DynSIHA 的原始定义)
- **REF-002**: 相关的开发会议讨论记录。
