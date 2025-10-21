---
title: "ADR-0005: Architectural Evolution to DynTRM (Dynamic Tiny Recursive Model)"
status: "Proposed"
date: "2025-10-21"
authors: "Ω Researcher"
tags:
  [
    "architecture",
    "decision",
    "universal-transformer",
    "recursive-model",
    "adaptive-depth",
  ]
supersedes: "ADR-0004"
---

# ADR-0005: Architectural Evolution to DynTRM (Dynamic Tiny Recursive Model)

## 状态 (Status)

**Proposed**

## 背景 (Context)

在 `ADR-0004` 中，我们决定从细粒度的特征掩码演进到粗粒度的专家路由，以解决“模式坍塌”和“梯度死亡”的问题。在进一步的理论探索中，我们意识到这个演进方向可以被深化和推广为一个更强大、更优雅的计算范式。

对 `SamsungSAILMontreal/TinyRecursiveModels` (TRM) 的研究证实，通过权重共享和递归计算，可以在小参数量的模型上实现强大的推理能力。这与我们设想的“全局专家库”和“层即迭代”的思想不谋而合。同时，我们自己的理论文档 `docs/theory/adaptive_depth_inference.md` 提出了通过“提前退出”机制实现自适应计算深度的构想。

这些理论共同指向一个统一的、革命性的新架构。

## 决策 (Decision)

我们决定将模型架构从一个固定的、堆叠式的 Transformer，正式演进为一个**动态微型递归模型 (Dynamic Tiny Recursive Model, DynTRM)**。

1. **采纳通用递归结构 (Universal Recursive Structure)**:

    - 模型的核心将不再是一个由 `N` 个不同 `MoIETransformerBlock` 组成的 `ModuleList`，而是一个**单一的、可重入的 `DynTRMBlock` 实例**。
    - 在前向传播期间，输入将在该 `DynTRMBlock` 中被**迭代处理 `N` 次**（`N` 是一个可配置的最大深度）。
    - 这种设计在功能上等同于一个**通用 Transformer (Universal Transformer)**。

2. **实现全局动态专家库 (Global Dynamic Expert Library)**:

    - 所有的计算专家（用于注意力头和 FFN 的 `MLP`）将被定义在一个**全局共享的专家库**中，由整个模型共享。
    - 在每一次递归迭代中，`DynTRMBlock` 内部的 `DynSMHA` 和 `DynMoE` 模块将通过我们独特的原型路由机制，从这个全局库中**动态地“拉取”**所需的计算专家。这实现了**动态的、上下文相关的权重共享**。

3. **集成自适应计算时间 (Integrate Adaptive Computation Time, ACT)**:

    - 为了实现 `adaptive_depth_inference.md` 的构想，我们将为 `DynTRMBlock` 配备一个可学习的**“提前退出”机制**。
    - 该模块在每次迭代后会输出一个“停止”信号（例如，通过一个轻量级的“停止预测头”）。
    - 当满足停止条件（例如，停止信号超过阈值）或达到最大迭代次数时，递归将终止。这将使模型的计算深度能够自适应于任务的复杂度。

4. **训练范式**:
    - 在训练期间，我们将采用**固定步数展开**的策略。计算图最高可展开为最大深度 `N`，梯度将通过所有展开的步骤进行反向传播。

## 后果 (Consequences)

### 积极 (Positive)

- **POS-001 (极致的参数效率)**: 通过在不同迭代步骤和不同功能模块之间复用同一个全局专家库，模型的参数效率将得到极大提升。
- **POS-002 (自适应计算)**: ACT 机制的引入将使模型的推理成本与任务难度直接相关，在简单任务上实现低延迟，在复杂任务上进行深度思考。
- **POS-003 (涌现的程序合成)**: 模型不再是静态的层级结构，而是在功能上更接近一个真正的“计算引擎”。它在每个步骤中为自己动态“编译”计算图，为实现更高级的程序合成能力奠定了基础。
- **POS-004 (理论统一)**: 该架构统一了 `DynMoE`、`Universal Transformer` 和 `Adaptive Computation Time` 的核心思想，形成了一个理论上自洽且强大的新范式。

### 消极 (Negative)

- **NEG-001 (训练开销)**: 尽管比无限展开更优，但将计算图展开 `N` 步进行训练，依然会比标准 Transformer 带来更高的内存消耗和更长的计算时间。
- **NEG-002 (实现复杂度剧增)**: `DynTRM` 的实现，特别是涉及 `gather`/`scatter` 的动态专家调用和 ACT 机制的训练，其工程复杂度远高于之前的架构。
- **NEG-003 (梯度流挑战)**: 在长达 `N` 次的递归迭代中，如何确保 `SARS` 学习动力学所需的梯度信号能够有效、稳定地传播，将是一个核心的挑战。

## 考虑的备选方案 (Alternatives Considered)

### 方案 1: 静态的、分层的 DynMoE (ADR-0004)

- **ALT-001 (描述)**: 按照 `ADR-0004` 的设计，实现一个标准的、具有 `N` 个独立层的堆叠式 Transformer，其中每一层都使用粗粒度的 `DynSMHA` 和 `DynMoE`。
- **ALT-002 (拒绝理由)**: 虽然此方案解决了“模式坍塌”问题，但它未能抓住向“递归计算”和“自适应深度”演进的巨大理论机遇。它仅仅是一个更好的 Transformer，而 `DynTRM` 是一个全新的物种。

## 实施注意事项 (Implementation Notes)

- **IMP-001 (分阶段实施)**: 建议分阶段实施：1. 构建 `DynTRMBlock` 的基本递归结构。2. 实现基于 `gather`/`scatter` 的动态专家调用。3. 实现并集成 ACT 提前退出机制。
- **IMP-002 (梯度稳定性)**: 在长序列的递归反向传播中，必须密切关注梯度的稳定性。梯度裁剪 (Gradient Clipping) 和可能的梯度检查点 (Gradient Checkpointing) 技术可能需要被引入。
- **IMP-003 (ACT 损失设计)**: 需要为 ACT 机制设计一个合适的辅助损失函数，以有效训练“停止预测头”。

## 参考文献 (References)

- **REF-001**: `ADR-0004: Evolving from Fine-Grained CAFM to Coarse-Grained Dynamic Experts`
- **REF-002**: `docs/theory/adaptive_depth_inference.md`
- **REF-003**: `DeepWiki` query on `SamsungSAILMontreal/TinyRecursiveModels`
