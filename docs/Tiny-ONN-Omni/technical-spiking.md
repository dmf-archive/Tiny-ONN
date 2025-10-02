---
title: "技术预研：单塔 Omni-Modal 架构可行性验证"
category: "架构与设计"
status: "🔴 未开始"
priority: "高"
timebox: "2 周"
created: 2025-10-02
updated: 2025-10-02
owner: "Core-AI-Agent"
tags: ["技术预研", "架构", "多模态", "MoIE", "DynSIHA"]
---

# 技术预研：单塔 Omni-Modal 架构可行性验证

## 摘要

**探索目标 (Spike Objective):** 验证一个统一的、基于 MoIE/DynSIHA 的“单塔”架构，在不依赖独立模态编码器的情况下，处理并学习多种模态（从 ARC 的抽象视觉推理开始，扩展到文本、图像）作为统一序列的可行性与有效性。

**重要性 (Why This Matters):** 这是对主流“多塔”架构（如 LLaVA）的根本性偏离。若成功，它将为实现一个真正统一的、能够涌现跨模态协同信息的世界模型奠定架构基础，是 Tiny-ONN 永续学习愿景的核心技术路径。

**时限 (Timebox):** 2 周（11 月启动）

**决策截止日期 (Decision Deadline):** 2025-11-16。此决策将直接影响后续的数据处理管线、模型扩展策略和训练基础设施的设计。

## 研究问题 (Research Question(s))

**主要问题 (Primary Question):** 单一 MoIE/DynSIHA 主干网络能否有效处理表示为统一序列的不同模态输入（例如，ARC 网格、图像 Patch、文本字符），并在共享的计算工作空间中学习到有意义的跨模态表征？

**次要问题 (Secondary Questions):**

- **嵌入有效性**: “分层嵌入注入”方案（已在 `exp/arc` 的 [`ArcEmbedding`](exp/arc/model.py:258) 中得到初步验证）在扩展到更多模态（如图像 Patch + 文本）时，能否有效保留各模态的关键信息（如空间几何 vs. 序列顺序）？
- **元学习可扩展性**: 在 `exp/arc` 中驱动专家分化的 SARS/CARC 元学习框架（见 [`exp/arc/train.py`](exp/arc/train.py:58)），在面对更复杂、更多样化的混合模态数据时，能否继续驱动有意义的、模态感知的专家功能特化？
- **性能与实现**: `exp/arc` 中基于 `torch.einsum` 的稠密计算实现，在模型规模从 ARC 的 ~0.1B 扩展到 3B/7B 目标时，其预期的性能瓶颈在哪里？与手写稀疏计算的潜在收益相比如何？
- **Tokenizer 策略**: 由 `word_embed` 支持的小型、基于字符的统一词汇表（如 [`exp/arc/tokenizer.py`](exp/arc/tokenizer.py:4) 中的 16 词汇表概念），能否在保持泛化能力的同时，为大规模语言任务提供足够的表达能力？

## 调查计划

### 研究任务

- [ ] **1. 设计通用嵌入层**: 基于 [`ArcEmbedding`](exp/arc/model.py:258) 的成功经验，设计并实现一个可处理文本、图像 Patch 和 ARC 网格的 `OmniEmbedding` 模块。
- [ ] **2. 构建混合模态数据集**: 创建一个原型 `DataLoader`，能够将来自 ARC 数据集和纯文本语料库的数据混合并序列化为统一格式，类似于 [`exp/arc/data.py`](exp/arc/data.py:12) 中的 `GridSerializer`。
- [ ] **3. 实施原型训练**: 使用 `exp/arc` 的训练循环（[`exp/arc/train.py`](exp/arc/train.py:153)）作为模板，在小规模（~0.1B）的 MoIE/DynSIHA 模型上，对混合模态数据进行有限步数的训练。
- [ ] **4. 分析与观察**:
  - [ ] 使用 [`Observer`](exp/arc/observer.py:20) 工具监控训练过程，重点观察模型在处理不同模态数据时，`PI-Score`、`activation_rate` 和专家原型分化（`visualize_prototypes`）的情况。
  - [ ] 评估模型是否在两种任务上都表现出学习迹象，而非灾难性遗忘或模态干扰。
- [ ] **5. 性能基准测试**: 对现有基于 `einsum` 的 [`spl_forward`](exp/arc/model.py:39) 函数进行性能分析，估算其在更大 `hidden_size` 和序列长度下的计算和内存开销。
- [ ] **6. 记录发现与建议**: 基于实验结果，撰写明确的 go/no-go 建议。

### 成功标准

**本次探索完成的标志是：**

- [ ] 原型模型在混合模态数据集上训练后，在文本任务上的损失均呈下降趋势，无灾难性干扰现象。
- [ ] `OmniEmbedding` 模块已实现并成功集成到模型中。
- [ ] `Observer` 的输出（特别是原型 PCA 可视化）显示出专家激活模式对不同输入模态的敏感性，表明初步的专家分化。
- [ ] 一份明确的、基于证据的建议已记录在案，用于指导 `Tiny-ONN-LLM-Omni` 的最终架构决策。

## 技术背景

**相关组件 (Related Components):**

- **核心架构**: [`exp/arc/model.py`](exp/arc/model.py:1) - `ArcTransformer`, `MoIETransformerBlock`, `SparseProtoLinear`
- **数据序列化**: [`exp/arc/data.py`](exp/arc/data.py:1) - `GridSerializer`
- **训练动力学**: [`exp/arc/train.py`](exp/arc/train.py:1) - `LearningDynamics`, JSD/CARC 损失

**依赖项 (Dependencies):** 此项探索是基础性的。所有关于数据预处理、大规模训练策略和模型扩展的未来决策均依赖于本次探索的结论。

**限制条件 (Constraints):** 如原始笔记所述，由于缺乏定制化 CUDA 核心，所有原型和实验必须基于 PyTorch 原生的 `einsum` 操作，这被证明是现阶段最稳健和高效的实现。

## 研究发现

### 调查结果

_[待填写：记录混合模态训练的损失曲线、准确率、以及从 Observer 收集到的关键指标，如 PI-Score 和激活率。]_

### 原型/测试记录

_[待填写：记录 `OmniEmbedding` 模块的实现细节和单元测试结果。附上原型训练中最具代表性的专家分化可视化图表。]_

### 外部资源

- [Tiny-ONN 理论基础：DFC-Theory.md](file://e:\Dev\Chain\Tiny-ONN.roo\rules\DFC-Theory.md)
- [Tiny-ONN 理论基础：Tiny-ONN-ARC-Theory.md](file://e:\Dev\Chain\Tiny-ONN.roo\rules\Tiny-ONN-ARC-Theory.md)

## 决策

### 建议

_[待填写：基于研究发现，提出明确的建议，例如：“建议采纳单塔 Omni-Modal 架构，并按计划扩展至 3B 参数规模”，或“建议暂停单塔方案，转而研究交叉注意力机制”。]_

### 理由

_[待填写：详细阐述做出上述建议的原因，并与次要研究问题的答案相关联。]_

### 实施说明

_[待填写：如果建议继续，列出下一步实施的关键考虑因素，例如“优先扩展数据处理管线以支持图像数据”或“启动 `triton` 或 `mojo` 实现稀疏计算核心的并行探索”。]_

### 后续行动

- [ ] [行动项 1]
- [ ] [行动项 2]
- [ ] 更新核心架构设计文档
- [ ] 创建后续实施任务

## 状态历史

| 日期       | 状态      | 备注                              |
| ---------- | --------- | --------------------------------- |
| 2025-10-02 | 🔴 未开始 | 探索文档已根据 `exp/arc` 经验创建 |
| [日期]     | 🟡 进行中 | 研究开始                          |
| [日期]     | 🟢 已完成 | [解决方案摘要]                    |

---

_上次更新时间：2025-10-02，由 Core-AI-Agent_
