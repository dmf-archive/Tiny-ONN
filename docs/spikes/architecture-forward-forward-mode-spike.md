---
title: "Tech-Spike-1: 引入 Forward-Forward (FF) 学习模式的技术预研"
category: "架构"
status: "🔴 未开始"
priority: "高"
timebox: "1 周"
created: "2025-10-06"
updated: "2025-10-06"
owner: "Ω Researcher, Tiny-ONN 课题组"
tags: ["技术预研", "架构", "研究", "forward-forward", "continual-learning"]
---

# Tech-Spike-1: 引入 Forward-Forward (FF) 学习模式的技术预研

## 摘要

**探索目标 (Spike Objective):** 尽管 [ADR-0002](../adr/adr-0002-inference-space-conceptual-sparsity.md) 决定当前阶段坚持使用反向传播 (BP)，但 `Forward-Forward (FF)` 范式作为解决未来“可扩展性悖论”的关键技术路径，其理论潜力巨大。本次技术预研的目标是：形式化地定义一个可行的、从当前 BP 模式平滑过渡到未来 FF 模式的技术路线图，并设计一个初步的实验方案，用于在 ARC 环境下小规模验证 FF 的核心机制。

**重要性 (Why This Matters):** 解决 `Tiny-ONN` 架构长期扩展性的核心瓶颈，为模型从“实验室模式”（依赖全局梯度）走向“边缘部署/在线学习模式”（依赖本地信号）提供理论和工程基础。

**时限 (Timebox):** 1 周

**决策截止日期 (Decision Deadline):** 2025-10-20

## 研究问题 (Research Question(s))

**主要问题 (Primary Question):** 如何在 `SPL` 架构下，设计并实现一个与当前 `SARS` on BP 范式在理论上兼容、在工程上可切换的 `Forward-Forward` 学习模式？

**次要问题 (Secondary Questions):**

- FF 模式下最合适的本地“好度”函数是什么？是直接使用 `原型匹配度`，还是需要一个更复杂的、包含“惊奇度”信号的函数？
- 如何在 ARC 这种非成对 (non-paired) 数据集上，高效地生成用于 FF 对比学习的“负样本”？
- 在 ARC 任务上，FF 的收敛速度、最终性能和抗灾难性遗忘能力与 BP 相比具体表现如何？
- 从 BP 切换到 FF 是否需要对 `SPL` 架构本身进行修改，或者仅仅是替换优化器和训练循环？

## 调查计划

### 研究任务

- [ ] 形式化推导 FF 在 SPL 架构下的本地更新规则，明确其与 VFE 优化的关系。
- [ ] 在 `tiny_onn/` 模块中实现一个实验性的 `FF-Optimizer`。
- [ ] 设计一个最小化的 ARC 任务子集用于快速迭代测试。
- [ ] 创建一个概念验证 (PoC) 脚本 (`exp/arc/train_ff.py`)，在指定任务上运行 FF 模式并记录学习曲线。
- [ ] 记录实验发现，并与 BP 基线进行对比分析。

### 成功标准

**本次探索完成的标志是：**

- [ ] 完成了 FF 本地更新规则的数学形式化文档。
- [ ] PoC 代码能够成功运行，模型参数能够通过 FF 模式进行有意义的更新。
- [ ] 记录了 FF 与 BP 在至少一个 ARC 任务上的初步性能（损失曲线、准确率）对比数据。
- [ ] 输出了一个关于“是否以及何时”将 FF 集成到主干分支的明确建议。

## 技术背景

**相关组件 (Related Components):**

- `tiny_onn/modular.py` (尤其是 `SparseProtoLinear` 类)
- `exp/arc/train.py` (当前的训练循环和优化器逻辑)
- `docs/adr/adr-0002-inference-space-conceptual-sparsity.md` (理论基础)

**依赖项 (Dependencies):** 依赖于 ADR-0002 中对 BP 和 FF 理论等效性的分析。

**限制条件 (Constraints):** PoC 阶段不追求极致性能，优先验证机制的可行性。必须保持与现有 `SPL` 架构的高度兼容性。

## 研究发现

### 调查结果

_[待填写]_

### 原型/测试记录

_[待填写]_

### 外部资源

- [Geoffrey Hinton's original paper on The Forward-Forward Algorithm](https://www.cs.toronto.edu/~hinton/absps/ff.pdf)
- [Relevant discussions on Policy Gradient and Reinforcement Learning](https://spinningup.openai.com/en/latest/algorithms/vpg.html)

## 决策

### 建议

_[待填写]_

### 理由

_[待填写]_

### 实施说明

_[待填写]_

### 后续行动

- [ ] [行动项 1]
- [ ] [行动项 2]
- [ ] [更新架构文档]
- [ ] [创建实施任务]

## 状态历史

| 日期       | 状态      | 备注                     |
| :--------- | :-------- | :----------------------- |
| 2025-10-06 | 🔴 未开始 | 探索文档已创建并确定范围 |
| [日期]     | 🟡 进行中 | 研究开始                 |
| [日期]     | 🟢 已完成 | [解决方案摘要]           |

---

_上次更新时间：2025-10-06，由 Ω Researcher_
