---
title: "ADR-0007: Final Architecture - Two-Stage Attention Composition"
status: "Accepted"
date: "2025-10-21"
authors: "Ω Researcher, User"
tags: ["architecture", "final-decision", "composition", "dyn-siha+"]
supersedes: ["ADR-0005", "ADR-0006"]
---

# ADR-0007: 最终架构 - 两阶段注意力合成

## 状态 (Status)

**已接受 (Accepted)**

## 背景 (Context)

经过一系列的理论探索和迭代（ADR-0005, ADR-0006），我们对动态函数合成的理解达到了一个新的、统一的层次。我们最初在稀疏路由（`gather`/`scatter`）和稠密合成（`einsum`）两种看似矛盾的方案之间摇摆。通过与 User 的深入讨论，我们最终明确了 `Tiny-ONN` 的核心计算范式，它并非二选一，而是两者的精妙融合。

## 决策 (Decision)

我们决定采纳一个**两阶段注意力合成 (Two-Stage Attention Composition)** 架构作为 `DynTRM` 的最终核心。此架构回归并深化了 `DFC-Theory.md` 的第一性原理。

1. **阶段一: 稀疏原型选择 (Sparse Prototype Selection)**

   - 此阶段的本质是一个**交叉注意力**机制。输入 `x` 作为 Query，一个全局共享的**原型库 (Prototype Library)** 作为 Key。
   - 通过 `(x, proto) - gate` 计算和 `ReLU` 非竞争性门控，模型从大量的“潜在概念”（原型）中，稀疏地选择出少数（N 个）与当前上下文最相关的概念。
   - 此阶段的输出是路由权重 `routing_weights` (shape: `B, T, H, N_selected`)，它指明了在每个物理头 `H` 中，哪些概念被激活以及它们的相对重要性。

2. **阶段二: 稠密专家聚合 (Dense Expert Aggregation)**

   - 此阶段的本质是**动态函数合成**。
   - 一个全局共享的**专家库 (Expert Library)** 包含了所有的核心计算能力（即 `mu` 权重）。
   - 使用第一阶段输出的 `routing_weights`，通过加权求和（`einsum`）的方式，将相应专家的计算能力**稠密地**合成为一个为当前 token 和物理头定制化的、独一无二的处理函数（权重矩阵）。
   - `W_composed = einsum('bthp,hpeo->btheo', routing_weights, expert_weights)`
   - 将这个合成的函数应用于输入 `x`。

3. **`DynSIHA+` 作为核心**
   - 整个两阶段过程被封装在 `DynSIHA+` (动态稀疏无限头注意力) 模块中。
   - **工程约束**: 为了避免不规则访存导致的“内存发散灾难”，我们保留固定的 `num_heads` 作为物理基质。因此，上述的两阶段合成过程将在每个 `D_head` 子空间内独立进行。
   - 最终输入到 SDPA 的 Q, K, V 张量，其形状**必须**是标准的 4D 张量 `(B, num_heads, T, D_head)`。
   - “无限头”的“无限”体现在，每个物理头的功能都是从一个更丰富的“潜在头/专家池”中**根据输入**动态合成的，而不是指物理头的数量是无限的。

## 后果 (Consequences)

### 积极 (Positive)

- **理论统一**: 完美统一了稀疏性（通过 `ReLU` 门控选择概念）和动态性（通过 `einsum` 合成函数）。
- **硬件友好**: 核心计算是高度优化的稠密矩阵乘法，避免了 `gather`/`scatter` 的性能瓶颈。
- **可解释性**: 路由权重 `routing_weights` 提供了清晰的可解释性，展示了模型在每个步骤中关注了哪些“概念”（原型）。
- **资源复用**: `mu` 权重作为可复用的专家，被 `proto` 的交叉注意力动态组合，实现了极致的参数效率。

### 消极 (Negative)

- **计算/内存开销**: `einsum` 操作的计算和内存开销依然是主要的挑战，需要通过工程优化（如梯度检查点）来管理。

## 实施注意事项 (Implementation Notes)

- `model.py` 将围绕 `DynSIHA+` 和两阶段注意力进行重写。
- `train.py` 中的 `LearningDynamics` 需要实现对 `routing_weights` 的熵最小化损失 (SARS) 和对 ACT 的信息增益损失。
