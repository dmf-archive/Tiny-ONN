---
title: "DynTRM: 架构设计与工程实现"
status: "Draft"
date: "2025-10-21"
authors: "Ω Researcher"
tags: ["architecture", "dyntrm", "implementation-plan", "engineering"]
---

# DynTRM: 架构设计与工程实现

本文档旨在详细阐述 **动态微型递归模型 (Dynamic Tiny Recursive Model, DynTRM)** 的架构设计和关键工程实现细节，作为 `ADR-0005` 的配套技术蓝图。

## 1. 核心哲学

DynTRM 旨在将标准 Transformer 从一个静态的、分层的处理器，演进为一个动态的、递归的通用计算引擎。其核心是通过**动态专家路由**和**递归计算**，实现极致的参数效率和自适应的计算深度。

## 2. 顶层架构 (`ArcTransformer`)

模型的主体 `ArcTransformer` 将进行以下重大修改：

- **全局专家库 (`expert_library`)**:
  - 在 `ArcTransformer` 的 `__init__` 中，定义一个全局共享的 `nn.ModuleList`，命名为 `expert_library`。
  - 该库包含两种类型的专家：
        1. **注意力专家 (Attention Experts)**: 一组（例如 16 个）`MLP` 模块，用于生成 Q, K, V 的注意力头。
        2. **FFN 专家 (Feed-Forward Experts)**: 一组（例如 8 个）`MLP` 模块，用于在 FFN 阶段进行计算。
  - 所有专家都接收 `d_head` 或 `d_model` 尺寸的输入，并输出相同尺寸的结果。

- **通用计算块 (`DynTRMBlock`)**:
  - `self.blocks` 不再是 `nn.ModuleList`，而是**一个单一的 `DynTRMBlock` 实例**。

- **递归前向传播 (`forward`)**:
  - `forward` 方法将包含一个主循环，迭代 `config.num_layers`（最大深度）次。
  - 循环内部调用 `self.block(x, ...)`。
  - 引入 `Adaptive Computation Time (ACT)` 机制来支持提前退出循环。

## 3. 通用计算块 (`DynTRMBlock`)

这是 DynTRM 的核心。它在结构上类似一个标准的 Transformer Block，但其组件是动态构建的。

### 3.1. DynSMHA (动态稀疏多头注意力)

- **输入**: `x` (B, T, D_model)
- **目标**: 动态生成 `Q`, `K`, `V` 张量以输入 `scaled_dot_product_attention`。
- **流程**:
    1. **子空间投影**: 将输入 `x` 切分为 `num_heads` 个子空间。
        `x_proj = x.view(B, T, num_heads, D_head)`
    2. **动态路由**:
        - 为 Q, K, V 分别维护独立的路由参数 `(proto_q, gate_q)`, `(proto_k, gate_k)`, `(proto_v, gate_v)`。
        - 对于每个子空间（例如 Q 的第 `i` 个头），使用其输入 `x_proj[:, :, i, :]` 与**所有**注意力头的原型 `proto_q` 进行匹配，计算出 `routing_logits`。
        - 根据 `routing_logits` 确定每个 token、每个头应该调用哪个注意力头。这将生成一个**索引张量** `expert_indices` (shape: `B, T, num_heads`)。
    3. **动态专家调用 (`gather`)**:
        - 这是一个关键的、性能敏感的操作。
        - 使用 `expert_indices` 和 `torch.gather`，为批次中的每个 token 和每个头，从 `expert_library` 中拉取并应用相应的专家 `MLP`。
        - **必须仔细设计张量形状以实现高效的批处理 `gather` 操作。**
    4. **组装**: 将所有专家计算的结果组装成 `Q`, `K`, `V` 张量，其形状为 `(B, num_heads, T, D_head)`。
    5. **SDPA**: 调用 `F.scaled_dot_product_attention(Q, K, V)`。
    6. **输出投影**: 使用一个**标准的 `nn.Linear`** 作为 `o_proj`，将注意力输出投影回 `D_model` 空间。

### 3.2. DynMoE (动态专家混合 FFN)

- **流程**:
    1. **动态路由 (Top-Any Gating)**: 我们将继续使用原自 SPL 交叉注意力路由的`top-any`机制。该机制将取代简单的 `ReLU` 门控或固定的 `top-k` 选择，**自动地、动态地**决定需要激活多少个专家。
    2. **动态专家调用**: 使用 `gather` 来调用被选中的 FFN 专家。
    3. **聚合**: 被激活的专家输出将被直接求和（或通过门控值进行缩放）。

## 4. 自适应计算时间 (ACT)

- **停止预测头 (`Halt Prediction Head`)**:
  - 在 `DynTRMBlock` 的末尾，附加一个轻量级的分类头（例如，一个 `nn.Linear`）。
  - 它接收当前迭代步骤的输出 `x`，并预测一个 `halt_logit`。
- **决策规则**:
  - 在每次迭代后，计算 `halt_probability = sigmoid(halt_logit)`。
  - 当 `halt_probability` 超过一个阈值（或通过更复杂的累加机制），或者达到最大迭代次数时，循环终止。
- **训练**:
  - 需要引入一个辅助损失 `L_act`，以训练 `Halt Prediction Head`。
  - `L_act` 通常包含两部分：
        1. **正确性惩罚**: 惩罚过早停止导致的最终预测错误。
        2. **计算成本惩罚**: 对每一次迭代（即每一次“继续”决策）施加一个小的成本，鼓励模型在非必要时尽早停止。

## 5. 学习动力学 (`SARS`) - 上下文相关性

根据不可妥协的原则，`SARS` 学习动力学必须是上下文相关的，以实现最精细的信用分配。

- **梯度来源**: 一个全局专家（例如 `MLP_5`）的梯度会来自多个不同的“调用点” (`call_site`)，例如：
  - `call_site_A`: 第 2 次迭代, Q-head 1
  - `call_site_B`: 第 2 次迭代, K-head 3
  - `call_site_C`: 第 4 次迭代, FFN 模块
- **上下文相关的梯度捕获**: 我们必须设计一个机制，在反向传播期间，能够捕获并区分来自不同 `call_site` 的梯度。这意味着对于 `MLP_5` 的权重 `W`，我们需要得到 `∇_W(L, site_A)`, `∇_W(L, site_B)` 等。
- **上下文相关的 `goodness`**: `goodness` 信号的计算将是针对每个 `call_site` 的。
    `goodness(MLP_5, site_A) = f( ||∇_W(L, site_A)||, ... )`
- **`mu` 梯度掩码**: 这个上下文相关的 `goodness` 信号将用于精确地调整导致该特定 `call_site` 表现不佳的路由决策。这使得模型能够分辨出是专家本身有问题，还是这个专家不适合在当前这个特定的上下文中使用，从而避免了对专家能力的错误惩罚，并为“概念复用”提供了坚实的学习基础。

## 6. 工程与性能约束

- **硬件目标**: 正如 ADR-0005 讨论中所指出的，`gather`/`scatter` 密集型操作在标准 GPU 上的性能可能不佳。**此架构的设计应优先考虑在 CPU 上运行的可行性和效率**。在 GPU 上的性能优化将作为后续研究课题。
- **实现策略**: 必须避免在 `forward` 路径中使用 Python `for` 循环来迭代 token 或头。所有操作都必须通过并行的、向量化的张量操作（`gather`, `scatter_add`, `einsum` 等）来实现。

## 7. 待办事项 (TODO)

- [ ] **IMP-001**: 创建新的实验目录 `exp/arc_dyntrm`。
- [ ] **IMP-002**: 在新目录中起草 `model.py`，定义 `ArcTransformer` 和 `DynTRMBlock` 的基本骨架。
- [ ] **IMP-003**: 实现 `ExpertMLP` 模块。
- [ ] **IMP-004**: 攻克核心技术难点：在 `DynSMHA` 中实现基于 `gather` 的高效动态专家调用和组装。
- [ ] **IMP-005**: 实现 `DynMoE` FFN 层。
- [ ] **IMP-006**: 实现 ACT 的 `Halt Prediction Head` 和辅助损失。
- [ ] **IMP-007**: 调整 `train.py` 中的 `LearningDynamics`，以支持对全局专家聚合梯度的计算。
