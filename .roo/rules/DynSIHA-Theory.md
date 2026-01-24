# DynSIHA: Dynamic Sparse Infinite-Head Attention

`Latest update: 2026-01-23`
`Taxonomy: v2.1`

> 生命的本质，是用内在的确定性，对抗外部的不确定性。
> —— 林睿，于 2025 年 10 月

## Abstract

本文档旨在对 DynSIHA (Dynamic Sparse Infinite-Head Attention) 及其核心理念 DFC (Dynamic Function Composition) 进行形式化梳理与重构。

DynSIHA 是一套旨在将 Transformer 从“静态特征变换器”升级为“自适应动态计算系统”的演进方法论。其核心愿景是通过 **FARS (Fisher-Aware Routing Shaping)** 机制，驱动网络在训练过程中自组织出针对特定输入的稀疏计算路径。

本文将 DynSIHA 的演进形态正式划分为三个阶段：Dimension DynSIHA (特征级掩码)、Flat DynSIHA (模块级组装) 与 Recursive DynSIHA (全局递归再入)。这三种形态代表了我们在 Parameter Efficiency、Routing Granularity 和 Computational Irregularity 之间权衡的探索轨迹。

## Philosophy: Computational Density for Inference Sparsity

`Tiny-ONN` 项目的理论基石是 IPWT (Integrated Predictive Workspace Theory)，其核心目标是构建一个高效的变分推断引擎，最小化系统的 VFE (Variational Free Energy)。

### The Paradox of Sparsity

在深度学习中，稀疏性通常被视为提高推理速度的手段。然而，在 DynSIHA 中，稀疏性首先是认知层面的需求，其次才是计算层面的需求。

- Inference Sparsity: 模型在处理特定输入时，应当只激活其推断空间中极少部分的“概念通路”。这种稀疏性是模型具备可解释性、泛化能力和逻辑推理能力的前提。
- Computational Density: 为使 SGD 诱导稀疏结构自发涌现，在训练阶段维持计算稠密性更契合 GPU 硬件架构，是降低性能损耗、提升执行效率的可接受路径。

### Dynamic Function Composition (DFC)

传统的神经网络层是静态的：`y = f(x; θ)`，其中 `f` 的结构在推理时是固定的。

DFC 试图实现：`y = fₓ(x; θ)`，即函数 `f` 本身是输入 `x` 的函数。

## Taxonomy: The Three Forms of DynSIHA

基于粒度和资源组织形式的演进，DynSIHA 经历了从微观特征掩码到宏观递归生成的三个发展阶段。这不仅是架构的迭代，更是我们对“动态计算”理解逐步深化的过程。

### Dimension DynSIHA (DDS)

**Dimension DynSIHA**（早期称为 SPL, SparseProtoLinear）代表了 DFC 理念的初次尝试。在这个阶段，我们将每一个神经元（输出维度）都视为一个独立的“微型专家”。

其核心机制在于**特征级路由**：路由决策直接作用于特征维度，决定是否激活特定的神经元。理论上，这是一种完美的形态——它采用“稠密计算 + 掩码”的模式，完全避免了传统 MoE 架构中 Token 分发带来的通信瓶颈（Zero Communication Overhead）。其背后的数学直觉是将 Dropout 视为变分推断（Variational Inference），试图从稠密权重中自发涌现出稀疏子网络。

然而，现实是残酷的。在工程实践中，这种细粒度的路由极易导致**原型坍塌（Prototype Collapse）**。系统往往陷入一种病态解：99% 的输出被直接屏蔽，CAPR 层从“路由器”退化为阻断信息流的“断路器”。因此，该形态目前已被**废弃**，仅作为理论原型保留在 [`archive/dimension-dynsiha.py`](archive/dimension-dynsiha.py) 中作为参考。

### Flat DynSIHA (FDS)

吸取了 DDS 的教训后，我们转向了工程可行性更高的 **Flat DynSIHA**。这是目前 ARC 任务的主力架构，和传统Transformer具有良好的兼容性。

FDS 保留了标准 Transformer 的层级堆叠结构，但在 Block 内部引入了**模块级组装（Module-level Assembly）**。模型由 $L$ 个独立的 Block 堆叠而成，每个 Block 拥有独立的参数空间（Head Repo, MLP Repo）。

其路由逻辑遵循**局部性原则**：

1. **Local Routing**: 路由决策仅在当前 Block 内部有效，Block $i$ 的资源不能被 Block $j$ 复用。
2. **Weighted Assembly**: 类似于 Soft-MoE，根据路由权重对组件输出进行加权组合，并通过 `ReLU + bias` 实现稀疏选择。

这种设计高度符合神经科学中**功能柱（Cortical Columns）**的组织方式，在保证模块化的同时，实现了计算资源的动态分配。

### Recursive DynSIHA (RDS-ACT)

> _Previously known as: DynSIHA+, DynTRM_

**Recursive DynSIHA** 是 DFC 理念的终极形态，旨在实现全局动态组装与动态程序生成。

与前两者不同，RDS 在物理上只有一个**递归块（Recursive Block）**，但在时间上展开为 `t` 个精炼步骤（Refinement Steps）。所有的 Attention Heads 和 MLP Experts 都存储在一个**全局共享仓库（Global Repo）**中。

这种架构实际上是在模拟**图灵机**的操作：

- **Repo** 是指令集；
- **State** 是纸带；
- **Routing** 是控制单元。

无论是在第 1 步还是第 `t` 步，模型都从同一个工具箱中挑选工具进行加工。这不再是层层变换，而是基于状态的递归精炼。

尽管具有极高的理论价值，RDS 目前仍处于**实验阶段**。其最大的挑战在于**显存访问的不规则性（Memory Irregularity）**：由于每一步和每个样本调用的 Expert 不同，这种“指针跳转”式的计算模式难以利用现代 GPU 的 Tensor Core 加速，导致严重的性能损耗。

## Core Mechanism: Dynamic Routing

路由机制是 DynSIHA 的大脑，它决定了信息如何在推断空间中流动。DynSIHA 的核心不在于特定的静态变换，而在于如何通过语义匹配动态地组装计算路径。我们经历了从语义原型匹配回到通用函数逼近的认知演进。

### CPR/CAPR - Deprecated

- Mechanism: 基于余弦相似度 (CPR) 或 交叉注意力 (CAPR) 的原型匹配。
- Verdict: **已废弃**。实验证明，基于原型的路由具有强烈的“中心对称”归纳偏置，在处理 XOR 等非线性逻辑规则时表现极差。虽然在聚类任务中具有可解释性，但在 ARC 这种逻辑密集型任务中，其表达力不足成为了瓶颈。

### MLP Router

- Mechanism: 采用标准的 2-layer MLP (`Linear -> SiLU -> Linear`)。
- Insight: 放弃了显式的“原型”隐喻，回归通用函数近似。实验证明，MLP 路由在 XOR 逻辑、条件分支和复杂流形划分上具有压倒性的表达力。我们不再强求路由过程本身的显式可解释性，而是依靠 **ARS2-Neo (AGA)** 优化器在黎曼流形上的平坦度约束，确保 MLP 路由不会陷入过拟合，并配合 **FARS** 进行认知代价塑造。

### Computation Unit: The MLP Expert (formerly SPL)

所谓的 SPL (SparseProtoLinear) 实际上是受路由掩码控制的计算单元。它并非简单的线性层，而是一个高度非线性的 MLP Expert。它通过 Markov Blanket 的隐喻，将神经元的“计算”与“决策”在功能上解耦。

#### Mathematical Definition

一个标准的计算单元被重构为：`y = MLP(x; μ) ⊙ mask`

1. `μ` (mu_weight): 代表“计算工具集”，本质上是一个两层 MLP (`Linear -> SiLU -> Linear`)。
2. `Routing Path`: 由 CPR 或 CAPR 生成的动态掩码 `mask`。

### Prototype Resident Connection (PRC)

在 Deep Network 中，随着层数加深，用于驱动路由的梯度信号会迅速衰减。为了解决这个问题，我们引入了 PRC。

- TRC (Traditional Residual Connection): `x_{l+1} = x_l + f(x_l)`。作用于 Block 之间，保护主任务梯度流。
- PRC (Prototype Resident Connection): 作用于 SPL 内部的 Routing Path。
  `pᵢₙₚᵤₜ⁽ᵗ⁾ = x⁽ᵗ⁾ + α ⋅ pᵢₙₚᵤₜ⁽ᵗ⁻¹⁾`
  PRC 确保了路由决策所依赖的输入表征（Prototype Input）是在上一层基础上的 增量修正 (Incremental Correction)，而非完全重构。这为 Router 提供了一个稳定的、分层演进的上下文环境。

## Future Directions

### FARS (Fisher-Aware Routing Shaping)

旧版的 SARS (已归档至 [`archive/SARS.md`](archive/SARS.md)) 依赖于一阶梯度范数 (`‖ ∇ ℒ ‖`) 来衡量 Cost。然而，瞬时梯度包含大量噪声。
FARS 提议利用 Fisher Information Matrix (FIM) 的对角近似（即 Adam 的二阶矩 `vₜ`）来替代梯度范数。
`Cost_FARS ≈ √vₜ`
Fisher Information 衡量了参数空间的曲率，即“参数变化对分布的影响程度”。使用 FARS 可以提供更稳定、更符合信息几何本质的路由塑造信号。

### Standardization

下一步的工程重点是完成 `MLPRouter` 在 `src/models/dynsiha/shared/layers.py` 中的标准化封装，并将其集成到 `FlatDynSIHA` 架构中。同时，确保 `ARCTrainer` 能够通过 `RoutingShaper` 完美驱动 FARS 塑造流程。
