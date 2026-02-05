# DynSIHA: Dynamic Sparse Infinite-Head Attention

`Latest update: 2026-01-25`
`Taxonomy: v2.2`

> 生命的本质，是用内在的确定性，对抗外部的不确定性。
> —— 林睿，于 2025 年 10 月

## Abstract

本文档旨在对 DynSIHA (Dynamic Sparse Infinite-Head Attention) 及其核心理念 DFC (Dynamic Function Composition) 进行形式化梳理与重构。

DynSIHA 是一套旨在将 Transformer 从“静态特征变换器”升级为“自适应动态计算系统”的演进方法论。其核心愿景是通过 **FARS (Fisher-Aware Routing Shaping)** 机制，驱动网络在训练过程中自组织出针对特定输入的稀疏计算路径。

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

### Dimension DynSIHA (DDS) - Deprecated

代表了 DFC 理念的初次尝试。将每一个神经元视为独立的“微型专家”。因极易导致原型坍塌（Prototype Collapse）已被废弃。

### Flat DynSIHA (FDS)

目前 ARC 任务的主力架构。保留层级堆叠，但在 Block 内部引入模块级组装。遵循局部性原则，实现计算资源的动态分配。

### Recursive DynSIHA (RDS-ACT)

> TL;DR: 传送带 (RNN) vs 塔吊 (DynSIHA)
>
> - 传送带 (RNN): 单一权重，固定循环。数据在传送带上反复经过同一个加工点，状态被动演化。
> - 塔吊 (DynSIHA): 共享权重，自展开组装。每一步 $t$ 都是在 $0 \dots t-1$ 搭建的结构之上进行的增量建设。递归不是循环，而是推断空间中的深度生长。

DFC 理念的终极形态。物理上只有一个递归块，但在时间上展开为 $t$ 个精炼步骤。所有的组件存储在全局共享仓库中，模拟图灵机的操作。

## Core Mechanism: Dynamic Routing

### MLP Router

采用标准的 2-layer MLP (`Linear -> SiLU -> Linear`)。依靠 **ARS2-Neo (AGA)** 优化器在黎曼流形上的平坦度约束，确保路由不会陷入过拟合，并配合 **FARS** 进行认知代价塑造。

### Prototype Resident Connection (PRC)

`pᵢₙₚᵤₜ⁽ᵗ⁾ = x⁽ᵗ⁾ + α ⋅ pᵢₙₚᵤₜ⁽ᵗ⁻¹⁾`
确保路由决策所依赖的输入表征是在上一层基础上的增量修正，为 Router 提供稳定的、分层演进的上下文。

# FARS: Fisher-Aware Routing Shaping

> **TL;DR**: FARS 是一种“二阶矩范数加权负载均衡损失”。它利用优化器状态（Fisher 信息近似）来量化专家的认知代价，驱动路由器将数据分发给那些能以最小参数位移解释最大输出贡献的专家。

## 1. 核心逻辑：梯度-路由对偶性 (Gradient-Routing Duality)

FARS 解决了旧版 SARS (Surprise-Aware) 的量纲冲突与高频噪声问题，也顺带解决了二次链式法则的性能问题。

- **认知代价 (Cost)**: 利用 [`ARS2-Neo`](src/optimizers/ars2_neo.py) 的二阶矩 $\sqrt{v_t}$ 作为 Fisher 信息的对角近似。它衡量了参数空间的局部曲率，即“参数变化对分布的影响程度”。
- **重要性 (Importance)**: 专家输出的梯度范数 ‖ ∇_out ℒ ‖。

我们发现，**主任务梯度天然携带了样本级的 Importance 信息**。因此，无需显式计算 Importance，只需将 Cost 作为正则项引入，即可实现等效的语义路由。

## 2. 统一路由塑造公式

路由器的目标是最小化以下塑造信号（Shaping Signal）：
`𝒢 = Belief ⋅ Cost_FARS`

- `Belief`: 路由器的 Softmax 权重。
- `Cost_FARS`: Norm(√v_t)，代表专家的认知复杂度。

在反向传播中，路由器参数 $\phi$ 接收到的总梯度为：
`∇_ϕ ℒ_total = ∇_ϕ ℒ_main + λ ⋅ ∇_ϕ (Belief ⋅ Cost)`

- **Utility (∇_ϕ ℒ_main)**: 包含样本特异性。如果专家 $e$ 对样本 $i$ 很重要，该项梯度会很大，驱动路由器选择它。
- **Tax (λ ⋅ Cost)**: 提供全局背景阻力。如果专家 $e$ 认知压力大，该项梯度会抑制选择。

这种**效用-代价平衡**等价于旧版 SARS 的复杂公式，但完全避免了昂贵的样本级梯度捕获。

## 3. 几何本质：切空间对齐

FARS 迫使路由器将数据分发至局部几何平坦的专家：

- **高 Cost**: 参数更新剧烈，局部曲率大，过拟合风险高（对应高 Kolmogorov 复杂度程序）。
- **低 Cost**: 参数更新微小，局部几何平坦，泛化能力强（对应最小描述长度 MDL 程序）。

### 4. 核心指标定义

- **ITJD (Inter-Task Jaccard Distance)**: 衡量不同任务路由分布的隔离程度。
- **RMI (Routing Mutual Information)**: 衡量 Task ID 对路由决策的确定性增益。

### 5.梯度-路由对偶性 (Gradient-Routing Duality)

主任务梯度天然携带了样本级的 Importance 信息。无需显式计算 Importance，只需将 Cost 作为正则项引入：
`𝒢 = Belief ⋅ Cost_FARS`
其中 `Cost_FARS ≈ √vₜ`（优化器二阶矩的对角近似）。

### 6. 轨迹正则化 (Trajectory Regularization)

在 RDS 架构中，我们不只惩罚神谕步 $t^*$，而是惩罚从 $t_0$ 到 $t^*$ 的**整条路径**：
`Loss_FARS = (1 / t*) * Σ_{s=0}^{t*} (Belief_s ⋅ Cost_FARS)`
这确保了“塔吊”在每一层上升过程中都保持最小描述长度（MDL）原则。

## PLSD-ACT: 递归动力学对齐

### 1. PLSD (Per-Layer Speculative Decode)

放弃强化学习的试错范式，转向全量观测的自监督对齐。

- **Oracle 步长**: $t^* = \text{argmin}_t \mathcal{L}_t$。代表模型在当前参数下能达到的认知饱和点。
- **神谕对齐**: 强制模型拟合以 $t^*$ 为转折点的阶跃函数。

### 2. ACT (Adaptive Computation Time)

寻找映射 $t = \phi(x)$，使计算资源与任务难度实现帕累托最优。

- **训练期**: 展开至 $T_{max}$，通过 PLSD 锁定 $t^*$。
- **推理期**: 遵循 `halt_head` 的预测进行自适应早停，支持比训练期更深的 fallback 深度。

## Future Directions

### Standardization

完成 `MLPRouter` 的标准化封装，并确保 `ARCTrainer` 能够通过 `RoutingShaper` 完美驱动轨迹 FARS 塑造流程。同时集成 **ADL (Adaptive Differential Loss)** 作为样本轨迹的“重力系数”。
