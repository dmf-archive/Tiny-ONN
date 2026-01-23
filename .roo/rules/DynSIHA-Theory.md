# DynSIHA: Dynamic Sparse Infinite-Head Attention

`Latest update: 2026-01-23`
`Taxonomy: v2.1`

> 生命的本质，是用内在的确定性，对抗外部的不确定性。
> —— 林睿，于 2025 年 10 月

## 0. Abstract

本文档旨在对 DynSIHA (Dynamic Sparse Infinite-Head Attention) 及其核心理念 DFC (Dynamic Function Composition) 进行形式化梳理与重构。

DynSIHA 是一套旨在将 Transformer 从“静态特征变换器”升级为“自适应动态计算系统”的演进方法论。其核心愿景是通过 **FARS (Fisher-Aware Routing Shaping)** 机制，驱动网络在训练过程中自组织出针对特定输入的稀疏计算路径。

本文将 DynSIHA 的演进形态正式划分为三个阶段：Dimension DynSIHA (特征级掩码)、Flat DynSIHA (模块级组装) 与 Recursive DynSIHA (全局递归再入)。这三种形态代表了我们在 Parameter Efficiency、Routing Granularity 和 Computational Irregularity 之间权衡的探索轨迹。

## 1. Philosophy: Computational Density for Inference Sparsity

`Tiny-ONN` 项目的理论基石是 IPWT (Integrated Predictive Workspace Theory)，其核心目标是构建一个高效的变分推断引擎，最小化系统的 VFE (Variational Free Energy)。

### 1.1 The Paradox of Sparsity

在深度学习中，稀疏性通常被视为提高推理速度的手段（如 MoE 的 conditional computation）。然而，在 DynSIHA 中，稀疏性首先是认知 (Cognitive) 层面的需求，其次才是计算层面的需求。

- Inference Sparsity (Activation Sparsity): 模型在处理特定输入时，应当只激活其推断空间（Inference Space）中极少部分的“概念通路”。这种稀疏性是模型具备可解释性、泛化能力和逻辑推理能力的前提。
- Computational Density: 为使 SGD 诱导稀疏结构自发涌现，在训练阶段维持计算稠密性更契合 GPU 硬件架构，是降低性能损耗、提升执行效率的最优路径。

### 1.2 Dynamic Function Composition (DFC)

传统的神经网络层是静态的：`y = f(x; θ)`，其中 `f` 的结构在推理时是固定的。
DFC 试图实现：`y = fₓ(x; θ)`，即函数 `f` 本身是输入 `x` 的函数。

## 2. Taxonomy: The Three Forms of DynSIHA

基于 Granularity 和 Resource Organization，将 DynSIHA 的演进定义为以下三种形态。

### 2.1 Dimension DynSIHA, DDS

这是 DFC 理念的初次尝试，其核心特征是 Feature-level Masking。在项目早期，这一形态曾被称为 SPL (SparseProtoLinear)。

- Mechanism: 在这个阶段，[`SparseProtoLinear`](archive/dimension-dynsiha.py:85) 模块内的每一个神经元（输出维度）都被视为一个独立的“微型专家”。路由机制直接作用于特征维度，决定是否激活该神经元。
- Pros (The Theoretical Ideal):
  - Zero Communication Overhead: 理论上它的算力浪费是最小的。因为它采用“稠密计算 + 掩码”的模式，完全不存在传统 MoE 中 Token 分发到 Expert 的 All-to-All 通信瓶颈。
  - Dropout as VI: 其理论基础源于将 Dropout 视为变分推断 (Variational Inference) 的思想，旨在从“稠密权重中寻找稀疏子网络”。
- Cons (The Practical Collapse):
  - Prototype Collapse: 在实践中，由于路由训练机制的动力学问题，系统极易陷入“原型坍塌”。
  - Information Blockage: 坍塌的结果是 99% 的输出直接被 Mask 掉。此时，CAPR 层根本没有起到“路由选择”的作用，反而变成了彻底阻断信息流的“断路器”。
- Status: Deprecated / Obsolete. 作为过时的理论原型，已不再用于实际工程，仅保留在 [`archive/dimension-dynsiha.py`](archive/dimension-dynsiha.py) 中作为参考。

### 2.2 Flat DynSIHA, FDS

这是目前工程可行性最高的形态。保留了标准 Transformer 的 Block Stacking 结构，但在 Block 内部引入了 Module-level Assembly。

- Structure: 模型由 `L` 个独立的 Block 堆叠而成。每个 Block 拥有自己独立的参数空间（Head Repo, MLP Repo）。
- Routing Logic:
  - Local Routing: 路由决策仅在当前 Block 内部有效。
  - Weighted Assembly: 类似于 Soft-MoE，它根据路由权重对 Head 或 MLP 的输出进行加权组合，并通过 `ReLU + bias` 实现稀疏选择。
  - Independence: Block `i` 的 Head 不能被 Block `j` 复用。
- Pros:
  - Modularity: 符合神经科学中“功能柱” (Cortical Columns) 的组织方式。
- Status: Current Baseline. 是目前 ARC 任务的主力架构。

### 2.3 Recursive DynSIHA, RDS-ACT

> _Previously known as: DynSIHA+, DynTRM_

这是 DFC 理念的终极形态，旨在实现 Global Dynamic Assembly 和 Dynamic Program Generation。

- Structure: Single Recursive Block。整个网络在物理上只有一个 Block，但在时间上展开为 `T` 个 Refinement Steps。
- Resource Organization:
  - Global Repo: 所有的 Attention Heads 和 MLP Experts 存储在一个全局共享的 Repository 中。
  - Universal Routing: 无论是在第 1 步还是第 `T` 步，模型都从同一个 Repo 中抓取计算基元。
- Mechanism:
  - 模型不再是“经过层层变换”，而是“在每一步递归中，根据当前状态，从工具箱中挑选工具进行加工”。
  - 这实际上是在通过神经网络模拟 图灵机 的操作：Repo 是指令集，State 是纸带，Routing 是控制单元。
- Cons (The "Performance Disaster"):
  - Memory Irregularity: 由于每一步调用的 Expert 不同，且不同 Sample 调用的 Expert 也不同，导致 GPU 显存访问模式极度不规则，无法利用 Tensor Core 加速。这种“指针跳转”式的计算模式与现代 GPU 的 SIMD 架构格格不入。
- Status: Experimental. 具有极高的理论价值，但受限于硬件架构，目前仅作为理论验证模型存在。工程上可以通过训练时展开为具有最大固定深度的 Flat DynSIHA 解决，但目前计划优先级较低。

## 3. Core Mechanism: Dynamic Routing (MLP-based)

路由机制（Routing）是 DynSIHA 的大脑，它决定了信息如何在推断空间中流动。DynSIHA 的核心不在于特定的静态变换，而在于如何通过语义匹配动态地组装计算路径。

### 3.1 The Evolution of Decision: From Prototypes to Universal Approximation

我们经历了从“语义原型匹配”到“通用函数近似”的认知演进。

#### 3.1.1 CPR/CAPR (Prototype-based Routing) - Deprecated

- Mechanism: 基于余弦相似度 (CPR) 或 交叉注意力 (CAPR) 的原型匹配。
- Verdict: **已废弃**。实验证明，基于原型的路由具有强烈的“中心对称”归纳偏置，在处理 XOR 等非线性逻辑规则时表现极差。虽然在聚类任务中具有可解释性，但在 ARC 这种逻辑密集型任务中，其表达力不足成为了瓶颈。

#### 3.1.2 MLP Router (Current SOTA)

- Status: Current Production.
- Mechanism: 采用标准的 2-layer MLP (`Linear -> SiLU -> Linear`)。
- Insight: 放弃了显式的“原型”隐喻，回归通用函数近似。实验证明，MLP 路由在 XOR 逻辑、条件分支和复杂流形划分上具有压倒性的表达力。
- Philosophy: **“黑箱路由 + 几何压路机”**。我们不再强求路由过程本身的显式可解释性，而是依靠 **ARS2-Neo (AGA)** 优化器在黎曼流形上的平坦度约束，确保 MLP 路由不会陷入过拟合，并配合 **FARS** 进行认知代价塑造。

### 3.2 Computation Unit: The MLP Expert (formerly SPL)

所谓的 SPL (SparseProtoLinear) 实际上是受路由掩码控制的计算单元。它并非简单的线性层，而是一个高度非线性的 MLP Expert。它通过 Markov Blanket 的隐喻，将神经元的“计算”与“决策”在功能上解耦。

#### 3.2.1 Mathematical Definition

一个标准的计算单元被重构为：

1. `μ` (mu_weight): Computation Core. 代表“计算工具集”，本质上是一个两层 MLP (`Linear -> SiLU -> Linear`)。
2. `Routing Path`: 由 CPR 或 CAPR 生成的动态掩码 `mask`。

Aggregation
`y = MLP(x; μ) ⊙ mask`

### 3.3 Prototype Resident Connection (PRC)

在 Deep Network 中，随着层数加深，用于驱动路由的梯度信号会迅速衰减。为了解决这个问题，我们引入了 PRC。

- TRC (Traditional Residual Connection): `x_{l+1} = x_l + f(x_l)`。作用于 Block 之间，保护主任务梯度流。
- PRC (Prototype Resident Connection): 作用于 SPL 内部的 Routing Path。
  `pᵢₙₚᵤₜ⁽ᵗ⁾ = x⁽ᵗ⁾ + α ⋅ pᵢₙₚᵤₜ⁽ᵗ⁻¹⁾`
  PRC 确保了路由决策所依赖的输入表征（Prototype Input）是在上一层基础上的 增量修正 (Incremental Correction)，而非完全重构。这为 Router 提供了一个稳定的、分层演进的上下文环境。

## 4. Meta-Learning: Surprise-Aware Routing Shaping (SARS)

SARS 是驱动 DynSIHA 自组织的核心动力学。它不是一个显式的 Loss 函数，而是一种元学习机制。

### 4.1 Bayesian Inversion

SARS 的核心目标是优化模型的 Implicit Prior `p(z)`（即 Router 的内在偏好）。
根据贝叶斯定理：
`P(z|x) = P(x|z)P(z) / P(x)`
取对数并移项，我们得到优化的核心关系式：
`log P(z) ∝ log P(z|x) - log P(x|z)`

- Posterior `log P(z|x)`: 模型在看到数据 `x` 后，实际激活了哪些模块。由 `routing_logits` 表示。
- Likelihood `log P(x|z)`: 模块 `z` 解释数据 `x` 的能力。这是未知的，需要估计。
- Prior `log P(z)`: 我们希望优化的目标。

### 4.2 The Goodness Function (Proxy for Likelihood)

如何衡量“模块 `z` 解释数据 `x` 的能力”？
根据 FEP，好的解释意味着低的惊奇 (Low Surprise)，或者说 低的学习成本 (Low Complexity Cost)。

我们定义 Goodness 为 Likelihood 的工程代理：
`Goodness ≈ Importance × (Belief - Cost)`

具体实现公式：
`𝒢 = ‖ ∇ₒᵤₜ ℒ ‖ ⋅ ( Norm(logits) - ‖ ∇ₒᵤₜ ℒ ‖ )`

- `‖ ∇ᵧ ℒ ‖` (Parameter Gradient Norm): 代表 Cost。如果激活模块 `z` 导致参数梯度很大，说明该模块“不懂”这个数据，需要大幅修改参数才能拟合，因此 Likelihood 低。
- `Norm(logits)` (Belief): 代表 Posterior Belief。模型当前的信心。
- `‖ ∇ₒᵤₜ ℒ ‖` (Output Gradient Norm): 代表 Importance。该模块的输出对最终 Loss 有多大影响。

### 4.3 Shaping the Prior

通过最小化 Prior 的熵，我们驱动 Router 形成明确的偏好：
`ℒₘₑₜₐ = Entropy( Softmax( logits - 𝒢.detach() ) )`
这迫使 Router 将概率集中在那些 既重要 (High Importance) 又 熟练 (Low Cost) 的专家上。

## 5. Future Directions

### 5.1 FARS (Fisher-Aware Routing Shaping)

目前的 SARS 依赖于一阶梯度范数 (`‖ ∇ ℒ ‖`) 来衡量 Cost。然而，瞬时梯度包含大量噪声。
FARS 提议利用 Fisher Information Matrix (FIM) 的对角近似（即 Adam 的二阶矩 `vₜ`）来替代梯度范数。
`Cost_FARS ≈ √vₜ`
Fisher Information 衡量了参数空间的曲率，即“参数变化对分布的影响程度”。使用 FARS 可以提供更稳定、更符合信息几何本质的路由塑造信号。

### 5.2 Standardization

下一步的工程重点是完成 `MLPRouter` 在 `src/models/dynsiha/shared/layers.py` 中的标准化封装，并将其集成到 `FlatDynSIHA` 架构中。同时，确保 `ARCTrainer` 能够通过 `RoutingShaper` 完美驱动 FARS 塑造流程。

_Document End._
