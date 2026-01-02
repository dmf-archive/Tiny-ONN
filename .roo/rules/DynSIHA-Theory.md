# DynSIHA: Dynamic Sparse Infinite-Head Attention

`Latest update: 2026-01-02`
`Taxonomy: v2.0`

> 生命的本质，是用内在的确定性，
> 对抗外部的不确定性。
> 将不确定性留给输入，
> 以不变应万变，
> 让动态替代先验，
> 在稠密计算中，
> 凝聚概念的结晶。
>
> —— 林睿，于 2025 年 10 月

## 0. Abstract

本文档旨在对 **DynSIHA** (Dynamic Sparse Infinite-Head Attention) 及其核心理念 **DFC** (Dynamic Function Composition) 进行形式化梳理与重构。

DynSIHA 并非单一的神经网络架构，而是一套旨在将 Transformer 从“静态特征变换器”升级为“自适应动态计算系统”的演进方法论。其核心愿景是通过 **SARS** (Surprise-Aware Routing Shaping) 机制，驱动网络在训练过程中自组织出针对特定输入的稀疏计算路径。

本文将 DynSIHA 的演进形态正式划分为三个阶段：**Dimension DynSIHA** (特征级掩码)、**Flat DynSIHA** (模块级组装) 与 **Recursive DynSIHA** (全局递归再入)。这三种形态代表了我们在 Parameter Efficiency、Routing Granularity 和 Computational Irregularity 之间权衡的探索轨迹。

## 1. Philosophy: Computational Density for Inference Sparsity

`Tiny-ONN` 项目的理论基石是 **IPWT** (Integrated Predictive Workspace Theory)，其核心目标是构建一个高效的变分推断引擎，最小化系统的 **VFE** (Variational Free Energy)。

这一理论目标决定了 DynSIHA 独特的设计哲学：**以计算的稠密性 (Computational Density)，换取推断的稀疏性 (Inference Sparsity)。**

### 1.1 The Paradox of Sparsity

在深度学习中，稀疏性通常被视为提高推理速度的手段（如 MoE 的 conditional computation）。然而，在 DynSIHA 中，稀疏性首先是认知 (Cognitive) 层面的需求，其次才是计算层面的需求。

- **Inference Sparsity** (Activation Sparsity): 模型在处理特定输入时，应当只激活其推断空间（Inference Space）中极少部分的“概念通路”。这种稀疏性是模型具备可解释性、泛化能力和逻辑推理能力的前提。
- **Computational Density**: 为了让这种稀疏结构能够通过 SGD 自发涌现，我们必须在训练阶段维持计算的稠密性。我们放弃了 Forward-Forward 等蒙特卡洛近似方法，坚持使用解析解（Analytical Solution）来计算梯度。因为对于 ARC 这种高精度、小样本任务，只有稠密的物理计算才能提供足够精确的梯度信号，驱动 VFE 的稳定下降。

### 1.2 Dynamic Function Composition (DFC)

传统的神经网络层是静态的：`y = f(x; θ)`，其中 `f` 的结构在推理时是固定的。
DFC 试图实现的是：`y = fₓ(x; θ)`，即函数 `f` 本身是输入 `x` 的函数。

DynSIHA 通过 **SPL** (SparseProtoLinear) 组件实现了这一愿景：它不再直接学习变换矩阵，而是学习一套“计算基元”和“组装规则”，为每个 Token 动态合成其所需的计算图。

## 2. Taxonomy: The Three Forms of DynSIHA

基于路由粒度（Granularity）和资源组织方式（Resource Organization），我们将 DynSIHA 的演进定义为以下三种形态。

### 2.1 Dimension DynSIHA

这是 DFC 理念的初次尝试，其核心特征是 Feature-level Masking。

- Mechanism: 在这个阶段，SPL 模块内的每一个神经元（输出维度）都被视为一个独立的“微型专家”。路由机制直接作用于特征维度，决定是否激活该神经元。
- Pros: 理论上拥有最高的表达自由度，能够雕刻出极其精细的稀疏流形。
- Cons:
  - Over-masking: 细粒度的掩码极易导致信息流的彻底阻断（Collapse），尤其是在网络深层。
  - Gradient Conflict: 稀疏性目标与主任务目标在微观层面频繁冲突，导致优化震荡。
- Status: Deprecated. 作为理论原型，已不再用于实际工程。

### 2.2 Flat DynSIHA

这是目前最稳定、工程可行性最高的形态。它保留了标准 Transformer 的 **Block Stacking** 结构，但在 Block 内部引入了 **Module-level Assembly**。

- **Structure**: 模型由 `L` 个独立的 Block 堆叠而成。每个 Block 拥有自己独立的参数空间（Head Repo, MLP Repo）。
- **Routing Logic**:
  - **Local Routing**: 路由决策仅在当前 Block 内部有效。
  - **Weighted Assembly**: 类似于 Soft-MoE，它根据路由权重对 Head 或 MLP 的输出进行加权组合。
  - **Independence**: Block `i` 的 Head 不能被 Block `j` 复用。
- **Pros**:
  - **Stability**: 梯度流稳定，易于训练。
  - **Modularity**: 符合神经科学中“功能柱” (Cortical Columns) 的组织方式。
- **Status**: **Current Baseline**. 是目前 ARC 任务的主力架构。

### 2.3 Recursive DynSIHA

> _Previously known as: DynSIHA+, DynTRM_

这是 DFC 理念的终极形态，旨在实现 **Global Dynamic Assembly** 和 **Dynamic Program Generation**。

- **Structure**: **Single Recursive Block**。整个网络在物理上只有一个 Block，但在时间上展开为 `T` 个 Refinement Steps。
- **Resource Organization**:
  - **Global Repo**: 所有的 Attention Heads 和 MLP Experts 存储在一个全局共享的 Repository 中。
  - **Universal Routing**: 无论是在第 1 步还是第 `T` 步，模型都从同一个 Repo 中抓取计算基元。
- **Mechanism**:
  - 模型不再是“经过层层变换”，而是“在每一步递归中，根据当前状态，从工具箱中挑选工具进行加工”。
  - 这实际上是在通过神经网络模拟 **图灵机** 的操作：Repo 是指令集，State 是纸带，Routing 是控制单元。
- **Cons (The "Performance Disaster")**:
  - **Memory Irregularity**: 由于每一步调用的 Expert 不同，且不同 Sample 调用的 Expert 也不同，导致 GPU 显存访问模式极度不规则，无法利用 Tensor Core 加速。工程上可以通过训练时展开为具有最大固定深度的 Flat DynSIHA 解决，暂未来得及实施。
- **Status**: **Experimental**. 具有极高的理论价值。

## 3. Core Mechanism: SparseProtoLinear (SPL)

SPL 是实现 DFC 的原子组件。它通过 **Markov Blanket** 的隐喻，将神经元的“计算”与“决策”在功能上解耦。

### 3.1 Mathematical Definition

一个标准的 Linear Layer 定义为 `y = σ(xW + b)`。
SPL 将其重构为三个正交的状态空间：

1. **`μ` (mu_weight)**: **Computation Core**. 代表“计算工具集”。
2. **`p` (proto_weight)**: **Pattern Matcher**. 代表“感知器官”，负责将输入 `x` 与原型进行匹配。
3. **`g` (gate_param)**: **Action Policy**. 代表“门控策略”。

前向传播过程分解为两条路径：

**Path 1: Computation (Transformation)**
`comp = Linear(SiLU(x), μ)`
_Note: 引入 SiLU 使得每个“神经元”实际上是一个两层 MLP，增强了单个基元的非线性变换能力。_

**Path 2: Routing (Decision)**
`z = Routing(x, p, g)`
`mask = ReLU(z) OR Softmax(z)`

**Aggregation**
`y = comp ⊙ mask`

### 3.2 Prototype Resident Connection (PRC)

在 Deep Network 中，随着层数加深，用于驱动路由的梯度信号会迅速衰减。为了解决这个问题，我们引入了 **PRC**。

- **TRC (Traditional Residual Connection)**: `x_{l+1} = x_l + f(x_l)`。作用于 Block 之间，保护主任务梯度流。
- **PRC (Prototype Resident Connection)**: 作用于 SPL 内部的 Routing Path。
  `pᵢₙₚᵤₜ⁽ᵗ⁾ = x⁽ᵗ⁾ + α ⋅ pᵢₙₚᵤₜ⁽ᵗ⁻¹⁾`
  PRC 确保了路由决策所依赖的输入表征（Prototype Input）是在上一层基础上的 **增量修正 (Incremental Correction)**，而非完全重构。这为 Router 提供了一个稳定的、分层演进的上下文环境。

## 4. Meta-Learning: Surprise-Aware Routing Shaping (SARS)

SARS 是驱动 DynSIHA 自组织的核心动力学。它不是一个显式的 Loss 函数，而是一种基于 **DVA (Differentiable Variational Analysis)** 的元学习机制。

### 4.1 Bayesian Inversion

SARS 的核心目标是优化模型的 **Implicit Prior** `p(z)`（即 Router 的内在偏好）。
根据贝叶斯定理：
`P(z|x) = P(x|z)P(z) / P(x)`
取对数并移项，我们得到优化的核心关系式：
`log P(z) ∝ log P(z|x) - log P(x|z)`

- **Posterior `log P(z|x)`**: 模型在看到数据 `x` 后，实际激活了哪些模块。由 `routing_logits` 表示。
- **Likelihood `log P(x|z)`**: 模块 `z` 解释数据 `x` 的能力。这是未知的，需要估计。
- **Prior `log P(z)`**: 我们希望优化的目标。

### 4.2 The Goodness Function (Proxy for Likelihood)

如何衡量“模块 `z` 解释数据 `x` 的能力”？
根据 FEP，**好的解释意味着低的惊奇 (Low Surprise)**，或者说 **低的学习成本 (Low Complexity Cost)**。

我们定义 **Goodness** 为 Likelihood 的工程代理：
`Goodness ≈ Importance × (Belief - Cost)`

具体实现公式：
`𝒢 = ‖ ∇ₒᵤₜ ℒ ‖ ⋅ ( Norm(logits) - ‖ ∇ᵧ ℒ ‖ )`

- **`‖ ∇ᵧ ℒ ‖` (Parameter Gradient Norm)**: 代表 **Cost**。如果激活模块 `z` 导致参数梯度很大，说明该模块“不懂”这个数据，需要大幅修改参数才能拟合，因此 Likelihood 低。
- **`Norm(logits)` (Belief)**: 代表 **Posterior Belief**。模型当前的信心。
- **`‖ ∇ₒᵤₜ ℒ ‖` (Output Gradient Norm)**: 代表 **Importance**。该模块的输出对最终 Loss 有多大影响。

### 4.3 Shaping the Prior

通过最小化 Prior 的熵，我们驱动 Router 形成明确的偏好：
`ℒₘₑₜₐ = Entropy( Softmax( logits - 𝒢.detach() ) )`
这迫使 Router 将概率集中在那些 **既重要 (High Importance)** 又 **熟练 (Low Cost)** 的专家上。

## 5. Routing Evolution: From CPR to CAPR

路由机制（即如何计算 `z`）是 DynSIHA 的大脑。我们经历了从简单的相似度匹配到复杂的注意力机制的演进。

### 5.1 CPR (Cosine Prototype Routing)

- **Origin**: DynMoE / Dimension DynSIHA（早期）.

- **Mechanism**: 计算输入 `x` 与原型 `p` 的余弦相似度，配合动态阈值。
- **Verdict**: 奠定了 **Semantic Routing** 的基础（即路由基于内容而非位置），但过于线性，余弦相似度归一化丢弃了大量非线性信息。

### 5.2 CAPR (Cross-Attention Prototype Routing)

- **Origin**: Recursive DynSIHA / DynTRM.

- **Status**: **Current SOTA**.
- **Mechanism**: 将路由过程建模为 **Scaled Dot-Product Cross-Attention**。
  - **Query**: 输入 `x`。
  - **Key/Value**: 可学习的原型 `P_k, P_v`。
  - **Output**: `z = Softmax(x Pₖᵀ / √d) Pᵥ`。
- **Insight**: CAPR 实际上引入了一个 **Gated Transformer** 结构。它不仅做选择，还对路由信号进行了阈值稀疏化。这与 **Qwen3-Next** 等前沿模型中出现的 Gated Attention 思想不谋而合。CAPR 提供了最强的表达能力，使得 Router 能够理解复杂的上下文关系。

## 6. Future Directions

### 6.1 FARS (Fisher-Aware Routing Shaping)

目前的 SARS 依赖于一阶梯度范数 (`‖ ∇ ℒ ‖`) 来衡量 Cost。然而，瞬时梯度包含大量噪声。
**FARS** 提议利用 **Fisher Information Matrix (FIM)** 的对角近似（即 Adam 的二阶矩 `vₜ`）来替代梯度范数。
`Cost_FARS ≈ √vₜ`
Fisher Information 衡量了参数空间的曲率，即“参数变化对分布的影响程度”。使用 FARS 可以提供更稳定、更符合信息几何本质的路由塑造信号。

### 6.2 Standardization

下一步的工程重点是将 **CAPR** 从复杂的递归架构中剥离出来，封装为标准的 `nn.Module`，并将其集成到 **Flat DynSIHA** (Baseline) 中，结合 **RMSuon** 优化器，在 ARC-AGI 任务上进行大规模验证。

_Document End._
