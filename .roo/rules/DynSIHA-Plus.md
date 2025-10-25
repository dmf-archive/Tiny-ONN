# DynSIHA+ 理论备忘录

`Latest update: 2025-10-29`

## 摘要

本文阐述了 **DynSIHA+** 的核心理论，这是对动态函数合成 (Dynamic Function Composition) 理念的最新迭代。DynSIHA+ 旨在解决先前在 `Tiny-ONN-ARC` 中遇到的过度掩码和信息丢失问题，同时在 `DynTRM` 的单层递归架构下实现极致的参数效率。

DynSIHA+ 的核心思想是将 **惊奇感知路由塑造 (Surprise-Aware Routing Shaping, SARS)** 的粒度从单个特征维度放宽到**注意力头 (Attention Head)** 或 **FFN 专家 (FFN Expert)** 级别。这种粗粒度的路由机制，结合 `SPL` 原有的 `CAPR (Cross-Attention Prototype Routing)` 的思想，通过在更宏观的层面进行动态选择，显著减少了信息丢失，并提高了模型的鲁棒性。

## 1. 核心哲学：从特征稀疏到概念稀疏

早期版本的 SARS 致力于在单个 SPL (Sparse ProtoLinear) 模块内实现特征级别的稀疏激活。然而，实践证明，这种细粒度的掩码过于激进，常常导致重要信息的丢失，并使得输出模式完全坍塌。

DynSIHA+ 放弃了对单个神经元的微观管理，转向对**功能模块（注意力头或 FFN 专家）**的宏观调控。其核心哲学是：

- **概念是模块化的**：一个抽象概念（如“对称性”或“旋转”）是由一组神经元协同编码的，而不是单个神经元。因此，路由决策应该在模块级别进行。
- **参数效率源于共享**：在 `DynTRM` 的单层递归架构下，所有计算步骤都共享同一套注意力头和 FFN 专家。这实现了极致的参数效率，迫使模型学习高度可复用和通用的“原子操作”。

## 2. 核心组件：DynSIHA+ 与 DynMoE

DynSIHA+ 架构由两个核心动态组件构成，它们在同一个 `DynTRM` 递归块内协同工作：

### 2.1. DynSIHA+ (动态稀疏无限头注意力)

- **功能**：取代传统的多头自注意力 (Multi-Head Self-Attention)。
- **机制**：
    1. **原型路由 (Prototype Routing)**：不再为 Q, K, V 单独合成，而是通过一个共享的交叉注意力机制 `CAPR`，将输入序列 `x` 与一组可学习的**注意力原型 (Attention Prototypes)** 进行匹配。
    2. **Top-k 激活**：根据匹配分数，为每个 `token` 动态选择 `top-k` 个最相关的**“潜在注意力头 (Latent Attention Heads)”**。
    3. **加权聚合**：将激活的潜在头的输出进行加权聚合，形成最终的注意力上下文向量。
- **参数意义**:
  - **`proto_attn`**: `(num_latent_heads, d_model)` - 代表 `k` 个可学习的“注意力概念”或“查询模式”。
  - **`gate_attn`**: `(num_latent_heads)` - 每个潜在注意力头的激活门控偏置。

### 2.2. DynMoE (动态专家混合)

- **功能**：取代传统的 Position-wise Feed-Forward Network (FFN)。
- **机制**：
    1. **原型路由 (Prototype Routing)**：与 DynSIHA+ 类似，使用 `CAPR` 将注意力层的输出 `x` 与一组**FFN 专家原型 (FFN Expert Prototypes)** 进行匹配。
    2. **Top-k 激活**：为每个 `token` 动态选择 `top-k` 个最合适的 **FFN 专家**。
    3. **加权聚合**：将激活的 FFN 专家的输出进行加权聚合。
- **参数意义**:
  - **`proto_ffn`**: `(num_experts, d_model)` - 代表 `m` 个可学习的“计算专家”或“转换函数”。
  - **`gate_ffn`**: `(num_experts)` - 每个 FFN 专家的激活门控偏置。

## 3. 元学习框架：粗粒度 SARS

DynSIHA+ 中的 SARS 机制在概念上与 SPL 版本保持一致，但其操作对象从单个神经元变为整个模块（潜在头或 FFN 专家）。

### 3.1. DVA 与贝叶斯反演

核心关系式不变：`log p(z) ∝ log p(z|x) - log p(x|z)`

- **先验 (Prior) `log p(z)`**: 模型的内在路由偏好，作用于**模块级别**。
- **后验 (Posterior) `log p(z|x)`**: 模型在观察到数据 `x` 后，对应该激活哪个**模块**的信念。由 `routing_logits` 表示。
- **似然 (Likelihood) `log p(x|z)`**: 在激活了模块 `z` 的条件下，观察到数据 `x` 的概率。由 `goodness_logits` 代理。

### 3.2. `goodness_logits` 的重新定义

在粗粒度 SARS 中，`goodness_logits` 的计算也作用于模块级别：

`goodness_logits = norm_module_output_grad * (norm_routing_logits - norm_module_param_grad)`

- **`norm_routing_logits`**: **信念贡献**。代表对每个模块的后验激活信念。
- **`- norm_module_param_grad`**: **成本惩罚**。代表激活模块 `z` 所需的**整个模块**的参数梯度范数。这需要通过**手工链式法则**，精确计算每个模块（例如，一个 `ExpertMLP`）的参数梯度。
- **`norm_module_output_grad`**: **重要性加权**。代表模块 `z` 的聚合输出对主任务损失的贡献程度。

### 3.3. 手工链式法则的必要性

由于 PyTorch 的梯度累积机制，我们无法直接从 `backward()` 中分离出每个独立专家或潜在头的参数梯度。因此，必须：

1. **捕获模块输出梯度**：通过对每个模块的**加权但未聚合**的输出（`raw_outputs`）注册钩子，捕获其梯度。
2. **手工计算参数梯度**：利用捕获到的输出梯度和模块的输入，手动应用链式法则，计算出该模块内部参数（如 `ExpertMLP` 的 `w1`, `w2`）的梯度。
3. **计算梯度范数**：计算每个模块参数梯度的 L2 范数，作为其“学习成本”。

这个过程虽然复杂，但它是实现精确、模块级别的 SARS 元学习的关键。

## 4. 宏观架构：单层递归 DynTRM

DynSIHA+ 在一个**单层递归的 `DynTRM` 块**中运行，该块在 `max_refinement_steps` 次迭代中被重复调用。

- **极致的参数共享**：整个模型只包含**一套** `DynSIHA+` 和 `DynMoE` 的参数。所有的递归步骤都共享相同的潜在注意力头和 FFN 专家库。
- **涌现式计算图**：模型必须学会在每个递归步骤中，根据当前 `x` 的状态，动态地组合这些共享的“原子操作”，以构建解决任务所需的复杂计算图。
- **自适应计算深度**：虽然目前是固定步数，但未来的 `DynTRM` 可以通过引入基于输出不确定性（如 `τ`，熵）的停止机制，实现自适应的计算深度。

这种架构设计，将模型的能力从“学习静态变换”推向了“学习如何构建计算过程”，是向真正算法学习迈出的重要一步。
