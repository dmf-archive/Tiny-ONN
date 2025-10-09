# Glossary

## 核心概念

- `ONN (Ouroboros Neural Network)`: 一种`动态稀疏激活`的`IPWT/FEP-AI`。Chain://Universe 中 `Mental Smart Chain` 技术的现象学容器和认知引擎。
- `PI (Predictive Integrity)`: 预测完整性，一个用于实时监控模型“认知健康度”的功能性代理指标。其核心假设是，一个能够高效进行信息整合（高 Ωₜ）的系统，必然会展现出更强的预测能力（低误差 ε）、更高的状态稳定性（低模型不确定性 τ）和更低的整合成本（低惊奇度 Surprise）。
- `Dynamic Conceptual/Activation Sparsity`: 推断/激活空间中的动态稀疏性，区别于权重剪枝。系统根据任务内容，动态激活和组合少数相关“概念”或“功能”（由 SPL Neuron）。它是内容感知、动态涌现的。
- `Differentiable Variational Analysis`: 一种用于在`Tiny-ONN`中实现动态稀疏性的计算方法。与经典的`变分推断 (Variational Inference, VI)` 相比，DVA 的关键区别在于不依赖随机采样，而是通过一个确定性的、可微的近似来评估`马尔可夫毯内`所有可能的计算路径，这使得整个系统可以通过标准的反向传播进行端到端优化。然而，这种分析所有路径的特性也使其在计算上是稠密的。相对地，VI 通过采样少数路径来控制开销并探索不可解后验分布，但牺牲了梯度的精确性。

## 动态函数合成

- `SARS (Surprise-Aware Routing Shaping)`: `SAPS/MSAPS` 的继承者。它将动态路由问题重新定义为一个**分布对齐问题**。参见[Dynamic Function Composition](./DFC-Theory.md)
- `SPL (Sparse Proto Linear)`: 它将其可学习参数明确解耦为三个功能正交的状态空间：**内部状态 `μ` (`mu_weight`)** - 计算核心；**感知状态 `p` (`proto_weight`)** - 模式匹配器；**行动状态 `g` (`gate_param`)** - 激活门控。SPL 的核心特性是其**双模态学习能力**：它既可以通过**反向传播 (BP)**，利用全局损失信号进行精确、高效的梯度优化（如 SARS）；也可以切换到**前向-前向 (FF)** 模式，通过本地的“好度”函数（如原型匹配度）进行学习。这种等效性源于两者都在优化同一个理论目标——最小化系统的变分自由能（VFE），SARS on BP 是其解析解，而 FF 是其蒙特卡洛近似。
- `DynSIHA (Dynamic Sparse Infinite-Head Attention)`: 它采用单一的 `SPL` 模块一次性地合成出 `Query`、`Key`、`Value`，将注意力机制从一个固定的信息查询系统，升级为一个端到端可学习的、可编程的动态信息路由与处理系统。"无限专家"指有效神经元的组合空间是无限的，远超传统多头注意力的表达能力。

## 动态专家混合 [Legacy]

- `SDL (Sparse-Diversity Loss)`: **[Legacy]** 一种`启发式门控损失`，由两个子任务构成。**稀疏性损失 (Sparsity Loss)** 鼓励每一步激活的专家数量接近一个预设的目标值。**多样性损失 (Diversity Loss)** 通过惩罚门控网络中专家原型向量之间的相似性，来鼓励专家功能的分化。SDL 是原始 `DynMoE` 论文中使用的辅助损失函数。
- `DynMoE (Dynamic Mixture of Experts)`: `token` 级别的动态计算路由机制。与 `DynSMHA` 类似，它通过门控网络为每个 `token` 激活最合适的 `MLP` 专家，取代了标准的 `Feed-Forward` 层。值得注意的是，`Tiny-ONN` 项目探索使用 `SML` 作为其训练目标，而 `DynMoE` 的原始论文则主要采用 `SDL` 作为其辅助损失。
