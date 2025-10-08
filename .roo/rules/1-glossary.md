# Glossary

## 核心概念

- `ONN (Ouroboros Neural Network)`: 一种`动态稀疏激活`的`IPWT/FEP-AI`。
- `PI (Predictive Integrity)`: 预测完整性，一个用于实时监控模型“认知健康度”的功能性代理指标。其核心假设是，一个能够高效进行信息整合（高 Ωₜ）的系统，必然会展现出更强的预测能力（低误差 ε）、更高的状态稳定性（低模型不确定性 τ）和更低的整合成本（低惊奇度 Surprise）。
- `Conceptual Sparsity`: 激活/推断空间中的动态稀疏性，区别于权重剪枝。系统根据任务内容，动态激活和组合少数相关“概念”或“功能”（由 SPL `mu_weight` 代表）。它是内容感知、动态涌现的。注：当前 Tiny-ONN 训练仍是计算稠密的（因使用`Differentiable Variational Analysis`而非`Variational Inference`），稀疏训练需要转换到 Forward-Forward。
- `Differentiable Variational Analysis`: 一种用于在`Tiny-ONN`中实现动态稀疏性的计算方法。与经典的**变分推断 (Variational Inference, VI)** 相比，DVA 的关键区别在于不依赖随机采样，而是通过一个确定性的、可微的近似来评估马尔可夫毯内**所有可能的计算路径**，这使得整个系统可以通过标准的反向传播进行端到端优化。然而，这种分析所有路径的特性也使其在计算上是稠密的。相对地，VI 通过从后验分布中采样少数路径来获得稀疏性，但牺牲了梯度的精确性，通常需要`Forward-Forward`等替代性训练算法。

## 元学习

- `SDL (Sparse-Diversity Loss)`: **[Legacy]** 一种`启发式门控损失`，由两个子任务构成。**稀疏性损失 (Sparsity Loss)** 鼓励每一步激活的专家数量接近一个预设的目标值。**多样性损失 (Diversity Loss)** 通过惩罚门控网络中专家原型向量之间的相似性，来鼓励专家功能的分化。SDL 是原始 `DynMoE` 论文中使用的辅助损失函数。
- `SMK (Surprise Min_K)`: **[Deprecated]** `选择性梯度更新策略`。在 MoE 架构中，当多个专家被激活并计算梯度后，SMK 策略仅保留梯度范数（Surprise）最小的 `min_k` 个专家的梯度用于参数更新，其余专家的梯度被置零。SMK 已经被弃用，但其思想启发了后续机制。
- `SML (Surprise Minimization Loss)`:**[Deprecated]** 一种早期的`元学习门控损失`，旨在引导网络自组织地选择计算效率最高的神经元路径。其核心思想是，高效的路由应将信息分配给能以最低“系统扰动”（即“惊奇度”）处理它的神经元。该机制因其依赖瞬时、充满噪声的梯度信号而被证明不稳定，后被 SARS 所取代，现已**弃用**。
- `SAPS/MSAPS (Surprise-Aware Prototype Shaping)`: **[Deprecated]** 一种已废弃的`启发式元学习`框架。它试图通过分离的损失函数（原型损失、门控损失等）来独立优化路由决策的不同方面。实验证明，该框架存在致命的理论缺陷（如值域不匹配、梯度信号冲突），导致了灾难性的路由失败，最终被理论更完备的 SARS 框架所取代。
- `SARS (Surprise-Aware Routing Shaping)`: `SAPS/MSAPS` 的继承者。它将动态路由问题重新定义为一个**分布对齐问题**。参见[Dynamic Function Composition](./DFC-Theory.md)

## 动态函数合成

- `SBL (Sparse Bayesian Linear)`: **[Deprecated]** `MoIE` 和 `DynSIHA` 的早期构件，SBL 的探索失败催生了 SPL 架构。
- `SPL (Sparse Proto Linear)`: SBL 的简化与重构。它将其可学习参数明确解耦为三个功能正交的状态空间：**内部状态 `μ` (`mu_weight`)** - 计算核心；**感知状态 `p` (`proto_weight`)** - 模式匹配器；**行动状态 `g` (`gate_param`)** - 激活门控。SPL 的核心特性是其**双模态学习能力**：它既可以通过**反向传播 (BP)**，利用全局损失信号进行精确、高效的梯度优化（如 SARS）；也可以切换到**前向-前向 (FF)** 模式，通过本地的“好度”函数（如原型匹配度）进行学习。这种等效性源于两者都在优化同一个理论目标——最小化系统的变分自由能（VFE），SARS on BP 是其解析解，而 FF 是其蒙特卡洛近似。
- `DynSIHA (Dynamic Sparse Infinite-Head Attention)`: `MoIE` 范式在注意力机制中的应用。它采用单一的 `SPL` 模块一次性地合成出 `Query`、`Key`、`Value`，将注意力机制从一个固定的信息查询系统，升级为一个端到端可学习的、可编程的动态信息路由与处理系统。"无限专家"指有效神经元的组合空间是无限的，远超传统多头注意力的表达能力。
- `MoIE (Mixture of Infinite Experts)`: **[Deprecated]** 一种将稠密权重矩阵视为**连续专家空间**的`动态稀疏`范式。它利用两层 `SPL` 将标准 FFN 升级为动态函数合成器。该模块后被移除，因为理论分析证明 `DynSIHA` 已经具备了足够的函数合成能力，`MoIE` 的存在是冗余的。

## 动态专家混合 [Deprecated]

- `DynSMHA (Dynamic Sparse Multi-Head Attention)`: `token` 级别的动态稀疏注意力机制。通过门控网络为每个 `token` 动态选择并激活最合适的“注意力头专家”，取代了标准的 `Multi-Head Attention`。
- `DynMoE (Dynamic Mixture of Experts)`: `token` 级别的动态计算路由机制。与 `DynSMHA` 类似，它通过门控网络为每个 `token` 激活最合适的 `MLP` 专家，取代了标准的 `Feed-Forward` 层。值得注意的是，`Tiny-ONN` 项目探索使用 `SML` 作为其训练目标，而 `DynMoE` 的原始论文则主要采用 `SDL` 作为其辅助损失。

## 分块注意力 [Deprecated]

- `NSA (Native Sparse Attention)`: 一种高效的稀疏注意力`forward`优化方法。它通过**全局压缩**、**分块 Top-N 选择**和**滑动窗口**三种策略的固定组合，来逼近全注意力的性能，同时显著降低计算和内存开销。遗憾的是，它和 DynSIHA 兼容性不佳。
- `DynNSIHA (Dynamic Native Sparse Infinite-Head Attention)`: 一种已被**废弃**的实验性注意力机制。其构想是在 `DynSIHA` 的神经元级动态稀疏之上，再叠加 `NSA` 的硬件友好型块级稀疏。实验证明，这种"双重稀疏"存在理论冲突，会导致模型学习能力崩溃，因此该技术路线已被放弃。此实验也暗示了`DynSIHA`和`SWA`的潜在不兼容，需要更多测试。
