# Tiny-ONN 术语表

- **ONN (Ouroboros Neural Network)**: 一种`动态稀疏激活`的`IPWT/FEP-AI`。
- **PI (Predictive Integrity)**: 预测完整性，一个用于实时监控模型“认知健康度”的功能性代理指标。其核心假设是，一个能够高效进行信息整合（高 Ωₜ）的系统，必然会展现出更强的预测能力（低误差 ε）、更高的状态稳定性（低模型不确定性 τ）和更低的整合成本（低惊奇度 Surprise）。
- **EAVI (Excursion-Alignment Variational Inference)**: 一种曾被探索用于解决 Teacher Forcing 暴露偏差的`对齐微调`范式。其核心思想是通过对模型独立生成的完整序列进行全局对齐来提供更强的学习信号。后因其无法从根本上解决暴露偏差问题，且增加了训练流程的复杂性而被**弃用**。

## 机制：辅助损失和元学习

- **SMK (Surprise Min_K)**: 一种基于“惊奇度”的`选择性梯度更新策略`。在 MoE 架构中，当多个专家被激活并计算梯度后，SMK 策略仅保留梯度范数（Surprise）最小的 `min_k` 个专家的梯度用于参数更新，其余专家的梯度被置零。SMK 已经被弃用，但其思想启发了后续机制。
- **SML (Surprise Minimization Loss)**: 一种早期的`元学习门控损失`，旨在引导网络自组织地选择计算效率最高的神经元路径。其核心思想是，高效的路由应将信息分配给能以最低“系统扰动”（即“惊奇度”）处理它的专家。“惊奇度” `Sᵢ` 被定义为任务损失 `L_main` 相对于 SBL 模块某个内部激活张量 `x` 的梯度范数：`Sᵢ = ||∇_{xᵢ} L_main||₂`。该机制因其依赖瞬时、充满噪声的梯度信号而被证明不稳定，后被 SAPS 所取代。
- **SAPS (Surprise-Aware Prototype Shaping)**: `SMK` 和 `SML` 思想的最终进化形态，是驱动 `SPL` 自组织的核心学习动力学。它不再最小化瞬时的梯度，而是将梯度场 `S_μ = ∇_μ L_main` 视为一个更稳定的**扰动地形图**。其核心机制是通过**对比学习**来优化路由：将表现“好”（低惊奇度）的原型拉向当前输入的语义方向，同时将表现“坏”（高惊奇度）的原型推离。
- **SDL (Sparse-Diversity Loss)**: 一种`启发式门控损失`，由两个子任务构成。**稀疏性损失 (Sparsity Loss)** 鼓励每一步激活的专家数量接近一个预设的目标值。**多样性损失 (Diversity Loss)** 通过惩罚门控网络中专家原型向量之间的相似性，来鼓励专家功能的分化。SDL 是原始 `DynMoE` 论文中使用的辅助损失函数。

## 机制：动态函数合成

- **SBL (Sparse Bayesian Linear)**: `MoIE` 和 `DynSIHA` 的早期核心构件。它通过一种内容感知的结构化 Dropout 机制，在一个稠密的权重矩阵中实现动态、稀疏的神经元激活。
- **SPL (Sparse Proto Linear)**: SBL 的简化与最终重构。它将其可学习参数明确解耦为计算核心 (`mu_weight`)、模式匹配器 (`proto_weight`) 和激活门控 (`gate_param`)。其最终范式采用**纯粹化的余弦相似度路由**，通过在路由前对输入和原型进行 L2 归一化，彻底消除了“范数污染”，实现了纯粹的内容寻址。
- **MoIE (Mixture of Infinite Experts)**: 一种将稠密权重矩阵视为**连续专家空间**的`动态稀疏`范式。它利用 `SPL` 层为每个输入动态地采样出一个临时的、专用的稀疏子网络，并由 `SAPS` 引导进行自组织学习。
- **DynSIHA (Dynamic Sparse Infinite-Head Attention)**: `MoIE` 范式在注意力机制中的应用。它将标准注意力中固定的 `Q, K, V` 投影矩阵替换为 `SPL` 层，从而为每个 `token` 动态地、内容感知地“合成”出专用的投影子网络，旨在实现一种可编程的、表达力远超传统多头注意力的机制。

## 机制：动态专家混合 [Deprecated]

- **DynSMHA (Dynamic Sparse Multi-Head Attention)**: `token` 级别的动态稀疏注意力机制。通过门控网络为每个 `token` 动态选择并激活最合适的“注意力头专家”，取代了标准的 `Multi-Head Attention`。
- **DynMoE (Dynamic Mixture of Experts)**: `token` 级别的动态计算路由机制。与 `DynSMHA` 类似，它通过门控网络为每个 `token` 激活最合适的 `MLP` 专家，取代了标准的 `Feed-Forward` 层。值得注意的是，`Tiny-ONN` 项目探索使用 `SML` 作为其训练目标，而 `DynMoE` 的原始论文则主要采用 `SDL` 作为其辅助损失。

## 机制：分块注意力 [Deprecated]

- **NSA (Native Sparse Attention)**: 一种高效的稀疏注意力`forward`优化方法。它通过**全局压缩**、**分块 Top-N 选择**和**滑动窗口**三种策略的固定组合，来逼近全注意力的性能，同时显著降低计算和内存开销。遗憾的是，它和 DynSIHA 兼容性不佳。
- **DynNSIHA (Dynamic Native Sparse Infinite-Head Attention)**: 一种已被**废弃**的实验性注意力机制。其构想是在 `DynSIHA` 的神经元级动态稀疏之上，再叠加 `NSA` 的硬件友好型块级稀疏。实验证明，这种“双重稀疏”存在理论冲突，会导致模型学习能力崩溃，因此该技术路线已被放弃。此实验也暗示了`DynSIHA`和`SWA`的潜在不兼容，需要更多测试。
