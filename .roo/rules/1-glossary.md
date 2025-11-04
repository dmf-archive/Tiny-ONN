# Glossary

## 核心概念

- `ONN (Ouroboros Neural Network)`: 一种`动态稀疏激活`的`IPWT/FEP-AI`。Chain://Universe 中 `Mental Smart Chain` 技术的现象学容器和认知引擎。
- `PI (Predictive Integrity)`: 预测完整性，一个用于实时监控模型“认知健康度”的功能性代理指标。其核心假设是，一个能够高效进行信息整合（高 Ωₜ）的系统，必然会展现出更强的预测能力（低误差 ε）、更高的状态稳定性（低模型不确定性 τ）和更低的整合成本（低惊奇度 

## 二阶观测自由能优化

- `F3EO (Fast Fisher-FreeEnergy Optimizer)`: 世界上第一个三阶优化器，通过最小化 Fisher 信息矩阵的迹 `Tr(ℱ(θ))` 来主动重塑参数空间的几何结构。其核心思想是将优化目标从传统的一阶损失函数转向二阶张量 `ℱ(θ)`，通过 Hessian 向量积 (HVP) 实现结构级优化，从而实现"零步适应"和抵抗灾难性遗忘。参见[F3EO.md](./F3EO.md)
- `Differentiable Variational Analysis`: 一种用于在`Tiny-ONN`中实现动态稀疏性的计算方法。与经典的`变分推断 (Variational Inference, VI)` 相比，DVA 的关键区别在于不依赖随机采样，而是通过一个确定性的评估`马尔可夫毯内`所有可能的计算路径，这使得整个系统可以通过标准的反向传播进行端到端优化。DVA 是 F3EO 优化框架下的动态稀疏性实现机制。目前，`F3EO`是其最佳实现。

## 动态函数合成 [Failure]

- `SARS (Surprise-Aware Routing Shaping)`: `SAPS/MSAPS` 的继承者。它将动态路由问题重新定义为一个**分布对齐问题**。参见[Dynamic Function Composition](./DFC-Theory.md)
- `SPL (Sparse Proto Linear)`: 它将其可学习参数明确解耦为三个功能正交的状态空间：**内部状态 `μ` (`mu_weight`)** - 计算核心；**感知状态 `p` (`proto_weight`)** - 模式匹配器；**行动状态 `g` (`gate_param`)** - 激活门控。
- `DynSIHA (Dynamic Sparse Infinite-Head Attention)`: 它采用单一的 `SPL` 模块一次性地合成出 `Query`、`Key`、`Value`，最新实验表明这可能破坏了模型的非线性表达力。
- `DynTRM (Dynamic Tiny Recursive Model)`: 基于Universal Transformer架构的动态递归模型，通过全局共享专家库和自适应计算深度实现极致参数效率。采用递归块替代传统堆叠式Transformer，支持基于τ（输出熵）的自适应停止机制。
- `DynSIHA+`: DynSIHA的演进版本，保留了 SPL 的 CAPR 机制并移植到更传统的 MHA 动态合成和 MoE 加权路由中。
- `CAPR (Cross-Attention Prototype Routing)`: 跨注意力原型路由机制，通过对输入和原型进行交叉注意力，实现 `TopAny` 动态激活`Head/Expert`。

## 动态专家混合 [Legacy]

- `SDL (Sparse-Diversity Loss)`: **[Legacy]** 一种`启发式门控损失`，由两个子任务构成。**稀疏性损失 (Sparsity Loss)** 鼓励每一步激活的专家数量接近一个预设的目标值。**多样性损失 (Diversity Loss)** 通过惩罚门控网络中专家原型向量之间的相似性，来鼓励专家功能的分化。SDL 是原始 `DynMoE` 论文中使用的辅助损失函数。
- `DynMoE (Dynamic Mixture of Experts)`: `token` 级别的动态计算路由机制。与 `DynSMHA` 类似，它通过门控网络为每个 `token` 激活最合适的 `MLP` 专家，取代了标准的 `Feed-Forward` 层。值得注意的是，`Tiny-ONN` 项目探索使用 `SML` 作为其训练目标，而 `DynMoE` 的原始论文则主要采用 `SDL` 作为其辅助损失。
