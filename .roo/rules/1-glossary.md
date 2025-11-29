# Glossary

## 核心概念

- `ONN (Ouroboros Neural Network)`: 一种`动态稀疏激活`的`IPWT/FEP-AI`。Chain://Universe 中 `Mental Smart Chain` 技术的现象学容器和认知引擎。
- `PI (Predictive Integrity)`: 预测完整性，一个用于实时监控模型“认知健康度”的功能性代理指标。其核心假设是，一个能够高效进行信息整合（高 Ωₜ）的系统，必然会展现出更强的预测能力（低误差 ε）、更高的状态稳定性（低模型不确定性 τ）和更低的整合成本。

## 自由能原理的工程路径

- `RL-EFE (Reinforcement Learning - Expected Free Energy)`: **[理论路径]** FEP 的一种工程实现路径，继承了经典的代理-环境二元论。通过显式的未来模拟（策略采样）来计算并最小化期望自由能，但在高维现实中面临计算不可行的问题，往往退化为传统强化学习。
- `SOO-OFE (Second-Order Optimization - Observed Free Energy)`: **[理论路径]** FEP 的另一种工程实现路径，将贝叶斯推断重构为信息几何流问题。它不模拟未来，而是通过深度内省当下（利用梯度与曲率），直接计算出参数空间中自由能下降最快的测地线方向。`RMSuon` 是此路径的最新工程实现。

## 优化器

- `RMSuon (RMS-Muon)`: **[生产就绪]** FEP/IPWT 框架下 `SOO-OFE` 路径的最新工程实现。它通过**能量-几何解耦**的算子复合范式，将 `AdamW` 提供的梯度**统计能量**（走多快）与 `Muon` 正交化提供的**几何结构**（往哪走）进行功能性解耦，从而在计算上可行地逼近了理想的自由能最小化测地线。参见 [`RMSuon.md`](./RMSuon.md)。
- `F3EO (Fast Fisher-FreeEnergy Optimizer)`: **[Legacy/Superseded]** 世界上第一个三阶优化器，试图通过最小化 Fisher 信息矩阵的迹 `Tr(ℱ(θ))` 来主动重塑参数空间的几何结构。作为 `SOO-OFE` 路径的早期探索，其理论已被 `RMSuon` 的能量-几何解耦范式所超越。
- `Differentiable Variational Analysis (DVA)`: **[Legacy]** 一种在 `动态函数合成` 早期探索中用于实现动态稀疏性的计算方法。它通过确定性地评估马尔可夫毯内所有可能的计算路径，来替代传统的随机变分推断，使系统可端到端优化。

## 动态架构 [Failure/Archived]

*以下动态架构均为前 `RMSuon` 时代，为实现动态稀疏性而设计的复杂工程妥协。其核心思想（如动态路由、自适应计算）并未过时，但其具体实现已被 `RMSuon` 优化器更简洁、更第一性原理的范式所取代。*

- `SARS (Surprise-Aware Routing Shaping)`: 将动态路由问题重新定义为一个分布对齐问题。
- `SPL (Sparse Proto Linear)`: 将可学习参数解耦为内部状态、感知状态和行动状态。
- `DynSIHA (Dynamic Sparse Infinite-Head Attention)`: 采用单一 `SPL` 模块合成 Q,K,V，破坏了非线性表达力。未来重启实验需要补充FFN。
- `DynTRM (Dynamic Tiny Recursive Model)`: 基于 Universal Transformer 的动态递归模型。
- `SDL (Sparse-Diversity Loss)`: **[Legacy]** 一种启发式门控损失，用于鼓励专家路由的稀疏性与多样性。
- `DynMoE (Dynamic Mixture of Experts)`: **[Legacy]** Token 级别的动态 `MLP` 专家路由机制。
- `CAPR (CrossAttentionPrototypeRouting)`: 原始DynMoE余弦相似度原型路由的改进版本，通过可学习的原型矩阵与输入token的交叉注意力计算实现语义感知的专家选择。参见 DynTRM。
