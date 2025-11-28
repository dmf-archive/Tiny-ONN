---
title: "ADR-0010: 采用动态 MLP-Attention 与 MoE 的下一代 ARC 求解器架构"
status: "Proposed"
date: "2025-11-29"
authors: "Tiny-ONN 课题组"
tags: ["architecture", "decision", "transformer", "moe", "attention"]
supersedes: ""
superseded_by: ""
---

# ADR-0010: 采用动态 MLP-Attention 与 MoE 的下一代 ARC 求解器架构

## 状态 (Status)

**Proposed** | Accepted | Rejected | Superseded | Deprecated

## 背景 (Context)

当前的 `exp/arc_rmsuon` 实验通过极端 Dropout (`0.9`) 强迫 Qwen3 模型学习鲁棒的全息表征，在 ARC 任务上已初步展现出学习有效逻辑规则的迹象。然而，这种“纯噪声”式的正则化与生物神经网络的语义路由机制相悖，且单一的密集模型（Dense Model）在处理 ARC 任务固有的组合复杂性和模态多样性时，可能存在表达能力瓶颈。为了突破这一瓶颈，需要设计一种既能捕捉高阶关系、又能动态分配计算资源的下一代架构。

## 决策 (Decision)

我们决定设计并实现一个名为 `exp/arc_dyn_mlp` 的新实验架构，该架构融合了非线性注意力和动态专家混合网络，并由 `RMSuon` 优化器驱动。具体规格如下：

1. **基座 (Backbone)**: 严格继承 **Qwen3** 的现代 Transformer 组件，包括 RMSNorm, SwiGLU 激活函数和 RoPE 旋转位置编码。
2. **注意力 (Attention)**: 采用 **MLP-Based Attention**。使用 `Linear -> Activation -> Linear` 结构的 MLP 替代标准的 `nn.Linear` 来生成 Q, K, V 向量，以增强捕捉非线性关系的能力，同时保持与 SDPA 的兼容性。
3. **前馈网络 (FFN)**: 实现 **DynMoE (动态专家混合)**。在 FFN 层引入 Top-Any Gating 路由，并采用 SDL (Sparse Diversity Loss) 作为专家分化的正则化损失。
4. **优化器 (Optimizer)**: 使用 **RMSuon**，并利用其参数分组功能。对 2D 权重矩阵应用 Muon 通道，对 1D 偏置、范数和 Gating 参数自动回退到 AdamW。
5. **活性管理 (Liveness Management)**: 为 DynMoE 引入基于 Epoch 路由统计的**专家生命周期管理**机制。在每个 Epoch 结束后，重置从未被激活的“死亡专家”，初始化策略可选择随机或根据未被充分服务的 Token 特征进行引导。
6. **超参数 (Hyperparameters)**: 鉴于 MoE 引入的稀疏性，将 **Dropout 率大幅下调**（例如至 `0.1`），并可能**适度减小 Hidden Size** 以平衡显存消耗。

## 后果 (Consequences)

### 积极 (Positive)

- **POS-001**: **增强表达能力 (Enhanced Expressiveness)**: MLP-Attention 能学习非线性关系，理论上更适合 ARC 任务中复杂的抽象规则推理。
- **POS-002**: **自适应计算 (Adaptive Computation)**: DynMoE 允许模型根据输入样本的复杂度动态分配算力。
- **POS-003**: **优化稳定性 (Optimizer Stability)**: RMSuon 的智能分组可为不同类型的参数提供最适合的优化策略，避免了 Gating 网络因不当优化而产生的路由震荡。
- **POS-004**: **长期活性 (Sustained Liveness)**: 基于统计的专家生命周期管理机制能有效防止路由坍塌，确保专家库的长期多样性和利用率。

### 消极 (Negative)

- **NEG-001**: **架构复杂性增加 (Increased Complexity)**: 相比标准 Transformer，该架构集成了多种高级组件，增加了实现、调试和维护的复杂度。
- **NEG-002**: **显存压力 (Memory Pressure)**: MoE 和 MLP-Attention 均会增加模型参数量，即使减小 Hidden Size，显存消耗仍可能显著高于基线模型。
- **NEG-003**: **训练动态不稳定 (Training Dynamics Instability)**: 专家动态重置可能导致训练曲线出现短暂的 Loss 尖峰。

## 考虑的备选方案 (Alternatives Considered)

### 方案 1: 保留 DynSIHA

- **ALT-001**: **描述 (Description)**: 在注意力层实现 `DynSIHA`，即每个 Head 都是一个由原型（Proto）路由的 MoE。
- **ALT-002**: **拒绝理由 (Rejection Reason)**: MLP-Attention 本身已是巨大的表达力改进，引入 DynSIHA 会使架构过于复杂。在证明 MLP-Attention 不足之前，应遵循“如无必要，勿增实体”的原则。

### 方案 2: 仅使用标准 Dropout 正则化

- **ALT-003**: **描述 (Description)**: 仅使用 MLP-Attention 改造 Qwen3，继续沿用高强度 Dropout (`0.9`) 进行正则化，不引入 MoE。
- **ALT-004**: **拒绝理由 (Rejection Reason)**: 虽然 Dropout 是一种隐式负载均衡 MoE，但其随机路由缺乏语义导向性。显式的 DynMoE 提供了更灵活、更高效的算力分配方式，更符合解决 ARC 任务多样性的需求。

## 实施注意事项 (Implementation Notes)

- **IMP-001**: **RoPE 兼容性**: 必须确保 MLP-Attention 的投影层最后一层是线性的（无激活函数），以保证 RoPE 的数学有效性。
- **IMP-002**: **梯度稳定性**: 严格保证模型中所有残差连接（Residual Connection）的有效性。可考虑引入 LayerScale 并以小值初始化，以稳定训练初期的梯度流。
- **IMP-003**: **SDL 权重衰减**: SDL 损失的权重系数应在训练过程中进行退火（Annealing），以在训练早期鼓励专家分化，在后期允许功能协同。

## 参考文献 (References)

- **REF-001**: Qwen3 模型技术报告
- **REF-002**: [Neural Attention: Enhancing QKV Calculation in Self-Attention Mechanism with Neural Networks (arXiv:2310.11398)](https://arxiv.org/abs/2310.11398)
- **REF-003**: [Dynamic Mixture of Experts: An Auto-Tuning Approach for Efficient Transformer Models (arXiv:2405.14297)](https://arxiv.org/abs/2405.14297)
- **REF-004**: 内部讨论历史 (`exp/arc_rmsuon` 及相关对话)
