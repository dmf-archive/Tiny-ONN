# Tiny-ONN 最终架构与训练范式

## 1. 核心思想

通过**解耦元学习 (Decoupled Meta-Learning)** 范式，训练一个动态稀疏混合专家（SMoE）语言模型。

其核心机制是：专家网络（快系统）专注于最小化预测误差，而门控网络（慢系统）则通过元学习，学会将输入 Token 路由到能以**最低学习成本（最小梯度范数/Surprise）** 处理它的专家。

这种设计旨在通过动态稀疏路由在架构层面“物理隔离”专家的知识领域，以对抗灾难性遗忘，并通过元学习优化路由策略，最小化内部惊奇（Surprise），从而提高模型的能效比与性能。

## 2. 核心架构：动态稀疏混合专家 (DynMoE) 与 SMK 路由

我们的核心架构借鉴了 **DynMoE** 的动态激活思想，并结合了我们独创的 **SMK (Surprise Min_K)** 路由训练机制，形成了一套独特的“**动态激活量 + 海量超轻量专家**”的超稀疏混合专家（Hyper-SMoE）方案。

与将大型 MLP 直接作为专家的主流 MoE 思路不同，Tiny-ONN 将模型的复杂性从“单个专家的深度”转移到了“海量专家间的协同”。

- **结构**: 在每个 Transformer Block 中，我们将原有的稠密 MLP 层替换为一个 MoE 层。该层包含一个门控网络（Router）和海量的、极度轻量化的专家网络。
  - `num_experts_per_layer`: 32
  - `moe_intermediate_size`: 64 (每个专家都非常小)

- **路由与训练机制 (SMK)**: 我们的训练范式本质上是一个**双层预测系统**：
    1. **专家（快系统）**: 负责**预测内容**，其优化目标是最小化传统的预测损失（`L_main`）。
    2. **门控（慢系统）**: 负责**预测成本**，即预测“哪个专家能用最低的惊奇度（Surprise）解决问题”。
  
这个机制通过以下两个核心部分实现：

  1. **动态K选择**: 门控网络为每个Token动态决定激活多少个（`K`）专家。这个决策由一个全局的、作为超参数的**认知预算**（`Surprise_Budget`，由PI公式中的`α`调控）来约束。门控会激活其预测中`Surprise`最低的专家，直到它们的预测`Surprise`之和达到预算上限。这强制模型以最能接受的方式使用其专家资源——未来可以考虑直接接入**算力配额或钱包余额**。
  2. **路由优化**: 门控的学习信号（`L_router`）来自于一个交叉熵损失。它驱动门控的**路由权重**（`router_logits`）去对齐那个在事后被证明（通过`backward_hook`捕获的逐token 梯度L2范数作为变分自由能的启发式代理——计算参数空间的KL散度代价高昂）实际产生**最小`Surprise`**的专家。

## 3. 训练范式：统一训练步内的解耦优化

我们的训练范式在一个统一的训练步骤（`train_step`）中，通过两个串联的优化阶段，实现对专家（快系统）和门控（慢系统）的解耦优化。

### **第一阶段：专家优化与“惊奇度”信号提取**

1. **前向传播**:
    - 输入批次通过模型，门控根据当前策略路由，模型输出 `logits`。
    - 在此过程中，通过 `forward_hook` 捕获并暂存每个专家接收到的 `input_tensor` 和对应的 `token_indices`。
2. **主损失计算与专家更新**:
    - 根据模型 `logits` 计算主预测损失 `L_main` (交叉熵)。
    - **执行第一次反向传播**: `L_main.backward(retain_graph=True)`。`retain_graph=True` 是关键，它保留了计算图，为后续门控优化做准备。
    - **更新专家**: `optimizer_experts.step()`。此时，只有专家网络的权重被更新。
3. **“惊奇度”信号提取**:
    - 在 `L_main.backward()` 过程中，`backward_hook` 被触发。
    - Hook 内部利用暂存的 `input_tensor` 和传入的 `grad_output`，通过 `torch.einsum` 精确计算出每个被激活的专家在每个 Token 上的梯度范数，即“逐token惊奇” (`per_expert_surprise`)。

### **第二阶段：门控优化**

1. **真值（最优专家）确定**:
    - 收集所有专家的 `per_expert_surprise`。
    - 对每个 Token，通过 `torch.argmin()` 找到产生最小惊奇度的专家索引，作为该 Token 的“最优路由目标” `optimal_expert_indices`。
2. **门控损失计算与更新**:
    - 从前向传播中获取门控网络输出的 `router_logits`。
    - **计算门控损失**: `L_router = CrossEntropyLoss(router_logits, optimal_expert_indices)`。这个损失驱动门控去预测能产生最小惊奇度的专家。
    - **执行第二次反向传播**: `L_router.backward()`。
    - **更新门控**: `optimizer_router.step()`。此时，只有门控网络的权重被更新。

## 4. 核心技术验证

- **Per-Token 梯度捕获**: 已通过 `exp/grad_hook_poc` 验证，使用 `forward_hook` 保存输入、`backward_hook` 结合 `torch.einsum` 计算梯度的方式，是高效且可行的，能够在单次反向传播中精确捕获 per-token 梯度范数，作为“Surprise”信号。
- **`transformers` 集成**: 已通过 `exp/integration_poc` 验证，将自定义 MoE 模块动态替换 `transformers` 模型的 MLP 层是完全可行的。
