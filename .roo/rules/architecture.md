# Tiny-ONN 最终架构与训练范式

## 1. 核心思想

通过**解耦的元学习 (Decoupled Meta-Learning)** 范式，训练一个动态稀疏混合专家（SMoE）语言模型。

其核心机制是：专家网络（快系统）专注于最小化预测误差，而门控网络（慢系统）则通过元学习，学会将输入 Token 路由到能以**最低学习成本（最小梯度范数/Surprise）** 处理它的专家。

这种设计旨在通过动态稀疏路由在架构层面“物理隔离”专家的知识领域，以对抗灾难性遗忘，并通过元学习优化路由策略，最小化内部惊奇（Surprise），从而提高模型的能效比与性能。

## 2. 核心架构：动态稀疏混合专家 (DynMoE) 与 SMK 路由

我们的核心架构借鉴了 **DynMoE** 的动态激活思想，并结合了我们独创的 **SMK (Surprise Min_K)** 路由训练机制，形成了一套独特的“**动态激活量 + 海量超轻量专家**”的超稀疏混合专家（Hyper-SMoE）方案。

与将大型 MLP 直接作为专家的主流 MoE 思路不同，Tiny-ONN 将模型的复杂性从“单个专家的深度”转移到了“海量专家间的协同”。

- **结构**: 在每个 Transformer Block 中，我们将原有的稠密 MLP 层替换为一个 MoE 层。该层包含一个门控网络（Router）和海量的、极度轻量化的专家网络。
  - `num_experts_per_layer`: 32
  - `moe_intermediate_size`: 64 (每个专家都非常小)

- **路由与训练机制 (SMK)**: 我们抛弃了早期需要完整反向传播来收集梯度的复杂元学习范式。新的 SMK 策略借鉴了 DynMoE 的经验，并进行了关键修正，实现了更高效的训练：
  1. **选择性更新**: 门控网络进行标准的 Top-k 选择，只有被选中的专家会被激活并参与计算，类似于标准 MoE 的前向传播。
  2. **惊奇度元学习**: 门控网络的学习目标被修正为 **最小化内部惊奇（Surprise）**。其损失函数是基于 **被选中的专家** 在每个 token 上产生的梯度 L2 范数（即 Surprise）计算的交叉熵损失。通过这种方式，门控网络通过元学习，学会将输入 Token 路由到能以最低学习成本（最小梯度扰动）处理它的专家，而无需对所有专家进行反向传播。

## 3. 训练范式：解耦的元学习 (快思慢想)

训练流程被明确地分为两个解耦的阶段，分别优化专家和门控。

### 3.1 专家学习 (快系统) 与经验提取

此阶段的目标是让专家网络最小化预测误差，并同时捕获用于训练门控的“惊奇度”信号。

1. **状态设定**: 门控网络（Router）参数被冻结 (`requires_grad=False`)。
2. **Hook 注册**: 在每个专家模块上注册 `forward_hook` 和 `backward_hook`。
3. **前向传播**: 
   - 输入批次通过模型，门控根据当前策略进行路由。
   - `forward_hook` 触发，保存每个专家接收到的输入张量 `input_tensor`。
4. **专家更新**:
   - 根据最终 `logits` 计算主损失 `L_main` (交叉熵)。
   - 执行 `L_main.backward()`。
   - `optimizer_experts.step()` 更新专家权重。
5. **经验提取 (在 `backward` 过程中)**:
   - `backward_hook` 被触发，接收到输出梯度 `grad_output`。
   - 在 Hook 内部，通过 `torch.einsum("bi,bo->boi", input_tensor, grad_output)` 计算出 per-token 的真实梯度。
   - 计算梯度范数 `torch.linalg.vector_norm()` 得到 **Surprise**。
   - 将包含以下信息的经验元组缓存下来：
     - `token_hidden_state`: 输入到当前 MoE 层的隐状态。
     - `router_logits`: 门控对该 Token 输出的原始 logits。
     - `per_expert_surprise`: 该 Token 在其**被路由到的**专家上产生的真实梯度范数。

### 3.2 门控学习 (慢系统)

此阶段的目标是让门控网络学会预测并选择能产生最小“内部惊奇”的路由路径。

1. **状态设定**: 专家网络参数被冻结。
2. **数据准备**: 从之前步骤缓存的经验中获取一批训练数据。
3. **真值计算**: 对于每一个样本，通过对缓存的 `per_expert_surprise` 取 `argmin()`，找到能产生最小 Surprise 的**最优专家 (`optimal_expert`)**。
4. **门控更新**:
   - 使用缓存的 `router_logits` 作为预测值。
   - 计算门控损失 `L_router = CrossEntropyLoss(router_logits, optimal_expert)`。
   - 执行 `L_router.backward()`。
   - `optimizer_router.step()` 更新门控权重。

## 4. 核心技术验证

- **Per-Token 梯度捕获**: 已通过 `exp/grad_hook_poc` 验证，使用 `forward_hook` 保存输入、`backward_hook` 结合 `torch.einsum` 计算梯度的方式，是高效且可行的，能够在单次反向传播中精确捕获 per-token 梯度范数，作为“Surprise”信号。
- **`transformers` 集成**: 已通过 `exp/integration_poc` 验证，将自定义 MoE 模块动态替换 `transformers` 模型的 MLP 层是完全可行的。
