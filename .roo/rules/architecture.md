# Tiny-ONN 最终架构与训练范式

## 1. 核心思想：解耦元学习

通过**解耦元学习 (Decoupled Meta-Learning)** 范式，训练一个动态稀疏混合专家（SMoE）语言模型。其核心机制是将训练目标分解为两个相互正交的子任务，并通过独立的梯度计算和手动的梯度合并进行优化。

- **专家网络（主任务）**: 由 `main_loss` 驱动，专注于最小化传统的语言模型预测误差。
- **门控网络 - 选择 (Selection)**: 由 `smk_loss` (Surprise-Minimizing Cross-Entropy) 驱动，通过元学习，学会预测并将每个 Token 路由到能以**最低学习成本**（最小梯度范数/Surprise）处理它的专家。这种基于学习成本的路由机制会**自然地**促使专家功能分化和系统稀疏性，而无需额外的稀疏损失项。

## 2. 核心架构：基于 DynMoE 的动态稀疏 MoE

我们将 `Qwen3` 的稠密 `MLP` 层替换为自定义的 `TinyOnnMoE` 模块，其关键特性如下：

- **海量轻量专家**: `num_experts_per_layer=32`, `moe_intermediate_size=64`。
- **动态 K 选择 (DynMoE)**:
  1. `sim_matrix` 和 `temperature` 参数产生基础的 `logits`，代表专家亲和度。
  2. 一个可学习的、每个专家独立的阈值参数 `gates` 对 `logits` 进行截断：`activated_logits = relu(logits - gates)`。
  3. 被激活的专家数量 `K` 因此成为一个动态的、由输入 `token` 和可学习参数 `gates` 共同决定的值。

## 3. 训练范式：基于 `torch.autograd.grad` 的单优化器解耦

为在实现两个正交目标的同时保证计算图的清晰和高效，我们采用**单一优化器**配合**两次独立的 `backward` 调用**。

### 核心流程

```mermaid
graph TD
    subgraph 训练循环 (_hyper_step)
        A[开始] --> B{1. 主模型前向传播};
        B --> C[计算 main_loss];
        
        subgraph "梯度计算与 Surprise 捕获"
            C --> D[main_loss.backward(retain_graph=True)];
            D -- "通过 autograd.Function 钩子" --> E[填充 surprise_context];
            D -- "PyTorch 自动" --> F[计算并累加 main_grads 到 expert_params.grad];
        end
        
        subgraph "元学习与门控梯度计算"
            E -- "用 surprise 生成 optimal_indices" --> G[计算 smk_loss];
            G --> H[smk_loss.backward()];
            H -- "PyTorch 自动" --> I[计算并累加 selection_grads 到 selection_params.grad];
        end
        
        F & I --> J{2. 单优化器步进};
        J --> K[optimizer.step()];
        K --> L[optimizer.zero_grad()];
        L --> M[结束];
    end
```

### 实现细节

1. **`TrainerEngine` (`training/engine.py`)**:
    - **职责**: 实现上述的单优化器解耦循环。
    - **核心操作**:
        1. 执行前向传播，计算 `main_loss`。
        2. 调用 `main_loss.backward(retain_graph=True)`。此操作同时完成两件事：
            - 通过自定义的 `autograd.Function` 钩子，拦截反向传播流，计算 `surprise` 并将其填充到 `surprise_context` 字典中。
            - PyTorch 自动计算 `main_loss` 相对于 `expert_params` 的梯度，并将其累加到 `.grad` 属性中。
        3. 使用 `surprise_context` 中的 `surprise` 值作为标签，计算 `smk_loss`。
        4. 调用 `smk_loss.backward()`。PyTorch 将计算 `smk_loss` 相对于 `selection_params` 的梯度，并将其**累加**到 `.grad` 属性中。
        5. 调用**一次** `optimizer.step()`，同时更新 `expert_params` 和 `selection_params`。
        6. 调用 `optimizer.zero_grad()` 清空梯度。

2. **优化器 (`train.py`)**:
    - 只创建一个 `AdamW` 优化器。
    - 为该优化器传入一个包含两个字典的列表，每个字典定义一个参数组（`expert_params`, `selection_params`）及其对应的学习率。

## 4. 观测与验证

- **`gating_acc`**: 门控路由选择与最优专家选择的匹配度，衡量**选择**任务的学习效果。
- **`avg_k`**: 平均激活专家数，衡量**稀疏**任务的学习效果。
- **专家激活热力图**: 观察专家是否形成功能分化。
