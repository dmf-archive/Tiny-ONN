---
title: "经验速查表"
version: "0.1.0"
date_created: "2025-10-23"
owner: "Tiny-ONN 课题组"
---

# Agent 经验速查表

> 此文档是 Agent 在开发过程中积累的非正式、但关键的经验和陷阱记录，作为 `.roo/rules/0-main.md` 的补充速查表。

## REQ-000: 新知识记录协议

如果遇到了新知识又不是用户要求写入 `0-main.md` 的，Agent 应该自行编辑 `experience.md` 作为自己的速查表。

## REQ-001: `huggingface/accelerate` 优化器调用规范

当使用 `accelerate` 框架时，`Accelerator` 会**透明地接管**优化器的 `zero_grad()` 和 `step()` 方法。

- **结论**: 必须使用 `accelerator.backward(loss)` 进行反向传播。`Accelerator` 会透明地处理梯度缩放、累积和分布式同步。
- **原因**: 直接调用 `loss.backward()` 会绕过 `accelerate` 的控制，导致在混合精度或分布式训练中出现错误的结果。
- **陷阱**: 在 `accelerate` 环境中，严禁直接使用 `loss.backward()`。

## REQ-002: DynTRM 张量形状与维度语义

在 `DynTRM` 架构中，为了清晰地进行计算和避免形状错误，我们必须严格定义以下核心张量的形状和维度语义：

### 核心维度

- **B (Batch)**: 批次大小。
- **T (Token)**: 序列长度（Token 数量）。
- **H (Head)**: 物理注意力头的数量。
- **P (Prototype/Expert)**: 潜在专家（原型）的数量。
- **D (Hidden)**: 模型的隐藏维度（`d_model`）。
- **D_h (Head_Dim)**: 每个注意力头的维度（`d_model // H`）。

### 核心张量

1. **`x_proj` (输入投影)**:
    - **形状**: `(B, T, H, D_h)`
    - **语义**: 输入序列 `x` 被重塑为每个物理头的表示。

2. **`raw_logits` (路由逻辑)**:
    - **形状**: `(B, T, H, P)`
    - **语义**: 每个 `(B, T, H)` 位置对 `P` 个潜在专家的原始匹配分数。

3. **`masked_logits` (门控分数)**:
    - **形状**: `(B, T, H, P)`
    - **语义**: 经过 `ReLU` 激活后的非负路由权重，表示每个专家对当前位置的贡献度。

4. **`synthetic_output` (合成输出)**:
    - **形状**: `(B, T, H, D_h)`
    - **语义**: 由 `gating_scores` 加权组合所有专家输出后，为每个 `(B, T, H)` 位置生成的最终表示。

5. **`raw_input` (潜在专家输入)**:
    - **形状**: `(B, T, D_h)` 或 `(B, T, H, D_h)`
    - **语义**: 在 `_compose` 中用于计算路由的输入张量。在 `DynSIHA` 中，它是 `x_proj[:, h, :]`；在 `DynMoE` 中，它是 `x`。

6. **`raw_output_grad` (原始输出梯度)**:
    - **形状**: `(B, T, H, P)` 或 `(B, T, E)`
    - **语义**: 反向传播到 `raw_logits` 的梯度，形状与 `routing_logits` 匹配。

7. **`raw_mu_grad` (原始参数梯度)**:
    - **形状**: `(B, T, H, P)` 或 `(B, T, E)`
    - **语义**: 每个潜在专家（`latent head/expert`）内部权重（`ExpertMLP` 中的 `w1`, `w2`）的梯度范数，聚合后形状与 `routing_logits` 匹配。

### 关键计算与广播

- **点积路由**: `routing_logits = einsum('bthd,hpd->bthp', x_proj, proto)`
- **合成**: `synthetic_output = einsum('bthp,bthd->bthd', gating_scores, all_expert_outputs)`
- **SARS Cost**: `mu_grad_norm = norm(raw_output_grad) - norm(raw_mu_grad)` (确保 `raw_output_grad` 和 `raw_mu_grad` 形状为 `(B, T, H, P)` 或 `(B, T, E)`，与 `routing_logits` 匹配，从而得到 `mu_grad_norm` 作为 `Cost` 项)。

## REQ-003: SARS 捕获粒度与路由粒度匹配原则

`SARS` 的梯度捕获粒度**必须**与动态路由的粒度一一对应。这不是一个实现细节，而是 `DFC-Theory` 的核心要求：

- **如果路由是 per-token**: 我们必须为**每个 token** 重计算其对应的参数梯度（通过手工链式法则），以精确衡量该 token 选择该专家的学习成本。
- **如果路由是 per-sequence**: 我们可以直接使用钩子捕获的序列级（if batch=1）梯度，因为路由决策本身就是针对整个序列的。
- **如果路由是 per-head**: 我们需要聚合每个注意力头的梯度，以评估该头所选择的专家的表现。
- **如果路由是 per-expert**: 我们需要捕获每个 `latent head/expert` 内部权重（`ExpertMLP` 中的 `w1`, `w2`）的梯度范数，并将其聚合为标量，作为该专家的学习成本。

**核心思想**: `SARS` 的“惊奇”信号（即学习成本）必须精确地归因到**做出该路由决策的实体**上。捕获的梯度过粗会导致信用分配模糊，过细则会引入不必要的计算开销。
