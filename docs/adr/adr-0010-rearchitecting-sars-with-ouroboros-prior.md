---
title: "ADR-0010: 通过严格的贝叶斯反演重构SARS元学习"
status: "Accepted"
date: "2025-10-09"
authors: "Ω Researcher, Tiny-ONN 课题组"
tags:
  [
    "architecture",
    "decision",
    "meta-learning",
    "sars",
    "bayesian-inversion",
    "information-gain",
  ]
supersedes: "ADR-0009"
---

# ADR-0010: 通过严格的贝叶斯反演重构 SARS 元学习

## 状态 (Status)

Proposed | **Accepted** | Rejected | Superseded | Deprecated

## 背景 (Context)

自项目启动以来，SARS 元学习框架始终无法引导出有意义的稀疏路由，模型持续表现出“过度激活”和“灾难性遗忘”的核心问题。在经历了对 `Goodness` 函数的多次迭代以及对损失函数（从分布对齐的 JSD/BCE/MSE 到错误的贝叶斯更新）的反复探索后，所有尝试均告失败。这迫使我们进行一次最根本的、从第一性原理出发的理论反思，以解决这一长期存在的顽疾。

## 核心诊断：在贝叶斯更新与贝叶斯反演之间的理论摇摆

我们最终认识到，之前所有失败的根源在于对我们自身系统（DVA）特性的误解，以及由此导致的在“贝叶斯更新”和“贝叶斯反演”两种对立范式间的摇摆不定。

- **错误的“分布对齐”范式**: 该范式将 `goodness_logits` 视为一个理想的“目标后验”，并试图用 MSE 等损失函数让模型的实际后验 `routing_logits` 去直接拟合它。这在理论上是错误的，因为它忽略了 `goodness_logits` 只是一个充满噪声的启发式信号，而非一个完备的概率分布。
- **错误的“贝叶斯更新”范式**: 该范式错误地将元学习的目标设定为 `L_meta = Entropy(Softmax(posterior + likelihood))`。这虽然看似符合贝叶斯思想，但它旨在“更新”后验，而非塑造我们真正关心的、模型内在的“先验”。

我们系统的本质是**可微分变分分析 (DVA)**，而非经典的变分推断 (VI)。这意味着：

- 我们的**后验 (Posterior)** 是可解析的，由 `routing_logits` 直接表示。
- 我们的**似然 (Likelihood)** 是可以被代理的，由 `goodness_logits` 近似表示。
- 我们真正想要优化的，是那个不可解的、隐式的**先验 (Prior)**。

因此，我们的任务性质决定了我们必须采用**贝叶斯反演 (Bayesian Inversion)**。

## 决策 (Decision)：坚持贝叶斯反演，并加固其理论基础

我们最终的决策是回归并坚持贝叶斯反演框架，同时为其核心假设提供更坚实的理论基础和更鲁棒的工程实现。

### 1. 理论基础：从信息增益推导似然

我们承认“梯度 ≠ 似然”的批判。然而，我们通过更深刻的理论推导，在这两者之间建立了一座桥梁：

- **信息增益 (Information Gain)**: 参数的梯度范数，尤其是 `mu_grad_norm` (学习成本)，衡量了为了拟合数据，模型信念需要更新的幅度。这在信息论上等价于后验与先验之间的 KL 散度 `$D_{KL}(Posterior || Prior)$`。
- **似然与信息增益的反比关系**: 一个高似然的事件意味着模型的先验能很好地解释数据，因此无需大幅更新，信息增益低。反之，一个低似然的事件（高“惊奇度”）则需要高信息增益。
- **结论**: `log(Likelihood) ∝ -Information Gain`。这为我们将 `goodness_logits` 中包含的 `-mu_grad_norm` 惩罚项视为对数似然的合理组成部分提供了理论依据。`goodness_logits` 被最终确认为对 `log(Likelihood)` 的一个**有噪声但必要的工程代理**。

### 2. 最终公式：最小化先验熵

基于此，我们重申最终的元学习损失函数：

1. **推断先验**: `prior_logits = routing_logits - goodness_logits.detach()`
2. **最小化其熵**: `L_meta = Entropy(Softmax(prior_logits))`

最小化这个损失函数，等价于驱动模型的隐式路由偏好（先验）向一个更“尖锐”、更稀疏、更确定的状态演进。

### 3. 最终实施细节 (包含稳定性加固)

1. **`Goodness` (Likelihood Proxy) 公式**: 我们最终确定的 `Goodness` 公式如下，其内部组件现在有了清晰的理论对应：
   `goodness_logits = norm_masked_output_grad * (norm_masked_output - norm_mu_grad)`

   - `norm_masked_output_grad` & `norm_masked_output`: **正面证据** (任务相关性与前向贡献)。
   - `norm_mu_grad`: **负面证据** (信息增益惩罚)。
   - **内部归一化**: 所有三个组件在组合前都经过了独立的 `mas_normalize`，以确保它们在相同的尺度上进行比较，这是一种保留了本地相对性的排序。

2. **损失函数 (Prior Entropy)**: [`exp/arc/train.py`](exp/arc/train.py) 中的 `_calculate_meta_loss` 被实现为计算 `Entropy(Softmax(routing_logits - goodness_logits.detach()))`。

3. **稳定性加固 (采纳自外部报告)**:
   - **似然信号平滑**: 为进一步处理 `goodness_logits` 的噪声，我们在 `_calculate_goodness_jit` 的末尾对其应用了**标准化** `(x - mean) / std`。
   - **元梯度裁剪**: 在 `compute_and_apply_gradients` 中，对计算出的元梯度应用 `torch.nn.utils.clip_grad_norm_`，以防止优化过程中的梯度爆炸。

## 后果 (Consequences)

### 积极 (Positive)

- **理论自洽**: 新框架在理论上是自洽的，它承认了其核心组件的“代理”性质，并通过贝叶斯反演给出了一个逻辑上最合理的解决方案。
- **目标明确**: 元学习的目标被最终确定为单一、明确的“最小化隐式先验的不确定性”。
- **实验成功**: 实施该最终框架后，模型首次表现出稳定、持续的自发稀疏化趋势，同时保持了高效、快速的任务收敛能力，解决了长期存在的“过度激活”和“学习缓慢”的问题。

### 消极 (Negative)

- **理论探索成本**: 承认在此次重构过程中，由于对贝叶斯推断、FEP 和信息增益之间关系的反复误解，导致了多次失败的尝试和资源消耗。

## 实施注意事项 (Implementation Notes)

- **IMP-001**: [`exp/arc/train.py`](exp/arc/train.py) 已被修改，`_calculate_goodness_jit` 增加了最终的标准化步骤。
- **IMP-002**: [`exp/arc/train.py`](exp/arc/train.py) 中的 `compute_and_apply_gradients` 增加了元梯度裁剪步骤。
- **IMP-003**: [`exp/arc/train.py`](exp/arc/train.py) 中的 `meta_loss` 计算被最终确认为使用 `routing_logits` 和减法操作。
- **IMP-004**: 整个调试过程凸显了在复杂理论的工程落地中，保持最高理论警惕性、进行频繁第一性原理审查以及对外部批判意见进行辩证吸收的重要性。
