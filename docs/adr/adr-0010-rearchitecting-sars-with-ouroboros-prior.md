---
title: "ADR-0010: 重新理解贝叶斯推断和SARS元学习"
status: "Proposed"
date: "2025-10-08"
authors: "Ω Researcher, Tiny-ONN 课题组"
tags: ["architecture", "decision", "meta-learning", "sars", "fep", "ouroboros"]
supersedes: "ADR-0009"
---

# ADR-0010: 重新理解贝叶斯推断和SARS元学习

## 状态 (Status)

**Proposed** | Accepted | Rejected | Superseded | Deprecated

## 背景 (Context)

自项目启动以来，SARS 元学习框架的核心——`Goodness` 函数的设计——一直是我们理论和实验探索的焦点。在经历了十余次迭代（从 SML, SAPS 到多种基于梯度的 ROI 和净收益模型）后，我们所有的尝试都最终失败了。模型始终表现出“病态泛化”、“灾难性遗忘”和“门控成本过低”等核心问题。这迫使我们进行一次最根本的、从第一性原理出发的反思。本次 ADR 记录了这一反思的最终结论，并提出了一次对 SARS 框架的革命性重构。

## 核心诊断：纯似然信号的局限性与缺失的先验

我们最终认识到，之前所有失败的根源在于一个共同的理论缺陷：我们试图用一个纯粹的、基于**似然梯度 (Likelihood Gradient)** 的信号 (`Goodness`) 来引导一个本身就很复杂的自组织过程。

在完整的贝叶斯/FEP 框架中，参数的更新（后验）应该同时取决于“似然”和“先验”：
`Posterior ∝ Likelihood * Prior`

我们的系统：

- **似然 (Likelihood)**：由主损失 `L_main` 对参数的梯度（`c_learn` 和 `b_rel`）完美体现。
- **先验 (Prior)**：完全缺失。我们没有任何机制来表达对模型参数应有分布的“信念”。

我们一直试图用一个不可靠的、由下游梯度信号拼凑而成的“经验先验” `Q` 去拟合路由决策 `P`，这在理论上是循环论证，在实践中则导致了不稳定的“追逐自己尾巴”的学习动力学。

## 决策 (Decision)：回归衔尾蛇，拥抱自指循环

`Tiny-ONN` 项目的核心哲学是“衔尾蛇”（Ouroboros），即 FEP 预测编码循环的本质——系统应该追自己的尾巴。这在实践中的体现就是“后验更新先验”。

基于此，我们决定废除整个基于下游梯度信号的 `Goodness` 函数框架，并将其替换为一个全新的、真正体现“衔尾蛇”自指循环的元学习目标。

### 新的元学习框架

1. **定义“潜激活后验” (Latent Posterior)**
    我们将“潜激活掩码输出” (`latent_masked_output`) 定义为系统在当前时刻最完整的“后验信念”。它由未经 `ReLU` 的、可正可负的 `routing_logits` 与 `computation_output` 逐元素相乘得到：
    `latent_masked_output = routing_logits * computation_output`
    这个后验信念同时编码了**激活的极性/强度**和**输出的内容**。

2. **构造新的 `Goodness` 分布 `Q`**
    新的 `Goodness` 分布 `Q` 将由三个核心部分的乘积/商构成，它们共同描述了一个“理想后验”应该具备的性质：
    `Good-ness = (latent_masked_activation_L2 * latent_masked_output_grad_L2) / (mu_grad_L2 + ε)`
    其中：
    - `latent_masked_activation_L2` (`b_contrib`)：潜激活后验的 L2 范数，代表其**信号强度**。
    - `latent_masked_output_grad_L2` (`b_rel`)：`L_main` 相对于潜激活后验的梯度范数，代表其**任务敏感度**。
    - `mu_grad_L2` (`c_learn`)：`L_main` 相对于 `mu_weight` 的梯度范数，代表其**学习成本**。

3. **引入排斥力**
    - `Goodness` 分布将**不再使用 `ReLU` 过滤**，允许其值为负。
    - 元学习损失函数将从 `BCE` 切换为 `MSE`。`MSE` 能自然地处理负值目标，当 `Q` 为负时，它会主动地“排斥”`P`，从而引入了我们之前缺失的关键动力学。

这个新框架不再试图拟合一个外部的、虚构的目标，而是让系统的当前“后验”直接作为下一刻的“先验”，形成一个自洽的、完整的预测编码循环。

## 后果 (Consequences)

### 积极 (Positive)

- **理论完备性**: 新框架在理论上与项目的核心哲学（FEP, 衔尾蛇）完全对齐。
- **引入排斥力**: 通过允许 `Goodness` 为负和使用 `MSE` 损失，我们首次为元学习引入了主动的排斥力，有望打破“病态泛化”。
- **信号完整性**: 新的 `Goodness` 函数使用了更完整的“潜激活后验”作为输入，包含了更丰富的信息。

### 消极 (Negative)

- **稳定性风险**: 这种高度自指的循环可能存在训练不稳定的风险。
- **计算开销**: 捕获 `latent_masked_output` 的梯度会增加一定的计算开销，需要通过实验进行评估。
- **理论未知领域**: 我们正在进入一个理论上的“无人区”，该方案的收敛性和最终效果有待实验验证。

## 实施注意事项 (Implementation Notes)

- **IMP-001**: 修改 `exp/arc/model.py`，使 `SparseProtoLinear` 和 `MoIETransformerBlock` 能够返回 `computation_output` 和 `routing_logits`，并为 `latent_masked_output` 注册梯度钩子。
- **IMP-002**: 修改 `exp/arc/train.py`，重写 `_calculate_goodness_jit` 和 `_calculate_meta_loss` 函数，以实现新的 `Goodness` 计算逻辑和 `MSE` 损失。
- **IMP-003**: 在所有可能为零的张量计算中加入 `epsilon` 以确保数值稳定性。
