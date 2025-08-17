# Tiny-ONN 架构与实现笔记

## 核心思想：自组织的、永续学习的智能体

回归第一性原理，我们将从一个极简的、随机初始化的“盆景模型”（~0.1B）开始，通过一种模拟有机体发育的“永续预训练”范式，让其成为一个**在有限资源下通过高效信息整合（专家分化/生长/回收 + 动态注意力）来最小化自身变分自由能（VFE）的自组织智能体。**

- **内部状态 (Internal States) - 计算层**: 模型的计算核心由一系列动态稀疏组件构成，均遵循“门控 + 专家”的范式。其学习目标是最小化**预测误差 (`main_loss`)**的同时，最小化对**自身内部模型的扰动（`SurpriseMin`）**。当前实现和规划中的机制包括：
  - **SurpriseMin-DynMoE**: 源自 [`exp/smk_poc.py`]，一个基于惊奇最小化（Surprise Minimization）的动态MLP专家混合层。它实现了 per-token 粒度的动态路由，其门控机制结合了余弦相似度原型路由和动态激活阈值，并修改自 [`ref/DynMoE`](ref/DynMoE)。
  - **SurpriseMin-DynSMHA**: 源自 [`exp/dyn_smha_poc`]，一个动态稀疏多头注意力层。它实现了 per-sequence 粒度的路由，路由机制与 `SM-DynMoE` 类似。**（注意：该模块尚在实验阶段，暂不可用）**
  - **DynNSA**: (规划中) 动态原生稀疏注意力。计划参考 [`ref/native-sparse-attention-pytorch`] 的实现，并对其分块注意力部分进行动态自适应改进。
- **马尔可夫毯 (Markov Blanket) - 门控/自适应机制**:
  - **统一路由机制**: 所有动态组件的门控网络构成了模型的马尔可夫毯。它们负责感知内部状态（如梯度范数 `Surprise` 或内容亲和度），并采取行动（路由决策），其统一目标是通过一个混合辅助损失（梯度最小化路由 + 多样性惩罚），将信息路由到能以最低成本稳定处理它的专家，从而实现效率最大化和自然稀疏性。
- **生长/回收机制**: 模型的**横向宽度（专家数量）**是动态的，且具有**跨生命周期的记忆**。
  - **生长**: 当系统无法有效处理新信息时（体现为存在持续未被路由的 token），它会“生长”出新的专家来专门应对。
  - **回收**: 当一个专家的**生命周期激活率**低于阈值时，它会被回收，以释放计算资源。

这个理论框架与标准 MoE 最大的不同在于，它试图用一个**内生的、基于梯度的元学习目标**来取代或增强传统的、外部的**负载均衡损失 (`load_balancing_loss`)**。

## 已验证机制

### 统一训练循环

经 `exp/smk_poc.py` 和 `exp/DynNSA/` 实验验证，当前**唯一、正确**的训练范式是**统一训练循环**，将所有辅助损失（门控损失、稀疏性损失等）与主损失相加，进行单次联合反向传播。

### `DynMoE` 核心机制 (源自 `exp/smk_poc.py`)

- **门控**: 基于 `Surprise`（`main_loss` 对专家输出的梯度范数）的路由。
- **梯度捕获**: 通过 `torch.autograd.grad(main_loss, full_expert_outputs)` 来获取 `per-token-per-expert` 粒度的梯度范数，即 `Surprise` 矩阵。
- **损失函数**: 混合损失 `gating_loss = w_ce * CrossEntropy(logits, argmin(Surprise)) + w_kl * KL_Div(logits, softmax(-Surprise))`。该损失函数已被验证可以有效引导门控进行效率最大化路由，并自然产生稀疏性。
- **废弃机制：专家梯度保护**: 经 `exp/SurpriseMin_DynMoE_PoC.py` 实验最终验证，任何形式的手工梯度干预（无论是裁剪还是选择性更新）都是不必要的过度设计。最有效、最符合自由能原理的训练范式是让 `main_loss` 和 `gating_loss` 组成的联合损失函数进行端到端的、无干预的自由优化。系统本身的动态（DynMoE 动态稀疏路由+ `gating_loss` 作为梯度的函数）已经构成了足够且更优的自调节机制。

## 观测数据与可视化实施备忘录

### 核心理念

所有可视化都应服务于以下核心问题：

- **DynMoE**: 门控是否学会了将信息路由到能产生最低 `Surprise` 的专家？专家网络的功能分化是如何演变的？
- **DynSMHA & DynNSA**: 注意力机制的稀疏度 (`avg_k`) 和策略权重是否随着模型的困惑度 (`tau`) 动态变化？
- **系统整体**: 系统的整体“认知健康度”（`PI Score`）是如何演变的？专家生长/回收事件是否与 `PI` 的剧烈变化相关？

### 实施方案

使用 `TensorBoard` 进行细粒度数据记录，使用 `Matplotlib` 生成即时概览图表。

#### 核心指标定义与计算

| 指标名称 | 计算公式 / 伪代码 | 记录工具 | 描述 |
| :--- | :--- | :--- | :--- |
| **`main_loss`** | `CrossEntropyLoss(outputs.logits, labels)` | TB, Matplotlib | 模型的标准预测损失。 |
| **`moe_gating_loss`** | `w_ce * ce_loss + w_kl * kl_loss` | TB, Matplotlib | `DynMoE` 的混合门控损失。 |
| **`sparsity_attention_loss`** | `w_ent * MSE(k, ent) + w_sparse * k` | TB, Matplotlib | `DynNSA` 的混合稀疏性损失。 |
| **`main_acc`** | `(logits.argmax(-1) == labels).float().mean()` | TB, Matplotlib | 模型的标准预测准确率。 |
| **`gating_acc`** | `(logits.argmax(-1) == surprise.argmin(-1)).float().mean()` | TB, Matplotlib | `DynMoE` 门控路由“正确率”。 |
| **`avg_k`** | `mean(dynamic_k)` | TB, Matplotlib | `DynNSA` 的平均激活块数。 |
| **`surprise`** | `mean(norm(grad(L_main, expert_outputs)))` | TB, Matplotlib | **仅由 `L_main` 产生的**专家参数梯度范数。 |
| **`pi_score`** | `exp(-alpha * (main_loss/tau + gamma*surprise))` | TB, Matplotlib | 预测完整性分数，综合评估模型的“认知健康度”。 |
| **`tau`** | `Categorical(logits=outputs.logits).entropy().mean()` | TB, Matplotlib | 模型输出的平均不确定性（熵）。 |
| **`num_experts`** | `len(model.moe.experts)` | TB | 模型当前的专家总数。 |

#### b. Matplotlib 实时图表规范

`Visualizer` 模块将负责在每个训练步（或每 N 步）结束时，更新并覆盖保存两张图。

**图 1: `core_metrics_latest.png`**

- **布局**: 3x2 的子图面板。
- **内容**:
  1. **Loss**: `main_loss`, `gating_loss`, `sparsity_loss` 的曲线。
  2. **Accuracy**: `main_acc` 和 `gating_acc` 的曲线。
  3. **PI Score**: `pi_score` 的曲线。
  4. **Surprise & Tau**: `surprise` 和 `tau` 的曲线。
  5. **DynNSA Dynamics**: `avg_k` 的曲线。
  6. **Expert Dynamics**: `num_experts` 的曲线。

**图 2: `expert_dashboard_latest.png`**

- **布局**: 2x2 的子图面板。
- **内容**:
  1. **Router Choice Scatter**: 专家激活散点图。Y轴为专家ID，X轴为训练步。
  2. **Lifecycle Activation Rate**: 专家生命周期激活率条形图。Y轴为专家ID，X轴为激活率。
  3. **Router Choice Heatmap**: 专家激活热力图（如果有多数据集）。
  4. **Expert Age**: 专家“年龄”（自创建以来的步数）条形图。

### 模块接口

- **`Logger`**: 提供 `log_metrics(metrics_dict)` 方法。
- **`Visualizer`**: 提供 `update_plots(history)` 方法。

## 核心元参数

本节旨在明确 `Tiny-ONN` 架构中必须由研究员设定的**核心元参数**。我们的目标是通过最大化模型的自调节能力，将需要手动调整的参数数量降至最低。

### 废除的传统超参数

以下传统超参数已被模型的内在动态机制所取代或可以被动态控制：

- **学习率调度器 (LRS)**: 由于`SurpriseMin`和`DynMoE`等机制，无需使用LRS。
- **MoE Top-K**: 被 `DynMoE` 的 `Surprise` 门控动态决定。
- **NSA Top-K**: 被 `DynNSA` 的熵驱动机制动态决定。
- **专家数量 (`num_experts`)**: 被专家动态再生/回收机制所取代。
- **结构可塑性 (`k_growth`, `k_prune`)**: 控制专家网络的生长与回收。默认值为1.0，可由 `PI` 动态调节。
- **内在损失权重 (`w_ce`, `w_kl`, `w_entropy`, `w_sparse`)**: 定义 `DynMoE` 和 `DynNSA` 模块的内在学习目标。默认值为1.0，可由 `PI` 动态调节。

### 核心元参数

研究员需要设定的，是定义这个自调节系统**基本行为模式**的少数几个元参数：

1. **`PI` 超参数 (`alpha`, `gamma`)**:
    - **描述**: 定义了“认知健康度” `PI` 分数的计算方式，是整个系统的“价值函数”。
    - `alpha`: 控制 `PI` 分数的整体敏感度，即环境压力。某种意义上来说，可以链接到钱包余额/当前实例可用算力。
    - `gamma`: 平衡 `PI` 计算中“不准确性成本 (`loss/tau`)”与“复杂度成本 (`Surprise`)”之间的权重。代表了模型的学习风格趋于“适应外部世界”还是“维护自我稳定”。

2. **模型物理基质**:
    - **描述**: 定义了模型的基础物理尺寸，在一次生命周期中通常保持不变。
    - `hidden_size`, `num_heads`, `init_num_experts`, `max_num_experts`, `max_selected_blocks`。
