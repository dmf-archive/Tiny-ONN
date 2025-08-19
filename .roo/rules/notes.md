# Tiny-ONN 架构与实现笔记

## 核心思想：自组织的、永续学习的智能体

回归第一性原理，我们将从一个极简的、随机初始化的“盆景模型”（~0.1B）开始，通过一种模拟有机体发育的“永续预训练”范式，让其成为一个**在有限资源下通过高效信息整合（专家分化/生长/回收 + 动态注意力）来最小化自身变分自由能（VFE）的自组织智能体。**

## 核心组件与实现

`Tiny-ONN` 的核心计算单元是 **`DynONN Block`**，它在标准的 Transformer 块中用我们自研的动态稀疏组件取代了原有的注意力层和前馈网络层。

### 1. DynSMHA (Dynamic Sparse Multi-Head Attention)

- **定位**: 取代标准的 `Multi-Head Attention`。
- **工作原理**: `DynSMHA` 引入了一组可学习的“注意力头专家”（Attention Head Experts）。对于每个输入 `token`，一个共享的门控网络 (`GatingNetwork`) 会计算该 `token` 与所有专家头之间的“原型相似度”，并结合一个可学习的动态激活阈值，共同决定为该 `token` 激活哪些专家头。被激活的专家头将并行处理该 `token`，其输出被加权聚合。
- **核心优势**: 实现了 `token` 级别的、内容感知的动态稀疏注意力，允许每个 `token` 根据自身需求选择性地激活不同功能的注意力头。

### 2. DynMoE (Dynamic Mixture of Experts)

- **定位**: 取代标准的 `MLP`/`Feed-Forward` 层。
- **工作原理**: 与 `DynSMHA` 类似，`DynMoE` 包含一组 `MLP` 专家。`GatingNetwork` 同样会根据原型相似度和动态阈值，为每个 `token` 激活一个或多个最合适的 `MLP` 专家。
- **核心优势**: 实现了 `token` 级别的动态计算路由，允许模型以极高的参数效率处理复杂任务，同时保持低计算成本。

### 3. SurpriseMinLoss (Surprise Minimization Loss)

- **定位**: `DynSMHA` 和 `DynMoE` 中门控网络的统一训练目标。
- **工作原理**: `SurpriseMinLoss` 的核心假设是：一个高效的系统应将信息路由到能以最低“扰动”处理它的模块。在计算上，我们将“扰动”或“惊奇度”(`Surprise`) 定义为 **`main_loss` (主任务损失) 对各专家输出的梯度范数**。门控网络的辅助损失函数 (`gating_loss`) 被设计为预测并选择能产生最低 `Surprise` 的专家。
- **核心优势**: 该机制用一个内生的、基于梯度的元学习目标，取代了传统 MoE 模型中外部的、启发式的负载均衡损失 (`load_balancing_loss`)，使稀疏性成为模型为了最小化自身计算成本而自主学会的一种涌现行为。

## 近期与未来研究方向

### 1. DynNSA (Dynamic Native Sparse Attention)

- **定位**: `DynSMHA` 的下一代演进，旨在结合 `Native Sparse Attention (NSA)` 的 `forward` 优化与 `DynSMHA` 的动态稀疏性。
- **核心思想**: `NSA` 是一种高效的 `forward` 优化方法，它通过固定的稀疏模式（如滑动窗口和全局块的重要性采样）来逼近全注意力。`DynNSA` 的目标是用一个**可学习的、由 `Surprise` 驱动的重要性阈值**来取代 `NSA` 静态的 `top-n` 选择机制。这样，我们既可以利用 `NSA` 高度优化的计算核，又能实现真正动态、内容感知的稀疏注意力。

### 2. DynSE (Dynamic Sparse Embeddings) - (远期探索)

- **定位**: 将动态稀疏思想扩展到词嵌入层。
- **核心思想**: 探索是否能让模型根据上下文动态地选择或组合一部分嵌入向量，而不是总使用静态的、稠密的嵌入矩阵。
- **挑战**: 嵌入层的训练目标与 `SurpriseMinLoss` 的假设不完全匹配，需要探索新的训练范式。

## 性能优化

### 内外分离的梯度检查点

- **核心问题**: 在 `bfloat16` 精度下，梯度检查点的重计算过程会引入微小的浮点数差异，导致依赖硬阈值（如 `score > 0`）的门控逻辑产生不确定的结果，从而引发 `CheckpointError`。
- **解决方案**: 采用“内外分离”策略。将对浮点数敏感、计算量小的**门控逻辑 (Gating)** 放在梯度检查点**外部**正常执行；将其结果作为参数，传入一个只包含确定性且内存开销大的**主计算逻辑 (Execution)** 的函数，并对此函数应用梯度检查点。该方案已在 `exp/dyn_all_poc` 中成功实现。

## 观测数据与可视化实施备忘录

### 核心理念

所有可视化都应服务于以下核心问题：

- **路由效率**: 门控是否学会了将信息路由到能产生最低 `Surprise` 的专家？
- **专家动态**: 专家网络的生长与剪枝是如何发生的？专家数量（`num_experts`）是否能在一个动态平衡的区间内自适应调节？
- **系统健康度**: 系统的整体“认知健康度”（`PI Score`）是如何演变的？专家生长/回收事件是否与 `PI` 的剧烈变化相关？

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

#### Matplotlib 实时图表

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
  3. **Router Choice Heatmap**: 专家激活热力图，统计模型自创建依赖所有周期的专家激活频率。

## 核心元参数

本节旨在明确 `Tiny-ONN` 架构中必须由研究员设定的**核心元参数**。我们的目标是通过最大化模型的自调节能力，将需要手动调整的参数数量降至最低。

以下传统超参数已被模型的内在动态机制所取代或可以被动态控制：

- **学习率调度器 (LRS)**: 由于`SurpriseMin`和`DynMoE`等机制，无需使用LRS。
- **MoE Top-K**: 被 `DynMoE` 的 `Surprise` 门控动态决定。
- **专家数量 (`num_experts`)**: 被专家动态再生/回收机制所取代。

研究员需要设定的，是定义这个自调节系统**基本行为模式**的少数几个元参数：

1. **`PI` 超参数 (`alpha`, `gamma`)**: 定义了系统的“价值函数”，平衡“适应外部世界”与“维护自我稳定”的学习风格。
2. **结构可塑性 (`pruning_threshold`, `growth_threshold`)**: 控制专家网络的生长与回收的敏感度。在未来，该参数将由`PI Score`动态调节：低`PI`时，系统会更积极地探索（重生更多专家）；高`PI`时，则更倾向于稳定和利用（最小化重生）。
3. **内在损失权重 (`w_ce`, `w_kl`)**: 定义 `DynMoE` 和 `DynSMHA` 模块的内在学习目标。目前默认设置为1.0，但未来也可能由`PI Score`进行动态调节。
4. **模型物理基质**: 定义了模型的基础物理尺寸，在一次生命周期中通常保持不变。
    - `hidden_size`, `head_dim`, `init_num_experts`, `max_num_experts`。
