# Tiny-ONN 架构与实现笔记

## 核心思想：自组织的、永续学习的智能体

回归第一性原理，我们将从一个极简的、随机初始化的“盆景模型”（~0.1B）开始，通过一种模拟有机体发育的“永续预训练”范式，让其成为一个**在有限资源下通过高效信息整合（专家分化/生长/回收 + 动态注意力）来最小化自身变分自由能（VFE）的自组织智能体。**

- **内部状态 (Internal States) - 计算层**: 模型的计算核心是 **`DynONN Block`**，一个统一的动态稀疏计算单元。它整合了动态稀疏组合注意力（`DynSMHA`）和动态稀疏 MLP（`DynMoE`），均遵循“门控 + 专家”的范式。其学习目标是最小化**预测误差 (`main_loss`)** 的同时，最小化对**自身内部模型的扰动（`SurpriseMin`）**。
- **马尔可夫毯 (Markov Blanket) - 门控/自适应机制**: 所有动态组件的门控网络构成了模型的马尔可夫毯。它们负责感知内部状态（如梯度范数 `Surprise` 或内容亲和度），并采取行动（路由决策），其统一目标是通过一个混合辅助损失（梯度最小化交叉熵+KL散度损失），将信息路由到能以最低成本稳定处理它的专家，从而实现效率最大化和自然稀疏性。
- **生长/回收机制 (当前研究重点)**: 模型的**横向宽度（专家数量）**是动态的，且具有**跨生命周期的记忆**。
  - **生长 (Growth)**: 当系统无法有效处理新信息时（体现为存在持续未被路由的 token），它会“生长”出新的专家来专门应对。
  - **回收 (Pruning)**: 当一个专家的**生命周期激活率**低于阈值时，它会被回收，以释放计算资源。

这个理论框架与标准 MoE 最大的不同在于，它试图用一个**内生的、基于梯度的元学习目标**来取代传统的、外部的**负载均衡损失 (`load_balancing_loss`)**。

## 核心组件与实现

### 统一动态计算单元 (`DynONN Block` in `exp/dyn_all_poc`)

- **统一门控**: `DynSMHA` 和 `DynMoE` 共享一个统一的门控机制 `GatingNetwork`。该机制基于输入表征与一组可学习的“原型向量”（`sim_matrix`）的余弦相似度，并结合一个动态激活阈值（`gates`）来共同决定专家的激活。
- **Surprise 最小化路由**: 门控的训练目标是最小化 `Surprise`。通过 `torch.autograd.grad` 捕获 `main_loss` 对各专家输出的梯度范数，以此作为 `Surprise` 矩阵。门控的辅助损失函数被设计为预测哪个专家会产生最低的 `Surprise`，从而引导信息流向计算成本最低的路径。
- **per-token 路由**: 所有路由决策都在 token 级别独立进行，实现了最大程度的动态性和稀疏性。

### 性能优化：内外分离的梯度检查点

- **核心问题**: 在 `bfloat16` 精度下，梯度检查点的重计算过程会引入微小的浮点数差异，导致依赖硬阈值（如 `score > 0`）的门控逻辑产生不确定的结果，从而引发 `CheckpointError`。
- **解决方案**: 采用“内外分离”策略。将对浮点数敏感、计算量小的**门控逻辑 (Gating)** 放在梯度检查点**外部**正常执行；将其结果作为参数，传入一个只包含确定性且内存开销大的**主计算逻辑 (Execution)** 的函数，并对此函数应用梯度检查点。该方案已在 `exp/dyn_all_poc` 中成功实现。

## 近期路线图

### 阶段一：在 `dyn_all_poc` 中实现并验证专家自适应机制

- **目标**: 在现有实验环境中，实现专家的动态生长（Growth）和剪枝（Pruning）功能，为正式架构奠定基础。
- **实施方案**:
    1. **专家掩码 (`experts_mask`)**: 引入一个非训练的掩码，用于动态启用/禁用专家。
    2. **剪枝 (`Pruning`)**: 基于 `expert_activation_counts` 监控专家生命周期激活率，当低于阈值时，通过掩码禁用该专家。
    3. **生长 (`Growth`)**: 识别并收集“路由失败”的 token，当其数量超过阈值时，选择一个被禁用的专家进行“重生”，并使用这些 token 的平均表征来初始化其原型向量。
    4. **动态优化器**: 确保优化器能处理专家数量的动态变化。

### 阶段二：重建 `Tiny-ONN` 为 RC1

- **目标**: 废弃现有的 `tiny_onn` 架构和训练脚本，用经过验证的动态稀疏模型和支持持续学习的全新训练框架取而代之。
- **实施方案**:
    1. **架构迁移**: 将 `dyn_all_poc` 中成熟的 `DynONN Block` 迁移至 `tiny_onn/modular.py`。
    2. **重建训练框架**:
        - **健壮检查点**: 每步滚动保存最新权重，并在模型中附加一个不可训练的 `buffer` 用于记录权重生命周期的总训练步数（“年龄”）。
        - **持续学习**: 设计灵活可插拔的数据集和任务调度器，支持模型在多个数据集上进行无缝的持续学习。
    3. **多模态接口 (取代 DynSE)**: 放弃 `DynSE`。对于多模态输入，将手工设计专门的嵌入层（如 `VisionEmbedding`），其输出将与文本嵌入在同一维度空间，并统一交由 `DynONN Block` 进行处理，以实现功能的自然分化。

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
2. **结构可塑性 (`pruning_threshold`, `growth_threshold`)**: 控制专家网络的生长与回收的敏感度。
3. **内在损失权重 (`w_ce`, `w_kl`)**: 定义 `DynMoE` 和 `DynSMHA` 模块的内在学习目标。
4. **模型物理基质**: 定义了模型的基础物理尺寸，在一次生命周期中通常保持不变。
    - `hidden_size`, `head_dim`, `init_num_experts`, `max_num_experts`。
