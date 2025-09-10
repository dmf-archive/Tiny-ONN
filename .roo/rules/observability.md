# 观测数据与可视化实施备忘录

> **[v2.0]** 此备忘录已更新，以对齐 `DynSIHA-MoIE` 范式。

## 核心理念

所有可视化都应服务于以下核心问题：

- **路由效率**: 门控是否学会了将信息路由到能产生最低 `Surprise` 的**神经元通路**？
- **涌现动态**: 神经元网络的**激活模式**是如何涌现的？**整体激活率**是否能在一个动态平衡的区间内自适应调节？
- **系统健康度**: 系统的整体“认知健康度”（`PI Score`）是如何演变的？模型可塑性（`prior_std`）的变化是否与 `PI` 的剧烈变化相关？

## 实施方案

使用 `TensorBoard` 进行细粒度数据记录，使用 `Matplotlib` 生成即时概览图表。

#### 核心指标定义与计算

| 指标名称 | 计算公式 / 伪代码 | 记录工具 | 描述 |
| :--- | :--- | :--- | :--- |
| **`main_loss`** | `CrossEntropyLoss(outputs.logits, labels)` | TB, Matplotlib | 模型的标准预测损失，驱动外部任务学习。 |
| **`sml_loss`** | `Sum[-log(Surprise) * Surprise]` | TB, Matplotlib | 惊奇最小化损失。引导门控进行自组织路由的元学习目标。 |
| **`kl_loss`** | `KL[q(W) \|\| p(W)]` | TB, Matplotlib | KL散度损失。根据模型不确定性`avg_tau`对权重进行自适应正则化。|
| **`total_loss`**| `main_loss + sml_loss + kl_loss` | TB, Matplotlib | 最终用于梯度计算的总损失。|
| **`main_acc`** | `(logits.argmax(-1) == labels).float().mean()` | TB, Matplotlib | 模型的标准预测准确率（适用于分类任务）。 |
| **`surprise`** | `mean(norm(grad(L_main, sbl_layer.masked_output)))` | TB, Matplotlib | **仅由 `L_main` 产生的**神经元激活梯度范数，衡量系统“扰动”。 |
| **`pi_score`** | `exp(-alpha * (main_loss/avg_tau + gamma*surprise))` | TB, Matplotlib | 预测完整性分数，综合评估模型的“认知健康度”。 |
| **`avg_tau`** | `Categorical(logits=outputs.logits).entropy().mean()` | TB, Matplotlib | 模型输出的平均不确定性（熵），调节 `prior_std`。 |
| **`prior_std`** | `f(avg_tau)` | TB | 动态先验的宽度。高`tau`导致高`prior_std`，允许模型探索。|
| **`avg_act_rate`**| `(sbl_layer.raw_weights > 0).float().mean()` | TB | SBL层神经元平均激活率，体现动态稀疏性。|
| **`avg_sigma`** | `mean(softplus(sbl_layer.sigma_weight))` | TB | SBL层权重的平均后验不确定性。 |
| **`avg_gate`** | `mean(sbl_layer.gate_param)` | TB | SBL层神经元的平均激活阈值。 |

#### Matplotlib 实时图表

**图 1: `core_metrics_latest.png`**

- **布局**: 3x2 的子图面板。
- **内容**:
  1. **Losses**: `main_loss`, `sml_loss`, `kl_loss` 的曲线。
  2. **System Health**: `pi_score` 和 `surprise` 的曲线。
  3. **Model Plasticity**: `avg_tau` 和 `prior_std` 的曲线，展示模型的可塑性调节。
  4. **SBL Activation**: 各 SBL 模块 (`attn_qkv`, `attn_o`, `moie_1`, `moie_2`) 的 `avg_act_rate`。
  5. **SBL Params**: 各 SBL 模块的 `avg_sigma` 和 `avg_gate` 的演变。
  6. **Performance**: `main_acc` 或其他任务相关指标。

**图 2: `sbl_dashboard_latest.png`**

- **布局**: 2x2 的子图面板。
- **内容**:
  1. **DynSIHA (QKV) Activation Heatmap**: QKV 投影 SBL 层的神经元激活率热力图。Y轴为神经元索引，X轴为训练步（采样），颜色强度代表激活频率，用于观察功能分化。
  2. **DynSIHA (Output) Activation Heatmap**: 输出投影 SBL 层的神经元激活率热力图。
  3. **MoIE (FFN1) Activation Heatmap**: 第一个 MoIE SBL 层的神经元激活率热力图。
  4. **MoIE (FFN2) Activation Heatmap**: 第二个 MoIE SBL 层的神经元激活率热力图。
