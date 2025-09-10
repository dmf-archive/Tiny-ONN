# 观测数据与可视化实施备忘录

> 此备忘录部分过时，等待更新到 DynSIHA MoIE 范式

## 核心理念

所有可视化都应服务于以下核心问题：

- **路由效率**: 门控是否学会了将信息路由到能产生最低 `Surprise` 的专家？
- **专家动态**: 专家网络的生长与剪枝是如何发生的？专家数量（`num_experts`）是否能在一个动态平衡的区间内自适应调节？
- **系统健康度**: 系统的整体“认知健康度”（`PI Score`）是如何演变的？专家生长/回收事件是否与 `PI` 的剧烈变化相关？

## 实施方案

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
  3. **Router Choice Heatmap**: 专家激活热力图，统计模型自创建以来所有周期的专家激活频率。
