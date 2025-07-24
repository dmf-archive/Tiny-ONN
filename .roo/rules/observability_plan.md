# DevLog: 观测数据与可视化实施备忘录

## 1. 核心理念

本项目的训练过程不仅是为了优化模型，更是为了验证 IPWT 理论框架下的核心假设。因此，建立一套全面的、实时的观测体系至关重要。该体系旨在提供对模型“认知动态”的深度洞察，而不仅仅是传统的损失下降曲线。

所有可视化都应服务于以下核心问题：
- 模型的预测完整性（PI）是如何演变的？
- 门控（慢系统）是否学会了将信息路由到能以最低成本（最小Surprise）处理它的专家（快系统）？
- 专家网络的功能分化是如何体现的？

## 2. 实施方案

我们将通过 `TensorBoard` 和 `Matplotlib` 两个工具来满足不同的观测需求。

- **TensorBoard**: 用于记录所有原始标量数据，便于进行细粒度的、可交互的事后分析。
- **Matplotlib**: 用于在训练过程中生成高度定制化的、信息密集的图表快照，滚动更新，提供对最新训练动态的即时概览。

### 2.1 核心指标定义与计算

以下是在每个 "Hyper Step"（一次前向 + 一次 `autograd.grad` + 一次 `backward`）中必须计算和记录的核心指标：

| 指标名称 | 计算公式 / 伪代码 | 记录工具 | 描述 |
| :--- | :--- | :--- | :--- |
| **`main_loss`** | `CrossEntropyLoss(outputs.logits, labels)` | TB, Matplotlib | 模型的标准预测损失。 |
| **`gating_loss`** | `CrossEntropyLoss(router_logits, surprise_dyn_min_k)` | TB, Matplotlib | 门控的元学习损失。 |
| **`main_acc`** | `(outputs.logits.argmax(-1) == labels).float().mean()` | TB, Matplotlib | 模型的标准预测准确率。 |
| **`gating_acc`** | `(dyn_top_k == surprise_dyn_min_k).float().mean()` | TB, Matplotlib | 门控路由的“正确率”，衡量其选择是否与最小Surprise专家一致。 |
| **`surprise`** | `mean(norm(grad(L_main, expert_params)))` | TB, Matplotlib | **仅由 `L_main` 产生的**专家参数平均梯度范数。 |
| **`tau`** | `Categorical(logits=outputs.logits).entropy().mean()` | TB, Matplotlib | 模型输出的平均不确定性（熵）。 |
| **`pi_score`** | `exp(-alpha * (epsilon/tau + gamma*surprise))` | TB, Matplotlib | 预测完整性分数，综合评估模型的“认知健康度”。 |

### 2.2 Matplotlib 实时图表规范

`Visualizer` 模块将负责在每个 epoch 结束时，根据该 epoch 缓存的数据生成并覆盖保存两张图。

#### 图 1: `core_metrics_latest.png`

- **布局**: 3x2 的子图面板。
- **内容**:
  1. **Loss**: `main_loss` 和 `gating_loss` 的曲线。
  2. **Accuracy**: `main_acc` 和 `gating_acc` 的曲线。
  3. **PI Score**: `pi_score` 的曲线。
  4. **Surprise**: `surprise` 的曲线。
  5. **Tau**: `tau` 的曲线。

#### 图 2: `expert_dashboard_latest.png`

- **布局**: 2x2 的子图面板。
- **内容**:
  1. **Top-K Activation Scatter**: 专家激活散点图。Y轴为专家ID，X轴为Hyper Step。每个点代表一次`dyn_top_k`选择。
  2. **Surprise Min-K Activation Scatter**: 最小Surprise散点图。Y轴为专家ID，X轴为Hyper Step。每个点代表一次`surprise_dyn_min_k`的计算结果。
  3. **Top-K Activation Heatmap**: 专家激活热力图。Y轴为专家ID，X轴为数据集（或任务），颜色深浅表示该专家在该任务上的平均激活频率。
  4. **Surprise Min-K Heatmap**: 最小Surprise热力图。Y轴为专家ID，X轴为数据集，颜色深浅表示该专家在该任务上成为最小Surprise路径的频率。
- **数据处理**: 由于一个 epoch 包含大量 Hyper Steps，散点图需要进行**降采样**（例如每100个step取一个样本）以保持图表清晰可读。

## 3. 模块接口

- **`Logger`**: 提供 `log_metrics(metrics_dict)` 方法，接收一个包含所有核心指标的字典。
- **`Visualizer`**: 提供 `update_plots(metrics_cache, expert_data_cache)` 方法，接收两个数据缓存用于绘图。