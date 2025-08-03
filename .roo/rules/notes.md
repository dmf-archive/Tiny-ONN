# Tiny-ONN 架构与实现笔记

## 1. 核心思想：基于 FEP 的自适应、自组织智能体

我们放弃“器官嫁接”式的模型手术，回归第一性原理。我们将从一个极简的、随机初始化的“盆景模型”开始，通过一种模拟有机体发育的“永续预训练”范式，让其成为一个**在有限资源下通过高效信息整合（专家分化与动态生长）来最小化自身变分自由能（VFE）的自组织智能体。**

- **内部状态 (Internal States) - 专家网络**: 一组**超迷你**专家，负责对世界（数据）建立生成模型。其学习目标是最小化**预测误差 (`main_loss`)**，并在一个受保护的梯度环境（选择性激活和基于正态分布阈值的梯度过滤）中稳定学习。
- **马尔可夫毯 (Markov Blanket) - 门控网络**: 负责感知内部状态（`Surprise`）并采取行动（路由决策）。其学习目标是**将信息路由到能以最低“惊奇度”（最小化 `Surprise`）稳定处理它的专家**，从而优先巩固已有知识，降低整体系统的 VFE。`ΔSC` 在此过程中可作为一个重要的**观察指标**，用于分析专家激活与协同贡献之间的关系，但**不直接作为**训练目标。
- **生长机制**: 模型的**横向宽度（专家数量）**是动态的。当系统无法有效处理某些新信息时（体现为存在未被路由的 token），它会“生长”出新的专家来专门应对这些信息，这是一种结构性的主动推理。

## 2. 核心架构：基于模块化组合的 Tiny-ONN

根据 `transformers` 库的最佳实践，我们放弃对 `Mixtral` 等现有模型的深度修改。我们将采用**组合优于继承**的原则，从 `PreTrainedModel` 等基础组件开始，像搭积木一样构建我们自己的、完全独立的 `TinyOnn` 模型。

这种方法的核心优势在于**代码清晰、可维护性高、与上游库解耦**，避免了未来因 `transformers` 库更新而导致的兼容性问题。

```mermaid
graph TD
    subgraph "自定义 Tiny-ONN 架构"
        A["TinyOnnConfig\n(继承 PretrainedConfig)"] --> B;
        A --> C;
        A --> D;

        B["TinyOnnAttention\n(可直接复用 Qwen3Attention)"] --> D;
        C["TinyOnnMoE\n(核心自定义逻辑:\n- ΔSC门控\n- 动态生长)"] --> D;

        D["TinyOnnLayer\n(组合 Attention 和 MoE)"] --> E;
        E["TinyOnnModel\n(继承 PreTrainedModel\n组合多层 TinyOnnLayer)"] --> F;

        F["TinyOnnForCausalLM\n(继承 PreTrainedModel\n组合 TinyOnnModel 和 LM Head\n处理主损失和门控损失)"]
    end

    G["训练循环 (train.py)"] -- "加载" --> F
```

## 3. 训练范式：统一训练循环与混合损失 (已验证方案)

经 `exp/smk_poc.py` 实验验证，解耦的 `2f2b2o` 元学习范式无法稳定收敛。当前**唯一、正确**的训练范式是**统一训练循环**，将门控损失作为辅助损失（`aux_loss`）与主损失相加，进行单次联合反向传播。

### 3.1. 核心流程图

```mermaid
graph TD
    subgraph "统一训练步"
        Start["开始"] --> FWD["1. 全模型前向传播"];
        FWD --> Outputs["logits, full_expert_outputs, router_logits"];
        
        Outputs --> L_main["2. 计算 main_loss"];
        
        subgraph "门控损失计算"
            L_main & Outputs -- "autograd.grad" --> Surprise["3. 计算 Surprise 矩阵"];
            Surprise & Outputs --> L_gate["4. 计算混合门控损失 (Hybrid Gating Loss)"];
        end
        
        L_main & L_gate --> L_combined["5. 计算 combined_loss = main_loss + w_aux * gating_loss"];
        L_combined --> BWD["6. combined_loss.backward()"];
        
        subgraph "参数更新"
            BWD --> Filter["7. (可选) 专家梯度 2σ 过滤"];
            Filter --> Step["8. optimizer.step()"];
        end

        Step --> End["结束"];
    end
```

### 3.2. 核心学习机制：基于 `Surprise` 的混合 `gating_loss`

#### a. 梯度捕获 (`autograd.grad`)
此部分与原方案一致，是**已验证**的核心组件。通过创建并填充 `full_expert_outputs` 稠密张量，再调用 `torch.autograd.grad(main_loss, full_expert_outputs)` 来获取 `per-token-per-expert` 粒度的梯度范数，即 `Surprise` 矩阵。

#### b. 混合损失函数
为解决单一目标导致的模型坍缩问题，门控损失被设计为两种目标的加权和，以平衡“利用”和“探索”：

- **硬目标 (利用)**: `ce_loss = CrossEntropy(router_logits, argmin(Surprise))`
  - 鼓励门控精确地选择 `Surprise` 最小的最优专家。
- **软目标 (探索)**: `kl_loss = KL_Div(log_softmax(router_logits), log_softmax(-Surprise))`
  - 鼓励门控的输出概率分布，从整体上拟合 `-Surprise` 的分布，从而考虑到次优选项，避免坍缩到单一专家。

**`gating_loss = w_ce * ce_loss + w_kl * kl_loss`**

### 3.3. 关键稳定化机制

- **强制探索**: 在 `DynamicGate` 的前向传播中，如果一个 token 没有激活任何专家，则强制激活 `k = num_experts / 2` 个 `logits` 最高的专家。这是防止模型早期“沉默”、保证学习信号传递的关键。
- **专家梯度保护**: 在 `optimizer.step()` 之前，对所有专家的参数梯度进行 `2σ` 离群值过滤，将异常梯度置零。这可以保护专家网络不被少数高难度样本产生的巨大梯度所“污染”，从而稳定学习过程。

## 4. 观测数据与可视化实施备忘录

（*本节内容保持不变，核心指标定义依然有效*）

### 4.1. 核心理念

所有可视化都应服务于以下核心问题：

- 门控是否学会了将信息路由到能产生最高协同贡献（ΔSC）的专家？
- 专家网络的功能分化是如何通过 ΔSC 体现的？
- 系统的整体“认知健康度”（PI Score）是如何演变的？
- 模型的路由决策与理论最优选择的偏差（Similarity）是如何演变的？

### 4.2. 实施方案

使用 `TensorBoard` 进行细粒度数据记录，使用 `Matplotlib` 生成即时概览图表。

#### a. 核心指标定义与计算

| 指标名称 | 计算公式 / 伪代码 | 记录工具 | 描述 |
| :--- | :--- | :--- | :--- |
| **`main_loss`** | `CrossEntropyLoss(outputs.logits, labels)` | TB, Matplotlib | 模型的标准预测损失。 |
| **`gating_loss`** | `w_ce * ce_loss + w_kl * kl_loss` | TB, Matplotlib | 基于 `Surprise` 的混合门控损失。 |
| **`main_acc`** | `(outputs.logits.argmax(-1) == labels).float().mean()` | TB, Matplotlib | 模型的标准预测准确率。 |
| **`gating_acc`** | `(router_logits.argmax(-1) == surprise.argmin(-1)).float().mean()` | TB, Matplotlib | 门控路由“正确率”，衡量其选择是否与最低`Surprise`专家一致。 |
| **`routing_similarity`**| `cosine_similarity(softmax(logits), softmax(-surprise)).mean()` | TB, Matplotlib | 门控输出分布与 `-Surprise` 分布的相似度。 |
| **`delta_sc`** | `mean(sigmoid(Activation) - sigmoid(Gradient))` | TB, Matplotlib | **(仅观察指标)** 净协同贡献。|
| **`surprise`** | `mean(norm(grad(L_main, expert_outputs)))` | TB, Matplotlib | **仅由 `L_main` 产生的**专家参数梯度范数。 |
| **`pi_score`** | `exp(-alpha * (main_loss/tau + gamma*surprise))` | TB, Matplotlib | 预测完整性分数，综合评估模型的“认知健康度”。 |
| **`tau`** | `Categorical(logits=outputs.logits).entropy().mean()` | TB, Matplotlib | 模型输出的平均不确定性（熵）。 |

#### b. Matplotlib 实时图表规范

`Visualizer` 模块将负责在每个 epoch 结束时，根据该 epoch 缓存的数据生成并覆盖保存两张图。

**图 1: `core_metrics_latest.png`**

- **布局**: 3x2 的子图面板。
- **内容**:
  1. **Loss**: `main_loss` 和 `gating_loss` 的曲线。
  2. **Accuracy**: `main_acc` 和 `gating_acc` 的曲线。
  3. **Routing Similarity**: `routing_similarity` 的曲线。
  4. **PI Score**: `pi_score` 的曲线。
  5. **Surprise & Net ΔSC**: `surprise` 和 `delta_sc` 的曲线。
  6. **Tau**: `tau` 的曲线。

**图 2: `expert_dashboard_latest.png`**

- **布局**: 2x2 的子图面板。
- **内容**:
  1. **Router Choice Scatter**: 专家激活散点图。Y轴为专家ID，X轴为训练步。每个点代表一次路由选择。
  2. **Max-ΔSC Choice Scatter**: 最高ΔSC散点图。Y轴为专家ID，X轴为训练步。每个点代表该step具有最高ΔSC的专家。
  3. **Router Choice Heatmap**: 专家激活热力图。Y轴为专家ID，X轴为数据集（或任务），颜色深浅表示该专家在该任务上的平均激活频率。
  4. **Max-ΔSC Choice Heatmap**: 最高ΔSC热力图。Y轴为专家ID，X轴为数据集，颜色深浅表示该专家在该任务上成为最高ΔSC路径的频率。
- **数据处理**: 由于一个 epoch 包含大量训练步，散点图需要进行**步级降采样**以保持图表清晰可读。总的来说，我们对步内数据进行求 per-expert-avg ，并在步数超过1000时对step进行随机采样使其总分辨率保持在1000步左右。

### 4.3. 模块接口

- **`Logger`**: 提供 `log_metrics(metrics_dict)` 方法，接收一个包含所有核心指标的字典。
- **`Visualizer`**: 提供 `update_plots(metrics_cache, expert_data_cache)` 方法，接收两个数据缓存用于绘图。
