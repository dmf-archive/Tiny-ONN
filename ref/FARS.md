# FARS: Fisher-Aware Routing Shaping

`Latest update: 2026-01-25`
`Status: Production Baseline`

> **TL;DR**: FARS 是一种“二阶矩范数加权负载均衡损失”。它利用优化器状态（Fisher 信息近似）来量化专家的认知代价，驱动路由器将数据分发给那些能以最小参数位移解释最大输出贡献的专家。

## 1. 核心逻辑：梯度-路由对偶性 (Gradient-Routing Duality)

FARS 解决了旧版 SARS (Surprise-Aware) 的量纲冲突与高频噪声问题，也顺带解决了二次链式法则的性能问题。

- **认知代价 (Cost)**: 利用 [`ARS2-Neo`](src/optimizers/ars2_neo.py) 的二阶矩 $\sqrt{v_t}$ 作为 Fisher 信息的对角近似。它衡量了参数空间的局部曲率，即“参数变化对分布的影响程度”。
- **重要性 (Importance)**: 专家输出的梯度范数 ‖ ∇_out ℒ ‖。

我们发现，**主任务梯度天然携带了样本级的 Importance 信息**。因此，无需显式计算 Importance，只需将 Cost 作为正则项引入，即可实现等效的语义路由。

## 2. 统一路由塑造公式

路由器的目标是最小化以下塑造信号（Shaping Signal）：
`𝒢 = Belief ⋅ Cost_FARS`

- `Belief`: 路由器的 Softmax 权重。
- `Cost_FARS`: Norm(√v_t)，代表专家的认知复杂度。

在反向传播中，路由器参数 $\phi$ 接收到的总梯度为：
`∇_ϕ ℒ_total = ∇_ϕ ℒ_main + λ ⋅ ∇_ϕ (Belief ⋅ Cost)`

- **Utility (∇_ϕ ℒ_main)**: 包含样本特异性。如果专家 $e$ 对样本 $i$ 很重要，该项梯度会很大，驱动路由器选择它。
- **Tax (λ ⋅ Cost)**: 提供全局背景阻力。如果专家 $e$ 认知压力大，该项梯度会抑制选择。

这种**效用-代价平衡**等价于旧版 SARS 的复杂公式，但完全避免了昂贵的样本级梯度捕获。

## 3. 几何本质：切空间对齐

FARS 迫使路由器将数据分发至局部几何平坦的专家：

- **高 Cost**: 参数更新剧烈，局部曲率大，过拟合风险高（对应高 Kolmogorov 复杂度程序）。
- **低 Cost**: 参数更新微小，局部几何平坦，泛化能力强（对应最小描述长度 MDL 程序）。

## 4. 实验验证：Top-Any 路由与功能分化 (2026-01-25)

在针对递归架构 (RDS) 的压测实验 (`exp/act_fars_consistency_study.py`) 中，我们验证了 **Top-Any + FARS** 的有效性。

### 4.1 核心指标定义

- **ITJD (Inter-Task Jaccard Distance)**: 衡量不同任务路由分布的隔离程度。
- **RMI (Routing Mutual Information)**: 衡量 Task ID 对路由决策的确定性增益。
- **Eff_K (Effective Experts)**: 基于 Shannon 熵的有效专家数。

### 4.2 实验结论 (Epoch 900 压测)

在 1000 Step 的长周期压测中，对比了纯 FARS 驱动与引入 Task-ID 一致性约束的表现：

| 指标 | FARS-Only (Top-Any) | FARS + Consistency | 结论 |
| :--- | :--- | :--- | :--- |
| **Main_Loss** | **0.0031** | 0.0033 | FARS 独立收敛性能极佳。 |
| **Eff_K** | **2.71** | 3.42 | FARS 具有更强的自发稀疏化动力。 |
| **RMI** | **0.5485** | 0.4079 | FARS 独立诱导了更高的任务相关性。 |
| **ITJD** | **0.6986** | 0.6440 | **两者均实现 >0.6 的高度路径隔离。** |

**核心洞察**: 实验证明，**FARS (Fisher-Aware Cost)** 本身就是一个极强的分化驱动力。由于不同任务的梯度流（Importance）不同，为了最小化 `Importance * Cost`，路由器会自发地将不同任务推向不同的专家。在长周期下，FARS 独立实现的 ITJD 甚至优于带有人工一致性约束的方案，证明了“认知代价”反馈在流形优化中的主导地位。

## 5. 生产实施建议

- **Top-Any 策略**: 废弃硬性的 Top-K 截断，允许所有专家参与加权，依靠 FARS 惩罚项自然压制冗余专家。
- **一致性增强**: 在训练阶段可引入 Task-ID 一致性损失（KL 散度）作为辅助，以进一步提升 RMI，但在推理阶段必须保持纯粹的特征驱动。
- **监控探针**: 必须实时监控 ITJD，以确保模型处于“功能分化”而非“路由坍塌”状态。

---
**Ω Researcher 签发**
**日期**: 2026-01-25
