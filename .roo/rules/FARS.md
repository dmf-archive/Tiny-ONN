# FARS: Fisher-Aware Routing Shaping

`Latest update: 2026-01-23`
`Status: Production Baseline`

> **TL;DR**: FARS 是一种“二阶矩范数加权负载均衡损失”。它利用优化器状态（Fisher 信息近似）来量化专家的认知代价，驱动路由器将数据分发给那些能以最小参数位移解释最大输出贡献的专家。

## 1. 核心逻辑：从“惊奇”到“代价”

FARS 解决了旧版 SARS (Surprise-Aware) 的量纲冲突与高频噪声问题，也顺带解决了二次链式法则的性能问题。

- **认知代价 (Cost)**: 利用 [`ARS2-Neo`](src/optimizers/ars2_neo.py) 的二阶矩 $\sqrt{v_t}$ 作为 Fisher 信息的对角近似。它衡量了参数空间的局部曲率，即“参数变化对分布的影响程度”。
- **重要性 (Importance)**: 专家输出的梯度范数 ‖ ∇_out ℒ ‖。

## 2. 统一路由塑造公式

路由器的目标是最小化以下塑造信号（Shaping Signal）：
`𝒢 = Importance ⋅ (Belief - α ⋅ Cost_FARS)`

- `Belief`: 路由器的原始 Logits。
- `Cost_FARS`: Norm(√v_t)，代表专家的认知复杂度。
- `α`: 复杂度惩罚系数。

## 3. 几何本质：切空间对齐

FARS 迫使路由器将数据分发至局部几何平坦的专家：

- **高 Cost**: 参数更新剧烈，局部曲率大，过拟合风险高（对应高 Kolmogorov 复杂度程序）。
- **低 Cost**: 参数更新微小，局部几何平坦，泛化能力强（对应最小描述长度 MDL 程序）。

## 4. 实施参考

具体实现详见 [`src/tasks/arc/shaper.py`](src/tasks/arc/shaper.py)。它通过直接读取优化器状态实现“零额外反向传播”的高效路由塑造。
