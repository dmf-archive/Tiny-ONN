# Fisher-Aware Routing Shaping proposal

`Date: 2026-01-03`
`Status: Refined Theory / Design Specification`
`Version: 2.0`

## 1. SARS 局限性分析

[`SARS`](exp/arc_dyntrm/train.py:97) (Surprise-Aware Routing Shaping) 面临的核心挑战：

1. **量纲冲突**: `logits` (无量纲) 与 `norm_mu_grad` (具有物理量纲的 Frobenius 范数) 直接相减导致超参数 `w_meta` 缺乏物理含义。
2. **高频噪声**: 一阶瞬时梯度 `∇L` 包含大量随机噪声，导致路由决策在训练初期剧烈抖动。

## 2. Fisher 信息近似

利用 [`AdaRMSuon`](ref/F3EO/optimizer/ada_rmsuon.py) 的二阶矩估计 `v_t` 作为 Fisher 信息矩阵 (FIM) 的对角近似：
`F(θ) ≈ E[∇L ⊗ ∇L] ≈ v_t`

Fisher 信息定义了参数流形的度量，用于衡量专家的认知代价。

## 3. 实施路径

### 3.1 路径 A: SNR 视角 (无量纲化)

定义 `Cost_SNR` 为 Adam 更新步长的范数，代表专家在流形上的有效位移：
`Cost_SNR = ‖ m_t / (√(v_t) + ε) ‖`

### 3.2 路径 B: 信息论视角 (KL 散度)

定义 `Cost_IT` 为参数更新引起的局部 KL 散度（信息增益）：
`Cost_IT ≈ 0.5 ⋅ η² ⋅ Σ m_t²`
量纲为 **Nats**，与路由器的对数概率空间对齐，符合 MDL (最小描述长度) 原则。

## 4. 统一路由塑造公式

[`LearningDynamics`](exp/arc_dyntrm/train.py:103) 中的路由塑造逻辑重构为：
`𝒢 = Importance ⋅ (Belief - α ⋅ Cost_FARS)`

- `Importance`: 专家输出梯度范数 `‖ ∇_out ℒ ‖`。
- `Belief`: 路由器的原始 Logits。
- `Cost_FARS`: 选定的成本度量 (`Cost_SNR` 或 `Cost_IT`)。
- `α`: 复杂度惩罚系数。

## 5. 几何本质: 切空间对齐

FARS 迫使路由器将数据分发至局部几何平坦的专家，实现非线性流形在当前数据点附近的切平面与数据分布重合。

- **高 Cost**: 参数更新剧烈，局部曲率大 (High Rank)，过拟合风险高。
- **低 Cost**: 参数更新微小，局部几何平坦 (Low Rank)，泛化能力强。

## 6. ARC 离散域假设

在 [`ARC-AGI`](data/arc-agi_evaluation_challenges.json) 任务中，曲率对应程序的 **Kolmogorov 复杂度**：

- **平坦区域**: 可通过低复杂度程序解释的样本。
- **弯曲区域**: 需高复杂度特例解释的样本。

FARS 作为 MDL 过滤器，优先保留符合最小描述长度原则的程序路径。
