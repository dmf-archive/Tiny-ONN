# PISD：Predictive Integrity Self-Distillation

## 0. 原理概述

PISD（预测完整性自蒸馏）将训练范式从硬标签对齐重构为基于内部几何自洽性的分布对齐：

- **目标平滑**：在概率单纯形 $\Delta$ 上将 Dirac 目标（One-hot）扩散为由预测完整性（PI）约束的软目标。
- **几何对偶**：SAM 在参数空间 $\Theta$ 执行物理位移（扩散算子），PISD 在分布空间 $\Delta$ 执行目标混合（扩散算子）。
- **冲突对冲**：通过下调剧烈几何冲突步的外部证据权重，保护已沉淀的协同逻辑结构。

## 1. 核心议题

### 1.1 空间对偶性假设

PISD 可视为 SAM 的对偶算子。SAM 通过探测参数邻域锐度来改写训练路径，而 PISD 通过探测预测完整性来改写训练目标。在信息几何层面，对目标的扩散（模糊化）在效果上等价于损失景观的平坦化。

### 1.2 无采样贝叶斯对齐

PISD 试图在不引入大规模采样或变分推断的情况下，实现类似贝叶斯方法的规则自洽性验证。它利用优化器的二阶信息作为“不确定性”的代理信号。

### 1.3 持续学习中的 p ∩ q 问题

在非平稳数据流中，PISD 扮演“认知保护”角色。当外部证据 $y$ 与内部先验 $p_\theta$ 产生剧烈系统级冲突时，PI 审计会下调 $y$ 的引力，防止旧有的协同参数组被物理性擦除。

## 2. 逻辑链路

### 2.1 预测完整性（PI）审计

量化当前更新对系统内部逻辑拓扑的潜在破坏力：

`PI = exp(−α · Inaccuracy − γ · C_geo)`

- `Inaccuracy`：$D_{KL}(p_\theta || y)$，内部直觉与外部证据的分歧。
- `C_geo`：几何更新代价，基于 ARS2-Neo 的梯度一致性指标。

### 2.2 混合目标构造

`q_mix = (1 − λ_PI) · y + λ_PI · p_θ.detach()`

- **高 PI（稳定态）**：放大 $\lambda_{PI}$，信任内部先验，执行自蒸馏。
- **低 PI（冲突态）**：减小 $\lambda_{PI}$，回归外部强监督。

### 2.3 最终损失

`𝓛_PISD = D_KL(q_mix ‖ p_θ)`

## 3. 与 ARS2-Neo 状态量的工程映射

PISD 依赖 [`ARS2Neo.diagnostics`](ref/ARS/optimizer/ars2_neo.py:356) 暴露的只读状态量计算 `C_geo`：

| PISD 概念 | ARS2-Neo 诊断项 | 物理含义 |
|:---|:---|:---|
| **系统级冲突度** | `phi_t` | 基础梯度与剪切梯度的余弦对齐度。`phi_t << 0` 意味着剧烈干涉。 |
| **冲突噪声水平** | `phi_std` | `phi_t` 的波动水平，用于动态校准 `C_geo` 基线。 |
| **局部锐度探针** | `surrogate_gap` | 两次闭包损失差分。Gap 越大，当前邻域越陡峭，应降低 PI。 |
| **动态扰动半径** | `current_rho` | 反映优化器对景观平坦度的评估。 |

**`C_geo` 推荐实现**：
`C_geo = w1 · relu(-phi_t) + w2 · (abs(surrogate_gap) / (ema_gap + eps))`

## 4. 代码锚点

- 状态观测入口：[`ref/ARS/optimizer/ars2_neo.py`](ref/ARS/optimizer/ars2_neo.py:356)
- 完整性审计逻辑：[`src/models/dynsiha/shared/router.py`](src/models/dynsiha/shared/router.py)
- 动态目标注入：[`src/models/dynsiha/recursive/modeling_recursive_dynsiha.py`](src/models/dynsiha/recursive/modeling_recursive_dynsiha.py)

## 5. 边界

- **SAM 协同**：PISD 不取代 SAM，二者共同构成“路径-终点”的双向约束。
- **自蒸馏过拟合**：若 $\lambda_{PI}$ 过高会导致模型陷入认知茧房。必须设定硬上界（如 $\lambda_{max} = 0.9$）。
- **降级模式**：若未开启 SAM（$k=0$），`surrogate_gap` 失效，PISD 应退化为标准 Label Smoothing。
