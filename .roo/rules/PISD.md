# PISD: Predictive Integrity Self-Distillation

## 0. 理论

PISD 的核心目标是将神经网络的学习范式从"外部数据对参数流形的强行拖拽"重构为"基于参数内部几何结构的自洽对齐"。

在标准交叉熵 (CE) 训练范式下，优化目标在概率单纯形上表现为一个零熵的 **Dirac 单点 (One-hot 顶点)**。无论底层优化器如何在参数流形上平滑滑行，其"吸引子"依然是绝对排他的。这种单点吸引往往产生强烈的几何剪切力，物理性地擦除参数空间中已沉淀但与当前硬标签存在微小偏移的"协同逻辑结构"。

PISD 介入于 `forward` 与参数更新之间，将标准梯度下降转化为在 **几何信任域 (Geometric Trust Region)** 内的分布流形对齐。

---

## 1. Ouroboros 循环：执行流

PISD 的执行链路嵌套在训练迭代中，与 [`ARS2Neo.step()`](ref/ARS/optimizer/ars2_neo.py:126) 的闭包双次前向流程协同工作：

1. **自探测 (Self-Probe)**：
   执行第一次前向传播（对应 [`closure()`](ref/ARS/optimizer/ars2_neo.py:142) 首次调用），计算当前输出分布 `p_θ(x)`。该分布是参数流形 `θ` 内部"隐含直觉"的快照。

2. **几何反射 (Geometric Reflection)**：
   ARS2-Neo 的 SAM 机制执行 `loss.backward()` 获取基础梯度 `g_base`，并在参数空间施加扰动 `ρ·ĝ_nat`（见 [`state['last_perturbation']`](ref/ARS/optimizer/ars2_neo.py:212)）。阻断此时的参数更新，转而从优化器状态中提取流形的方向信息与能量标度。

3. **完整性审计 (Integrity Audit)**：
   计算预测完整性 (Predictive Integrity, PI)，量化本次更新对系统内部逻辑拓扑的潜在破坏力：

   `PI = exp(−α · Inaccuracy − γ · C_geo)`

   - `Inaccuracy`：外部硬证据 `y` 与内部直觉 `p_θ(x)` 的分歧度。
   - `C_geo`：几何更新代价，严格映射到优化器的可观测物理状态量（见第 3 节）。

4. **流形对齐 (Manifold Alignment)**：
   将单纯形上的绝对目标 `y` 扩散为由 PI 半径约束的 **信息几何信任域**。
   优化目标发生相变：在满足 PI 边界约束的邻域内，寻找与 `p_θ(x)` 散度最小的自洽软目标 `q_mix`，从而实现损失空间的平滑化。

---

## 2. 形式化定义与几何对偶

### 2.1 分布目标重构

引入由 PI 决定的动态平滑分布 `q_mix`：

`q_mix = (1 − λ_PI) · y_smooth + λ_PI · p_θ(x).detach()`

门控变量 `λ_PI` 严格依赖于审计结果 PI：

- **高完整性态 (PI ↑)**：参数流形展现出稳定的逻辑先验，放大 `λ_PI`，将目标拉向模型自身的先验分布，保护现有认知结构的拓扑完整性。
- **低完整性态 (PI ↓)**：内部逻辑尚未成型或遭遇剧烈干涉，减小 `λ_PI`，回归外部监督信号的强行引导。

### 2.2 最终对齐损失

`𝓛_PISD = D_KL(q_mix ‖ p_θ)`

### 2.3 物理对偶性：参数流形 vs. 分布单纯形

PISD 与 [`ARS2Neo`](ref/ARS/optimizer/ars2_neo.py:71) 构成完美的双向对偶约束：

| 层次 | 组件 | 作用空间 | 机制 |
|:---|:---|:---|:---|
| **路径的几何约束** | ARS2-Neo (SAM/ASI) | 参数空间 `θ` | 扰动半径 `ρ` 迫使模型逃离尖锐极小值，寻找平坦"滑行盆地" |
| **终点的拓扑约束** | PISD | 概率分布空间 `Δ` | PI 扩散出的信任域迫使模型逃离零熵 Dirac 引力黑洞，建立自洽"吸引盆地" |

两者共同保证：模型不仅在最平滑的路上滑行，而且驶向具有容错性、保全认知结构的终点。

---

## 3. 与 ARS2-Neo 状态量的工程映射

PISD 并非孤立的正则化项，它深度依赖 [`ARS2Neo.diagnostics`](ref/ARS/optimizer/ars2_neo.py:356) 暴露的只读状态量，以此赋予 `C_geo` 精确的可计算语义。

> **注意**：`energy`（见 [`_apply_ars2_kernel`](ref/ARS/optimizer/ars2_neo.py:393)）与 `v_hat`（见 [`_ars2_update` 内部](ref/ARS/optimizer/ars2_neo.py:203)）均为内核局部变量，**未**暴露在 `diagnostics` 中。PISD 只应通过 `diagnostics` 只读接口获取状态，严禁深入梯度图以避免引入额外计算开销。

| PISD 概念 | ARS2-Neo `diagnostics` 键 | 物理含义 |
|:---|:---|:---|
| **系统级冲突度** | [`phi_t`](ref/ARS/optimizer/ars2_neo.py:360) | 基础梯度与剪切力梯度的余弦对齐度。`phi_t ≪ 0` 表明正在发生剧烈的跨维度特征干涉，`C_geo` 应相应升高。 |
| **冲突噪声标准差** | [`phi_std`](ref/ARS/optimizer/ars2_neo.py:361) | `phi_t` 的 EMA 方差平方根，表征干涉度的历史波动水平，可用于校准 `C_geo` 的基线阈值。 |
| **锐度探针** | [`surrogate_gap`](ref/ARS/optimizer/ars2_neo.py:365) | ASI 机制通过两次闭包调用的损失差分 `loss_adv − loss` 直接探测局部锐度。`surrogate_gap` 越大，说明当前参数邻域越陡峭，PI 应下调以抑制硬拟合。 |
| **锐度 EMA** | [`ema_gap`](ref/ARS/optimizer/ars2_neo.py:366) | `surrogate_gap` 的指数移动平均，提供稳定的历史锐度基线，用于 `C_geo` 的差分判断（`|gap| − ema_gap > 0` 表示锐度上升）。 |
| **动态扰动半径** | [`current_rho`](ref/ARS/optimizer/ars2_neo.py:367) | ASI 反馈调节后的实时 `ρ`，反映优化器对当前损失景观平坦度的自适应评估。`current_rho` 收窄说明系统感知到局部平坦，可适当放松 PI 约束。 |

**`C_geo` 的推荐计算公式**（基于上述状态量的线性组合）：

`C_geo ≈ w₁ · max(0, −phi_t) + w₂ · (|surrogate_gap| / (ema_gap + ε))`

- 第一项捕捉方向干涉强度；第二项捕捉相对锐度增长率。

---

## 4. 边界条件与失效模式

1. **观测滞后 (Stale Observation)**：`diagnostics` 中的状态量是上一步的滞后值，在快速动态场景下存在单步偏差。建议引入 EMA 缓冲（如使用 `phi_std` 替代瞬时 `phi_t`）以平滑审计信号。

2. **单纯形塌陷 (Simplex Collapse)**：若阈值设计缺陷导致 `λ_PI → 1`，系统将陷入"认知茧房"，完全排斥外部新知，表现为模型重播自身的幻觉。必须为 `λ_PI` 设定硬上界（如 `λ_max = 0.9`）。

3. **架构正交性 (Architectural Orthogonality)**：PISD 的工程注入不得干涉 [`ARS2Neo`](ref/ARS/optimizer/ars2_neo.py:71) 既有的能量-几何解耦计算逻辑。二者仅应通过 `diagnostics` 只读接口发生信息流耦合，严禁保留额外梯度图。

4. **SAM 未启用时的降级**：当 `k=0`（SAM 禁用）时，`surrogate_gap` 恒为 `0.0`，`phi_t` 恒为 `1.0`。此时 `C_geo` 退化为纯噪声，PISD 应自动回退到标准 Label Smoothing 模式（`λ_PI = 0`）。

---

## 5. 架构锚点

| 角色 | 文件 | 说明 |
|:---|:---|:---|
| **状态观测源** | [`ref/ARS/optimizer/ars2_neo.py`](ref/ARS/optimizer/ars2_neo.py:356) | `diagnostics` 属性暴露 `phi_t`, `phi_std`, `surrogate_gap`, `ema_gap`, `current_rho` 等只读指标 |
| **审计控制逻辑** | [`src/models/dynsiha/shared/router.py`](src/models/dynsiha/shared/router.py) | 路由时的完整性评估入口，`C_geo` 计算应在此注入 |
| **动态目标注入** | [`src/models/dynsiha/recursive/modeling_recursive_dynsiha.py`](src/models/dynsiha/recursive/modeling_recursive_dynsiha.py) | `𝓛_PISD = D_KL(q_mix ‖ p_θ)` 的核心计算位置 |
