# PI 加权自蒸馏最小持续学习实验 Spike

## 摘要

- **目标**: 在固定小型 Transformer 架构下，构造 `mod-add → mod-sub` 的顺序学习实验，验证 PI 作为 `ground truth ↔ teacher distribution` 混合权重的理论合理性与经验可行性。
- **当前结论**: 第一轮原型没有验证出 PI 优势，但失败主因是 `λ_PI` 数值塌陷，而不是 PI 原理被否定。
- **当前决策**: 第二轮已完成一次修正与复验。PI 组已脱离 `λ≈0` 的数值塌陷，但当前 PI 代理仍未优于固定系数或熵门控，下一步应聚焦于“PI 与 confidence 的可分离性”而不是继续堆训练轮数。

## Theory Brief

### Problem Statement

给定两阶段顺序任务：

1. Task A: `y = (a + b) mod p`
2. Task B: `y = (a - b) mod p`

输入空间保持一致，模型架构保持一致，唯一允许变化的是 Task B 的学习目标。

### Formal Specification

设：

- `y` 为 ground truth one-hot 分布
- `q_teacher` 为 Task A 结束后冻结 teacher 在 Task B 输入上的输出分布
- `λ(x) ∈ [0, 1]` 为样本级蒸馏权重

则 Task B 的目标分布为：

`q_mix(x) = (1 - λ(x)) · y(x) + λ(x) · q_teacher(x)`

训练目标为：

`L_B = E_x [ CE(p_theta(x), q_mix(x)) ]`

其中 PI 加权的设计目标不是最大化 teacher 使用率，而是满足以下条件：

1. 当 teacher 可靠且结构代价低时，`λ_PI` 应显著大于 0。
2. 当 teacher 明显错误或结构代价高时，`λ_PI` 应下降。
3. `λ_PI` 不得在训练早期系统性塌陷到 0 附近，否则 PI 组将退化为纯 CE。

### Invariants

1. **架构不变量**: 所有对照组使用同一模型、同一数据协议、同一优化器主设置。
2. **混合不变量**: `∀x, q_mix(x)` 必须是合法概率分布。
3. **门控不变量**: `λ(x) ∈ [0, 1]`。
4. **可区分性不变量**: PI 组的平均权重与分布必须与 CE 组、固定系数组可区分，否则实验无法检验门控假设。

### Proof Sketch

若 `λ_PI ≈ 0` 对大多数样本成立，则：

`q_mix ≈ y`

从而 PI 组目标函数收敛为纯 ground-truth 监督，其训练动力学与 CE 组等价。故只要观察到 `lambda_mean → 0`，就不能把实验失败归因于 PI 理论本身，只能归因于门控函数的数值参数化失败。

## 第一轮实验与失效诊断

结果文件：

- [`experiment-pi-weighted-self-distillation-spike-results.md`](docs/spikes/experiment-pi-weighted-self-distillation-spike-results.md)
- [`experiment-pi-weighted-self-distillation-spike-results.json`](docs/spikes/experiment-pi-weighted-self-distillation-spike-results.json)

### 已观测事实

1. `teacher_only` 几乎完全保留 Task A，但几乎学不会 Task B。
2. `pi_weighted_kd` 与 `ce_only` 几乎完全重合。
3. `pi_weighted_kd.lambda_mean = 0.003846`，说明 PI 门控几乎完全关闭。
4. Task A 切换前精度仅约 `0.48`，teacher 本身仍不够成熟。

### 失效原因

第一轮 PI 代理采用：

`PI = exp(-α · (Inaccuracy + γ · Complexity))`

其问题不是方向错误，而是尺度过激：

- `α = 4.0` 过大
- teacher 早期 `Inaccuracy` 偏高
- `Complexity` 进一步放大惩罚
- 指数映射导致 `PI → 0`

于是：

`λ_PI → 0 => q_mix → y`

最终 PI 组退化为 CE 组。

## 第二轮修正原则

### 原则 1: 先保证 PI 组不是数值死路

PI 门控函数必须先满足“有效参与训练”，再讨论是否最优。

因此第二轮应优先保证：

- `lambda_mean` 明显大于 0
- `lambda_std` 非零
- `lambda_p50` 不贴近 0

### 原则 2: teacher 质量必须先进入可用区间

若 teacher 本身未学稳，则 `Inaccuracy` 将系统性偏高，PI 不可能正常工作。

因此第二轮应优先：

- 延长 Task A 训练
- 必要时减小任务规模或稍增模型容量
- 保证 teacher 至少达到“可作为软目标”的基础精度

### 原则 3: 保留最小变量集

第二轮仍然只改三类变量：

1. `λ_PI` 映射方式
2. Task A teacher 质量
3. 权重统计输出

暂不增加 OT、EMA teacher、多模型对照等新变量。

## 第二轮实验执行与结果

### 固定条件

- 同一简化 Transformer 架构
- 同一 `mod-add → mod-sub` 顺序任务
- 同一 6 组学习目标对照

### 已执行修改

1. **修正 PI 映射**
   - 将 `pi_alpha` 从激进指数衰减改为温和幂次缩放
   - 为 `λ_PI` 引入下界，避免权重彻底塌陷
   - 保留 `Inaccuracy + Complexity` 分解，但不再让指数直接把 PI 压到 0

2. **提高 teacher 质量**
   - 增加 Task A 训练轮数
   - 增大模型宽度
   - 将任务规模调整到更小的 `p=17`，并提升 `train_frac=0.8`

3. **补充统计**
   - 输出 `lambda_std`
   - 输出 `lambda_min/max`
   - 输出 `lambda_p10/p50/p90`

### 第二轮配置

第二轮主实验采用：

- `p = 17`
- `train_frac = 0.8`
- `epochs_a = 200`
- `epochs_b = 80`
- `d_model = 128`
- `d_mlp = 256`

### 第二轮结果摘要

结果文件仍为：

- [`experiment-pi-weighted-self-distillation-spike-results.md`](docs/spikes/experiment-pi-weighted-self-distillation-spike-results.md)
- [`experiment-pi-weighted-self-distillation-spike-results.json`](docs/spikes/experiment-pi-weighted-self-distillation-spike-results.json)

关键结果：

1. `teacher_only` 仍然是旧任务保持最强的组：
   - `task_a_acc_after_b = 0.706897`
   - `task_b_acc = 0.068966`

2. `pi_weighted_kd` 已经**不再数值塌陷**：
   - `lambda_mean = 0.489923`
   - `lambda_p50 = 0.456337`
   - `lambda_p90 = 0.714044`

3. 但 `pi_weighted_kd` 的任务表现仍未优于混合基线：
   - `task_a_acc_after_b = 0.396552`
   - `task_b_acc = 0.000000`

4. 在混合蒸馏组中，当前最强的是 `fixed_kd`：
   - `task_a_acc_after_b = 0.431034`
   - `task_b_acc = 0.017241`

### 第二轮结论

第二轮已经回答了一个关键问题：

**第一轮 PI 组失败，确实主要是数值塌陷问题。**

因为现在 `λ_PI` 已进入有效区间，但 PI 组仍未胜出。这表明当前剩余问题已经从“PI 是否被关死”转移为：

1. 当前 PI 代理是否真正比 confidence / entropy 携带更多有效信息。
2. 当前顺序任务是否过度偏向“保旧”而压制“学新”。
3. 当前 teacher 分布是否更适合作为保守正则，而不是迁移信号。

### 新的失效诊断

第二轮最重要的新发现不是 PI 胜出，而是：

**PI 代理已经从“塌陷到 CE”转变为“近似退化为一种温和的 confidence-like 门控”。**

证据在于：

- `pi_weighted_kd` 与 `confidence_kd` 的后验表现几乎重合
- 两者的 `lambda_p50` 与 `lambda_p90` 也处于同一量级

这说明当前 `Inaccuracy` 与 `Complexity` 的组合，尚未在实验上形成比简单 confidence 更强的归纳偏置。

## 第三轮修正方向

### 方向 1: 恢复 PI 与 confidence 的可分离性

下一轮不应继续单纯提高训练轮数，而应优先让 PI 不再退化为 confidence 的平滑变体。可行方向包括：

- 将 `Inaccuracy` 从“teacher 最大概率”改回与任务相关、但不直接等于 one-hot label 匹配的形式
- 引入 teacher 与 student 的分歧项，刻画“结构冲突”而非只看 teacher 自信度
- 让 `Complexity` 反映更稳定的分布代价，而不仅是单步熵

### 方向 2: 显式区分“保旧能力”与“学新能力”

当前任务对保旧非常敏感，但对学新不友好。下一轮应补充：

- Task B 学习曲线
- 两阶段平均风险而不只是终点精度
- teacher 对 Task B 的错误结构统计

### 方向 3: 保持最小实验哲学

仍然不引入 OT、EMA teacher、外部 teacher 或多架构比较。第三轮只允许修改 PI 代理定义与统计口径。

## 成功标准

若第三轮实验满足以下任一条件，即视为成功推进：

1. `pi_weighted_kd.lambda_mean` 进入非退化区间，且不再与 CE 组重合。
2. `pi_weighted_kd` 在相近 Task B 精度下，比固定系数或熵门控遗忘更少。
3. `pi_weighted_kd` 在相近旧任务保持下，比固定系数或置信度门控学得更多 Task B。

## 建议

后续实现必须遵循以下顺序：

1. 保留当前第二轮配置作为新的基线。
2. 下一步只修改 [`compute_lambda()`](exp/pi_weighted_self_distillation_spike.py:128) 中 PI 代理的定义，不再大幅改动训练规模。
3. 将第三轮目标明确设为“让 PI 与 confidence 可分离”，而不是单纯追求更大 `λ_PI`。

## 状态历史

| 日期 | 状态 | 备注 |
| --- | --- | --- |
| 2026-03-16 | 🟡 进行中 | Spike 已完成第二轮修正与复验，当前进入第三轮 PI 代理修正阶段 |
