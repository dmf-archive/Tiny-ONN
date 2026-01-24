# SARS: Surprise-Aware Routing Shaping (Archived)

> **注意**: 本文档已弃用并归档。SARS 机制由于其量纲冲突、高频噪声以及二次链式法则的性能问题，已被 **FARS (Fisher-Aware Routing Shaping)** 取代。

---

## 1. Meta-Learning: Surprise-Aware Routing Shaping (SARS)

SARS 是驱动 DynSIHA 自组织的核心动力学。它不是一个显式的 Loss 函数，而是一种元学习机制。

### 1.1 Bayesian Inversion

SARS 的核心目标是优化模型的 Implicit Prior `p(z)`（即 Router 的内在偏好）。
根据贝叶斯定理：
`P(z|x) = P(x|z)P(z) / P(x)`
取对数并移项，我们得到优化的核心关系式：
`log P(z) ∝ log P(z|x) - log P(x|z)`

- Posterior `log P(z|x)`: 模型在看到数据 `x` 后，实际激活了哪些模块。由 `routing_logits` 表示。
- Likelihood `log P(x|z)`: 模块 `z` 解释数据 `x` 的能力。这是未知的，需要估计。
- Prior `log P(z)`: 我们希望优化的目标。

### 1.2 The Goodness Function (Proxy for Likelihood)

如何衡量“模块 `z` 解释数据 `x` 的能力”？
根据 FEP，好的解释意味着低的惊奇 (Low Surprise)，或者说 低的学习成本 (Low Complexity Cost)。

我们定义 Goodness 为 Likelihood 的工程代理：
`Goodness ≈ Importance × (Belief - Cost)`

具体实现公式：
`𝒢 = ‖ ∇ₒᵤₜ ℒ ‖ ⋅ ( Norm(logits) - ‖ ∇ₒᵤₜ ℒ ‖ )`

- `‖ ∇ᵧ ℒ ‖` (Parameter Gradient Norm): 代表 Cost。如果激活模块 `z` 导致参数梯度很大，说明该模块“不懂”这个数据，需要大幅修改参数才能拟合，因此 Likelihood 低。
- `Norm(logits)` (Belief): 代表 Posterior Belief。模型当前的信心。
- `‖ ∇ₒᵤₜ ℒ ‖` (Output Gradient Norm): 代表 Importance。该模块的输出对最终 Loss 有多大影响。

### 1.3 Shaping the Prior

通过最小化 Prior 的熵，我们驱动 Router 形成明确的偏好：
`ℒₘₑₜₐ = Entropy( Softmax( logits - 𝒢.detach() ) )`
这迫使 Router 将概率集中在那些 既重要 (High Importance) 又 熟练 (Low Cost) 的专家上。
