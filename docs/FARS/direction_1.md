# FARS 研究报告：方向一 - 基于 Adam Step 的无量纲化

## 1. 理论背景 (Theoretical Context)

在 `Tiny-ONN` 的自组织架构中，路由器的任务是最小化系统的变分自由能 (VFE)。现有的 **SARS (Surprise-Aware Routing Shaping)** 机制通过瞬时梯度范数来衡量专家的“惊奇度”，并以此塑造路由先验。然而，瞬时梯度存在严重的量纲冲突和高频噪声问题。

**FARS (Fisher-Aware Routing Shaping)** 提出利用 Adam 优化器的二阶统计量来重构这一过程。

## 2. 形式化推导 (Formal Derivation)

### 2.1 无量纲性证明

设 $g_t$ 为梯度，$m_t$ 为一阶矩（动量），$v_t$ 为二阶矩。
量纲关系如下：

- $[m_t] = [g_t]$
- $[v_t] = [g_t]^2$

定义 FARS Cost 为 Adam 步长的范数：
$$\text{Cost} = \left\| \frac{m_t}{\sqrt{v_t} + \epsilon} \right\|$$
其量纲为：
$$\frac{[g_t]}{\sqrt{[g_t]^2}} = 1$$
该结果证明了 FARS Cost 是**无量纲**的，可以直接与路由器的 Logits 进行数学运算。

### 2.2 信息几何解释

Adam 的二阶矩 $v_t$ 是 Fisher 信息矩阵 (FIM) 的对角近似。在 Fisher 流形上，参数更新的长度代表了模型结构的“复杂度变化”。FARS Cost 实际上衡量了专家为了适应当前输入而进行的**自然梯度位移**。

## 3. 文献综述 (Literature Review)

1. **Adam SNR 解释** (Kingma & Ba, 2015): 确立了 $m_t/\sqrt{v_t}$ 作为信噪比的地位，证明了其在不确定性环境下的稳健性。
2. **AdaFisher** (Martins Gomes et al., ICLR 2025): 证明了利用 Fisher 信息进行梯度预白化（Pre-whitening）能显著提升二阶优化的收敛稳定性。
3. **MoE Routing Stability** (Nguyen et al., 2025): 指出利用平滑的统计量替代瞬时信号是解决 MoE 路由波动的关键。

## 4. 实施建议 (Implementation)

建议在 `LearningDynamics` 类中引入对优化器状态的访问，将 `norm_mu_grad` 替换为基于 Adam 状态的 `fars_cost`：

```python
# fars_cost = norm(m_t / (sqrt(v_t) + eps))
goodness_logits = norm_output_grad * (norm_logits - fars_cost)
```

## 5. 结论

FARS 方向一通过无量纲化解决了 SARS 的理论缺陷，并利用 Fisher 信息近似为路由决策提供了几何层面的确定性依据。
