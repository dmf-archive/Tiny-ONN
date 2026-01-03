# FARS 研究报告：方向二 - 基于 KL 散度的概率视角 (Information-Theoretic Cost via KL Divergence)

## 1. 理论背景 (Theoretical Context)

在 **IPWT 2.0 (Integrated Predictive Workspace Theory)** 框架下，认知过程被形式化为最小化变分自由能 (VFE)。根据最小描述长度 (MDL) 原则，系统的总目标是寻找对数据最简洁的解释。

**FARS 方向二** 试图将路由器的“学习成本” (Learning Cost) 从物理位移（方向一）提升到**信息增益**的层面。其核心思想是：激活一个专家不仅产生计算开销，更重要的是它增加了模型表征的**复杂度 (Complexity)**。

## 2. 形式化推导 (Formal Derivation)

### 2.1 从 KL 散度到 Fisher 信息

设参数的概率分布为 $p(x|\theta)$。当参数从 $\theta$ 更新为 $\theta + \Delta\theta$ 时，分布之间的差异可以用 Kullback-Leibler (KL) 散度衡量。

根据二阶泰勒展开，局部 KL 散度与 Fisher 信息矩阵 (FIM) $F(\theta)$ 存在如下关系：
$$D_{KL}(p(x|\theta + \Delta\theta) \| p(x|\theta)) \approx \frac{1}{2} \Delta\theta^T F(\theta) \Delta\theta$$

在 `Tiny-ONN` 使用的 Adam/RMSuon 优化器中，二阶矩 $v_t$ 是 FIM 对角线的移动平均估计：
$$F(\theta) \approx \text{diag}(v_t)$$

### 2.2 定义信息论 Cost

我们将单次更新产生的“信息成本”定义为该步更新引起的分布变化量：
$$\text{Cost}_{IT} = \sum_i \frac{1}{2} v_{t,i} (\Delta\theta_{t,i})^2$$

代入 Adam 的更新步长 $\Delta\theta_{t,i} = \eta \frac{m_{t,i}}{\sqrt{v_{t,i}} + \epsilon}$：
$$\text{Cost}_{IT} \approx \frac{1}{2} \eta^2 \sum_i v_{t,i} \left( \frac{m_{t,i}}{\sqrt{v_{t,i}}} \right)^2 = \frac{1}{2} \eta^2 \sum_i m_{t,i}^2$$

**结论**：在 Fisher 流形上，基于 KL 散度的信息成本在数值上等价于**动量平方和的缩放**。

### 2.3 无量纲性与信息单位

- **量纲分析**：$m_t$ 与梯度同量纲，其平方和需要通过适当的归一化（如除以参数总数或利用学习率平方抵消）来达到无量纲化。
- **单位**：该 Cost 的自然单位是 **纳特 (Nats)**。由于路由器的 Logits 本质上也是对数概率空间（Nats），两者在理论上实现了完美的**量纲对齐**。

## 3. 文献综述 (Literature Review)

1. **Deep Variational Information Bottleneck** (Alemi et al., 2016):

   - **核心摘要**：提出利用 KL 散度作为正则项来压缩中间表示 $Z$ 与输入 $X$ 之间的互信息 $I(X;Z)$。
   - **关联**：FARS 方向二实际上是在参数空间（而非激活空间）应用信息瓶颈原则，限制专家权重的“信息泄露”。
   - **URL**: [https://arxiv.org/abs/1612.00410](https://arxiv.org/abs/1612.00410)

2. **PAC-Bayes Information Bottleneck** (Wang et al., ICLR 2022):

   - **核心摘要**：证明了权重中的信息压缩 (Information in Weights, IIW) 是决定神经网络泛化能力的关键。
   - **关联**：该研究为“将 KL 散度作为路由成本”提供了坚实的泛化理论支撑——即更简单的专家组合具有更好的 PAC-Bayes 泛化界。
   - **URL**: [https://openreview.net/forum?id=iLHOIDsPv1P](https://openreview.net/forum?id=iLHOIDsPv1P)

3. **Variational Neural Information Bottleneck for MoE** (Conceptual Extension):
   - **核心摘要**：在混合专家模型中，通过最小化 $D_{KL}(Q(\theta)\|P(\theta))$ 来防止专家过拟合特定样本。
   - **关联**：这正是 FARS 试图在动态路由层面实现的自组织约束。

## 4. 量化比较 (Quantitative Comparison)

| 特性           | 方向一 (Adam Step SNR) | 方向二 (KL Divergence)          |
| :------------- | :--------------------- | :------------------------------ |
| **理论基础**   | 黎曼几何 / 测地线位移  | 信息论 / 变分推断 (VI)          |
| **计算公式**   | $\|m_t / \sqrt{v_t}\|$ | $\frac{1}{2} \eta^2 \sum m_t^2$ |
| **物理意义**   | 专家更新的“努力程度”   | 专家引入的“复杂度增量”          |
| **量纲**       | 绝对无量纲 (1)         | 概率空间对齐 (Nats)             |
| **过拟合抑制** | 侧重于抑制“剧烈波动”   | 侧重于抑制“信息冗余”            |
| **计算开销**   | 极低（逐元素除法）     | 极低（平方累加）                |

## 5. 实施建议 (Implementation)

建议在 `LearningDynamics` 中实现基于 KL 散度的 Cost 估算器：

```python
def compute_fars_direction_2(momentum, lr):
    # Cost = 0.5 * lr^2 * sum(m^2)
    # 实际上为了数值稳定，通常取均值并进行对数缩放
    info_gain = 0.5 * (lr ** 2) * torch.mean(momentum ** 2)
    return torch.log(info_gain + 1e-8)
```

在路由塑造中：
`goodness_logits = importance * (belief - info_gain_cost)`

## 6. 结论

FARS 方向二通过 KL 散度将路由决策转化为一个**变分信息瓶颈 (VIB)** 问题。它不仅解决了 SARS 的量纲问题，更从第一性原理出发，将“泛化”直接编码进了路由器的目标函数中。相比方向一，它在处理“过参数化”专家时具有更强的理论约束力。
