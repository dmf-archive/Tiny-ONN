# FARS (Fisher-Aware Routing Shaping): 理论重构与设计规范

`Date: 2026-01-03`
`Status: Refined Theory / Design Specification`
`Version: 2.0`

## 1. 背景：从 SARS 的局限性到 FARS 的必然

**SARS (Surprise-Aware Routing Shaping)** 确立了“基于学习成本塑造路由”的范式。然而，在 [`exp/arc_dyntrm/train.py`](exp/arc_dyntrm/train.py:97) 的初步实现中，我们发现了两个根本性挑战：

1. **量纲冲突 (Dimensional Inconsistency)**：
   - `logits` 是归一化的概率分布（无量纲）。
   - `norm_mu_grad` 是参数梯度的 Frobenius 范数（具有物理量纲）。
   - 在数学上，将概率对数与具有量纲的物理量直接相减是“非法”的，这导致超参数 `w_meta` 极难调节且缺乏物理意义。
2. **高频噪声 (High-Frequency Noise)**：
   - 一阶瞬时梯度 $\nabla L$ 包含大量随机噪声。
   - 直接使用瞬时梯度驱动路由会导致决策在训练初期剧烈抖动，破坏了专家特化的稳定性。

**FARS** 旨在通过**二阶统计量（Fisher 信息近似）**解决上述问题，将路由从“瞬时惊奇”驱动转向“长期价值”驱动。

## 2. 核心洞察：二阶矩即 Fisher 近似

在 **AdaRMSuon** 和 **ARS** 的开发中，我们确立了关键观点：
**Adam 的二阶矩估计 $v_t$ 是 Fisher 信息矩阵 (FIM) 的对角近似。**

$$F(\theta) \approx E[\nabla L \otimes \nabla L] \approx v_t$$

Fisher 信息衡量了参数空间中概率分布对参数变化的敏感度。在信息几何中，它定义了流形的度量。FARS 利用这一度量来衡量专家的“认知代价”。

## 3. 实施路径：双重形式化

基于研究报告 [`docs/FARS/direction_1.md`](docs/FARS/direction_1.md) 和 [`docs/FARS/direction_2.md`](docs/FARS/direction_2.md)，FARS 提供两种互补的 Cost 定义方式：

### 3.1 路径 A：基于 Adam Step 的无量纲化 (SNR 视角)

将 Cost 定义为 Adam 更新步长的范数，代表专家在流形上的“有效位移”：

$$\text{Cost}_{SNR} = \left\| \frac{m_t}{\sqrt{v_t} + \epsilon} \right\|$$

- **优点**：绝对无量纲（Dimensionless），天然降噪。
- **物理意义**：衡量专家为了拟合当前数据所付出的“努力程度”。

### 3.2 路径 B：基于 KL 散度的概率对齐 (信息论视角)

将 Cost 定义为参数更新引起的局部 KL 散度，即信息增益：

$$\text{Cost}_{IT} \approx \frac{1}{2} \eta^2 \sum m_t^2$$

- **优点**：量纲为 **纳特 (Nats)**，与路由器的对数概率空间完美对齐。
- **物理意义**：衡量激活该专家所引入的“模型复杂度增量”，符合 **MDL (最小描述长度)** 原则。

## 4. 统一路由塑造公式

在 FARS 框架下，[`LearningDynamics`](exp/arc_dyntrm/train.py:103) 中的路由塑造逻辑应重构为：

$$\mathcal{G} = \text{Importance} \cdot (\text{Belief} - \alpha \cdot \text{Cost}_{FARS})$$

其中：

- **$\text{Importance}$**: 专家输出梯度范数（价值）。
- **$\text{Belief}$**: 路由器的原始 Logits。
- **$\text{Cost}_{FARS}$**: 选定的 FARS 成本度量（SNR 或 IT）。
- **$\alpha$**: 复杂度惩罚系数（在 IT 路径下具有明确的信息论含义）。

## 5. 结论：从“意外”到“价值”

FARS 标志着 `Tiny-ONN` 路由机制的成熟：

- **SARS** 关注的是“哪里错了”（一阶梯度）。
- **FARS** 关注的是“哪里重要且高效”（二阶 Fisher 信息）。

通过将优化器状态（Momentum, Variance）引入架构决策，我们实现了优化算法与网络拓扑在信息几何层面的深度协同。

---

**参考文档：**

- [方向一：无量纲化报告](docs/FARS/direction_1.md)
- [方向二：信息论报告](docs/FARS/direction_2.md)
- [SARS 原始实现](exp/arc_dyntrm/train.py)

## 6. 附录：FARS 的几何本质——切空间对齐与低秩化

`Added: 2026-01-03`
`Context: Discussion on Linear Routing vs. Nonlinear Experts`

### 6.1 核心洞察：迫使非线性对齐线性

在 DynSIHA 架构中，存在一个看似矛盾的设计：

- **路由器 (CAPR)**：本质上是线性的（点积注意力）。
- **专家 ($\mu$)**：本质上是非线性的（MLP/SiLU）。

FARS 的作用不仅仅是正则化，它实际上是在执行一种**切空间对齐 (Tangent Space Alignment)**：
通过惩罚高 Cost（即参数更新剧烈、曲率大）的专家，FARS 迫使路由器只将数据分发给那些**局部几何平坦**的专家。

这意味着：**FARS 迫使非线性流形（专家能力）在当前数据点附近的切平面（路由决策）与数据分布重合。**

### 6.2 低秩化与泛化

- **Cost 高** $\rightarrow$ 需要复杂的参数更新来拟合 $\rightarrow$ 局部曲率大 (High Rank) $\rightarrow$ **过拟合风险**。
- **Cost 低** $\rightarrow$ 参数更新微小 $\rightarrow$ 局部几何平坦 (Low Rank) $\rightarrow$ **泛化能力强**。

因此，FARS 方向二 (KL/IT) 实际上是一个**线性化测试器**。它告诉路由器：“只信任那些你能线性预测其行为的专家。”

## 7. 思考：离散域 (ARC) 中的流形假设

`Status: Hypothesis`

### 7.1 离散空间的“曲率”

ARC 任务是高度离散的符号逻辑，传统的微分几何流形假设在此是否失效？

**猜想**：在 ARC 中，“曲率”对应的是**程序的柯尔莫哥洛夫复杂度 (Kolmogorov Complexity)**。

- **平坦区域**：可以通过简单的规则（低复杂度程序）解释的样本。
- **弯曲区域**：需要大量 `if-else` 特例（高复杂度程序）才能解释的样本。

### 7.2 FARS 作为 MDL 过滤器

在离散域中，FARS 方向二 (Cost $\propto \sum m_t^2$) 依然有效，因为它惩罚的是**为了解释样本而引入的信息增量**。

- 如果一个专家为了处理某个 ARC 样本，需要大幅修改其参数（即“背诵”该样本的特例），Cost 会激增。
- FARS 会切断该路由，迫使模型寻找通用的、不需要大幅修改参数就能处理该样本的专家（即找到了共享的底层规则）。

**结论**：在 ARC 中，FARS 实际上是在执行**软性的程序搜索 (Soft Program Search)**，优先保留那些符合最小描述长度 (MDL) 原则的“程序路径”。
