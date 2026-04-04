# NGD + 连续化目标软蒸馏：从 one-hot 锚点坍缩到 Fisher 几何对齐

## 摘要

- **主假设**: 当前自蒸馏失败的主因，不是缺少 [`Predictive Integrity Matrix`](docs/spikes/theory-predictive-integrity-matrix-spike.md)，而是监督目标仍以 one-hot ground truth 为主轴，导致 soft teacher 的结构在 [`CE + KL`](exp/pi_weighted_self_distillation_spike.py:249) 混合中被硬锚点压扁。
- **核心判断**: 若 `y = δ_c` 为 one-hot，而 `q_teacher` 为软分布，则线性混合目标 `q_mix = (1 - λ)δ_c + λq_teacher` 在概率单纯形上仍被顶点 `δ_c` 主导；当 `λ` 不足够大、teacher 与标签不一致、或优化器使用欧式梯度时，teacher 的关系结构会在更新中退化为“对正确类的额外平滑”，而不是可保真的分布几何。
- **新主线**: 将问题重构为“连续化 ground truth + Fisher/NGD 更新”的联合设计：先把监督目标从离散点质量扩散为合法软分布，再用自然梯度在概率流形而非欧式参数空间中执行更新。
- **路线决策**: [`PIM`](docs/spikes/theory-predictive-integrity-matrix-spike.md) 仍保留为潜在二阶结构对象，但不再作为当前主线；当前主线应升级为“连续化 ground truth + NGD 几何蒸馏”。

## Theory Brief

### Problem Statement

给定分类或离散 token 预测任务：

- 标签空间 `𝒴 = {1, 2, ..., K}`
- student 输出分布 `p_θ(·|x) ∈ Δ^(K-1)`
- teacher 输出分布 `q_T(·|x) ∈ Δ^(K-1)`
- ground truth 类别 `c(x) ∈ 𝒴`

现有实现采用的目标为：

`q_mix = (1 - λ(x)) · δ_{c(x)} + λ(x) · q_T`

并通过 [`soft_target_ce()`](exp/pi_weighted_self_distillation_spike.py:125) 最小化：

`ℒ_mix(θ) = E_x [ H(q_mix(x), p_θ(x)) ]`

其中 `δ_c` 为 one-hot 分布。

我们要重新回答的问题不是“如何让 `λ_PI` 更聪明”，而是：

1. one-hot ground truth 是否从根本上破坏了软蒸馏的几何结构；
2. 若是，则正确修复对象应是目标分布、更新几何，还是二者同时；
3. `PI`、`NGD`、连续化标签三者应如何分工。

### Domain

- 输入空间: `x ∈ 𝒳`
- 标签空间: `c(x) ∈ 𝒴`
- 目标分布空间: `q(·|x) ∈ Δ^(K-1)`
- 参数空间: `θ ∈ ℝ^d`
- Fisher 几何: `F(θ) = E_x E_{y∼p_θ}[∇_θ log p_θ(y|x) ∇_θ log p_θ(y|x)^T]`

### Specification

我们要求新的训练协议满足：

1. **合法性**: `∀x, q_target(·|x) ∈ Δ^(K-1)`。
2. **连续性**: `q_target` 不退化为 `δ_c`，除非证据明确要求零扩散极限。
3. **结构保真**: teacher 的非标签类概率结构不能仅以“正确类置信度修正”形式出现，而应在目标分布的多个坐标上保留可观测质量。
4. **几何一致性**: 更新方向应近似最小化概率流形上的局部 KL，而不只是参数空间中的欧式步长。

### Invariants

1. `q_target,k(x) ≥ 0` 且 `Σ_k q_target,k(x) = 1`
2. 当 teacher 不可信时，目标仍应收缩到以标签为中心的连续化真值，而不是回退到纯 one-hot 极限
3. `PI(x)` 的职责是控制 teacher 参与资格或扩散尺度，而不是单独承担全部结构表达
4. `NGD` 的职责是改变更新几何，而不是替代目标构造

## 一、现有 [`CE + KL`](exp/pi_weighted_self_distillation_spike.py:249) 混合在优化几何上的真实含义

### 1.1 线性混合的等价形式

对任意 `q_mix = (1 - λ)δ_c + λq_T`，有：

`H(q_mix, p_θ) = (1 - λ)H(δ_c, p_θ) + λH(q_T, p_θ)`

又因为：

- `H(δ_c, p_θ) = CE(c, p_θ) = -log p_θ(c)`
- `H(q_T, p_θ) = KL(q_T || p_θ) + H(q_T)`

故优化上等价于：

`ℒ_mix = (1 - λ) CE(c, p_θ) + λ KL(q_T || p_θ) + const`

因此，用户提出的“`CE(y, p)` 与 `KL(q_teacher || p)` 的线性混合”在当前实现里并不是近似，而是严格等价于“对一个靠近 `δ_c` 的混合目标做交叉熵拟合”。

### 1.2 概率单纯形上的几何解释

在输出分布空间中：

- `δ_c` 是单纯形顶点；
- `q_T` 是单纯形内部一点；
- `q_mix` 位于连接 `δ_c` 与 `q_T` 的线段上。

这意味着：

1. 混合并没有创造新的几何对象；
2. 它只是把目标从顶点沿直线向 teacher 拉回一小段；
3. 当 `λ` 较小，`q_mix` 仍非常接近 `δ_c`。

若 `K` 较大、teacher 的质量分散在多个非标签类上，则大多数结构信号在 `q_mix` 中仅以 `λ q_T(k)` 的量级出现，天然弱于标签坐标上的 `1 - λ + λ q_T(c)`。

### 1.3 梯度层面的含义

设 student logits 为 `z`，`p = softmax(z)`。则：

`∂ℒ_mix/∂z = p - q_mix`

即：

`∂ℒ_mix/∂z_k = p_k - (1 - λ)δ_{k=c} - λ q_T(k)`

对标签类 `c`：

`∂ℒ_mix/∂z_c = p_c - 1 + λ(1 - q_T(c))`

对非标签类 `k ≠ c`：

`∂ℒ_mix/∂z_k = p_k - λ q_T(k)`

这里的关键并非“是否有 teacher 项”，而是**梯度预算极不对称**：

- 标签类获得 `O(1)` 量级的强约束；
- 非标签类仅获得 `O(λ)` 量级修正。

所以在有限训练步、有限容量、以及 teacher 与当前标签冲突的顺序学习情形下，teacher 的结构非常容易被重新解释为“对标签边界的小扰动”，而非独立结构约束。

## 二、teacher 结构何时会被 one-hot 锚点吞没

### 2.1 必要判据

定义 teacher 的非标签结构质量：

`M_T(x) = Σ_{k ≠ c} q_T(k|x)`

定义混合目标中的非标签总质量：

`M_mix(x) = λ(x) · M_T(x)`

若 `M_mix(x) ≪ 1`，则除标签类外，所有结构信号总质量都被压缩到很小区间。此时 student 训练主要感受到的仍是“把质量推回标签类”。

因此 teacher 结构被吞没的充分危险条件是：

`λ(x) · M_T(x) << 1`

在顺序学习场景中，这常常发生，因为：

1. teacher 对新任务往往并不支持标签类；
2. `PI` 或 confidence 门控又会进一步压小 `λ`；
3. 结果是非标签结构被双重缩放。

### 2.2 冲突 teacher 情形

若 teacher 的高概率类 `t* ≠ c`，则混合目标并不是“同时保留二者”，而是在标签类与 teacher 主峰之间做人为凸组合。

当使用欧式优化器时，模型更倾向于选择下降最快的方向；而标签类坐标对应的 `-log p(c)` 通常提供最强、最稳定、最低方差的梯度锚点。于是 teacher 主峰会被当作一种需要压制的“错误偏差”。

这正与当前 [`experiment-pi-weighted-self-distillation-spike-results.md`](docs/spikes/experiment-pi-weighted-self-distillation-spike-results.md:3) 至 [`experiment-pi-weighted-self-distillation-spike-results.md`](docs/spikes/experiment-pi-weighted-self-distillation-spike-results.md:8) 中“混合蒸馏组几乎学不会 Task B”的现象一致：不是只有 `PI` 失效，而是监督几何本身鼓励 student 快速向新标签顶点坍缩。

### 2.3 为什么当前结果支持“监督几何错误”而不只支持“PI 退化”

现有结果表明：

- [`pi_weighted_kd`](docs/spikes/experiment-pi-weighted-self-distillation-spike-results.json:99) 的 `lambda_mean ≈ 0.49`，并未塌到 0；
- 但其 [`task_b_acc`](docs/spikes/experiment-pi-weighted-self-distillation-spike-results.json:102) 仍为 0；
- [`fixed_kd`](docs/spikes/experiment-pi-weighted-self-distillation-spike-results.json:54) 也未明显改善新任务学习。

这说明问题不再只是“权重太小”，而是即便 `λ` 进入有效区间，`q_mix = (1 - λ)δ_c + λq_T` 仍可能把 teacher 结构变成一个不具可学习意义的折中点。

换言之，**失败层级已经从门控问题上升到目标构造问题。**

## 三、“把 ground truth 扩散为高斯分布”在离散 setting 下的严格表述

### 3.1 不应使用含混说法

在离散 token / 分类 setting 中，“把 ground truth 扩散成高斯”不能直接理解为对类别索引做连续高斯核，除非类别空间本身带有可信的度量结构。否则“类别 3 离类别 4 更近于类别 17”并无语义根据。

因此必须区分三种对象：

1. **标签空间连续化**
2. **logit 空间连续化**
3. **嵌入/语义流形上的连续化**

### 3.2 四类候选对象

#### A. Label Smoothing

定义：

`q_ls = (1 - ε)δ_c + εu`

其中 `u` 是均匀分布。

优点：合法、简单、稳定。

缺点：

- 仅提供各向同性扩散；
- 不能表达类间相似性；
- 不能保留 teacher 结构。

因此它只是**连续化基线**，不是最终答案。

#### B. Class-Manifold Kernel Target

若类别存在可定义的语义嵌入 `e_k ∈ ℝ^m`，则可定义：

`q_kernel(k|c, σ) ∝ exp(-||e_k - e_c||² / 2σ²)`

这才是严格意义上的“高斯化目标”。

但前提极强：必须存在可解释的类流形。对模运算分类或一般 token ID，此前提通常不成立。

#### C. Logit-Space Gaussian Target

先定义以标签为中心的目标 logits：

`μ_c ∈ ℝ^K`

再在 logit 空间上加入高斯扰动并取期望 softmax：

`q_logit(k|c, Σ) = E_{ξ∼𝒩(0, Σ)} [ softmax(μ_c + ξ)_k ]`

这是一种更几何化的连续化方式。它保留了“目标是分布，而非单点标签”的思想，并允许协方差 `Σ` 携带结构。

若 `Σ = σ²I`，它是各向同性 logit 扩散；若 `Σ` 由 Fisher 或 teacher 结构决定，则可实现方向敏感的连续化。

#### D. Teacher-Anchored Continuous Ground Truth

这是当前主线最合理的对象。定义一个**连续化真值核** `q_gt^cont`，它既不退回 one-hot，也不直接等于 teacher：

`q_gt^cont = (1 - ρ)δ_c + ρ r_c`

其中 `r_c` 不是均匀分布，而是围绕标签 `c` 构造的扩散参考分布。然后再与 teacher 组合：

`q_target = (1 - λ_PI) q_gt^cont + λ_PI q_T`

这样 teacher 就不再与 one-hot 顶点直接对撞，而是与一个**已经连续化的标签锚点**相结合。

### 3.3 当前阶段的严谨结论

在一般离散分类任务中，最稳妥的术语不是“高斯化标签”本身，而是：

**连续化 ground truth（continuous ground truth）**

其可实现形式按理论强度排序为：

1. label smoothing 基线；
2. teacher-anchored continuous target；
3. logit-space Gaussian target；
4. class-manifold Gaussian kernel target。

当前最小实验应优先采用 1 + 2，必要时再比较 3。

## 四、NGD / Fisher 几何究竟应放在哪一层

### 4.1 三层分工必须拆开

#### 层 1：损失定义层

职责：定义要对齐什么分布，例如：

`ℒ(θ) = E_x [ KL(q_target(·|x) || p_θ(·|x)) ]`

这一层决定监督对象，不决定更新几何。

#### 层 2：目标分布构造层

职责：构造 `q_target`，例如：

`q_target = (1 - λ_PI) q_gt^cont + λ_PI q_T`

这一层决定“真值如何连续化、teacher 如何参与”。

#### 层 3：更新规则层

职责：决定如何在参数空间迈步：

`Δθ_NGD = -η F(θ)^(-1) ∇_θ ℒ`

这一层才属于 `NGD / Fisher geometry`。

### 4.2 结论

因此，`NGD` 不应被放进目标分布定义本身，也不应被误说成一种新的 loss。它属于**更新规则层**。

但 `Fisher` 又会反过来影响目标构造，因为 Fisher 或其代理可用于：

- 设定 logit-space 扩散协方差；
- 设定 teacher 结构保留的方向性权重；
- 在实验中比较 AdamW 与 ARS2-Neo 对结构保持的差异。

所以更精确的说法是：

- `NGD` 本体在更新层；
- `Fisher` 可兼作目标扩散的结构先验。

## 五、PI 在新框架中的正确角色

### 5.1 不应再把 PI 当作唯一混合系数主角

先前 [`self-distilling-note`](.roo/rules/self-distilling-note.md:73) 将 PI 主要放在 `λ_PI` 角色上。这一视角仍有价值，但已不足以解释 one-hot 锚点问题。

### 5.2 新框架中的三种合理职责

#### 职责 A：teacher 资格门控

定义：

`λ_PI(x) = g(PI(x))`

用于回答“teacher 是否应参与”。这是 PI 最保守、最稳健的职责，仍应保留。

#### 职责 B：连续化尺度门控

定义标签扩散强度：

`ρ(x) = h(PI(x))`

解释：若当前样本的 teacher 结构可靠，允许标签锚点更软；若 teacher 结构不可靠，则回到较保守的连续化真值。

#### 职责 C：自由能式正则系数

在更一般的自由能目标中：

`ℒ_total = E_x [ KL(q_target || p_θ) ] + β(PI) · Complexity(p_θ)`

其中 `PI` 控制复杂度惩罚或温度。但这属于下一阶段，不应抢占当前主线。

### 5.3 当前阶段的正式决策

当前应把 `PI` 定位为：

1. **首先是 teacher 资格门控**；
2. **其次是连续化尺度门控**；
3. **暂不作为 PIM 替代物，也暂不作为唯一结构承载体。**

## 六、对 [`PIM`](docs/spikes/theory-predictive-integrity-matrix-spike.md) 路线的状态判定

### 6.1 仍有理论价值的部分

1. `PIM` 正确指出了“标量 PI 不携带方向信息”这一事实；
2. 它区分了输出偏差结构、Jacobian 拉回结构、参数代价结构，这在长期上是对的；
3. 它为未来“结构化蒸馏 + 二阶几何”提供了矩阵对象语言。

### 6.2 当前应冻结的部分

1. 不应再把 `PIM` 当成解释现有失败的默认主因；
2. 不应在尚未修复 one-hot 目标几何前就进入复杂矩阵原型；
3. 不应把“需要方向信息”误读为“当前必须先做矩阵对象”。

### 6.3 正式路线调整

当前应将：

- [`PIM`](docs/spikes/theory-predictive-integrity-matrix-spike.md) **降级为未来工作**；
- “连续化 ground truth + NGD/Fisher-aware 几何蒸馏”**升级为主线**。

理由很直接：若目标本身仍被 one-hot 顶点控制，则即便引入矩阵结构，蒸馏仍会在错误监督几何上工作。

## 七、替代理论：连续化目标 + NGD 几何蒸馏

### 7.1 新的目标构造

定义连续化真值：

`q_gt^cont(·|x) = (1 - ρ(x))δ_c + ρ(x) r_c`

其中：

- `ρ(x) ∈ [0, 1]` 为扩散强度；
- `r_c` 为非退化参考分布，可取均匀基线、teacher-truncated 分布或 logit-Gaussian induced 分布。

再定义最终目标：

`q_target(·|x) = (1 - λ_PI(x)) q_gt^cont(·|x) + λ_PI(x) q_T(·|x)`

训练损失：

`ℒ(θ) = E_x [ KL(q_target(·|x) || p_θ(·|x)) ]`

更新规则：

`Δθ = -η Ĥ_F(θ)^(-1) ∇_θ ℒ`

其中 `Ĥ_F` 是 Fisher 近似，可由 Adam 二阶矩或 [`ARS2-Neo`](.roo/rules/routing-theory.md:317) 风格代理提供。

### 7.2 关键性质

#### 性质 1：teacher 不再直接面对 one-hot 顶点

teacher 对抗的不是 `δ_c`，而是 `q_gt^cont`。这把“硬冲突”改写为“两个内部点之间的对齐”。

#### 性质 2：结构质量不再只有 `λ` 一个旋钮

原框架里，结构保留强度几乎全部压在 `λ` 上；新框架中分成：

- `ρ` 负责标签连续化；
- `λ_PI` 负责 teacher 资格；
- `NGD` 负责更新几何。

#### 性质 3：Fisher 几何减弱坐标系依赖

欧式梯度对 logits/参数重参数化敏感，而 NGD 近似在分布层面对 KL 最陡下降方向做校正，更有利于保留软目标中的小概率结构。

## 八、最小可验证假设

### H1：one-hot 锚点坍缩假设

在固定 teacher 与 student 条件下，若比较：

- `q_target = (1 - λ)δ_c + λq_T`
- `q_target = (1 - λ)q_gt^cont + λq_T`

则前者会在训练过程中更快损失 teacher 的非标签结构质量。

可观测量：

- student 与 teacher 的非标签 KL
- student 输出熵
- 非标签总质量保持率

### H2：连续化真值保结构假设

只要 `q_gt^cont` 不退化为 one-hot，且 `ρ` 进入非零区间，则 student 对 teacher 关系结构的保留优于 one-hot 混合基线。

### H3：NGD 优于 AdamW 的结构保持假设

在相同 `q_target` 下，Fisher-aware 更新比 AdamW 更能维持：

- teacher-student 分布关系；
- 非标签质量；
- 小概率结构的局部曲率一致性。

## 九、最小实验设计

### 9.1 实验目标

优先验证三件事：

1. one-hot CE + soft teacher 混合是否导致结构坍缩；
2. continuous target 是否能显著减缓这种坍缩；
3. Fisher-aware update 是否进一步改善结构保持。

### 9.2 最小设置

沿用 [`exp/pi_weighted_self_distillation_spike.py`](exp/pi_weighted_self_distillation_spike.py:1) 的玩具框架，但重写 Task B 训练目标。

建议四组主对照：

1. **HardMix + AdamW**  
   `q_target = (1 - λ)δ_c + λq_T`

2. **ContMix + AdamW**  
   `q_target = (1 - λ)q_gt^cont + λq_T`

3. **HardMix + FisherAware**  
   与 1 相同目标，但使用 NGD 近似更新

4. **ContMix + FisherAware**  
   与 2 相同目标，使用 NGD 近似更新

其中 `q_gt^cont` 的最小实现可取：

`q_gt^cont = (1 - ρ)δ_c + ρu`

这虽然只是 label smoothing 基线，但足以先验证“连续化是否必要”。若成立，再升级到 teacher-anchored 或 logit-Gaussian 版本。

### 9.3 关键指标

除旧有精度指标外，必须新增结构指标：

1. **Teacher-Structure Retention**  
   `R_nonlabel = Σ_{k ≠ c} min(p_k, q_T,k)`

2. **Non-Label Mass Ratio**  
   `M_nonlabel = Σ_{k ≠ c} p_k`

3. **Target-to-Student KL**  
   `KL(q_target || p_θ)`

4. **Teacher-to-Student KL on non-label support**

5. **Logit Geometry Drift**  
   比较 AdamW 与 Fisher-aware 更新下的 logits 漂移范数

### 9.4 成功判据

若满足以下任一条件，则新主线得到支持：

1. `ContMix + AdamW` 比 `HardMix + AdamW` 有更高的结构保持指标；
2. `ContMix + FisherAware` 比 `ContMix + AdamW` 进一步提高结构保持；
3. 新任务精度不显著下降的同时，teacher 结构保持显著改善。

## 十、证明草图

### 命题 A

若目标分布为 `q_mix = (1 - λ)δ_c + λq_T`，则当 `λ · Σ_{k ≠ c}q_T(k) << 1` 时，优化梯度主导项必然由标签坐标控制。

**证明思路**:

- 标签坐标的目标质量为 `1 - λ + λq_T(c)`，接近 1；
- 非标签总质量为 `λΣ_{k ≠ c} q_T(k)`；
- softmax 交叉熵梯度为 `p - q_mix`，故标签方向的更新幅度与稳定性占优；
- 于是 teacher 结构只能作为标签梯度上的微扰存在。

### 命题 B

若以 `q_gt^cont` 替代 `δ_c`，则 target 位于单纯形内部，teacher 与标签锚点的几何冲突被削弱。

**证明思路**:

- `q_gt^cont` 与 `q_T` 都位于单纯形内部；
- 二者凸组合不再强制贴近顶点；
- 非标签坐标获得 `O(ρ)` 的基础质量，不再完全依赖 `λq_T(k)`；
- 因而 teacher 结构不再被双重缩放。

### 命题 C

在相同 loss 下，NGD 比欧式梯度更接近分布流形上的 KL 最速下降方向，因此更适合保留软目标结构。

**证明思路**:

- 欧式梯度依赖参数化；
- NGD 用 Fisher 度量修正参数方向，使小概率但高曲率方向不被系统性低估；
- 故当软目标结构主要存在于非主导概率维度时，NGD 更有可能维持其局部几何。

## 十一、决策结论

1. 当前主失败机理应从“PI 不够聪明”改判为“one-hot ground truth 破坏了软蒸馏的目标几何”。
2. 当前主线应正式切换为：**连续化 ground truth + NGD/Fisher-aware 几何蒸馏**。
3. [`PIM`](docs/spikes/theory-predictive-integrity-matrix-spike.md) 保留理论价值，但应冻结为未来工作。
4. 下一实验若执行，应先验证目标连续化与更新几何，而不是继续默认深挖标量 PI 或直接进入 PIM。

## 状态

| 日期 | 状态 | 备注 |
| --- | --- | --- |
| 2026-03-25 | 🟢 重建完成 | 已将主假设从 PIM 转为 one-hot 锚点坍缩与 NGD+连续化目标 |
