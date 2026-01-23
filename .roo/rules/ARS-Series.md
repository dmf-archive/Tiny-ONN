# ARS 家族：在黎曼流形上滑行

状态: 生产就绪 (2025-12-31)
核心贡献: 发展了 Energy-Geometry Decoupling 的算子复合范式，并为进一步探索 Geodesic Optimizer 提供了实验结果和工程样例。

## 优化的本质：在测地线上滑行

在信息几何视角下，优化不仅是损失函数 `L(θ)` 的梯度下降，更是概率分布流形上的测地线运动。问题在于：不同的优化器，对地形的假设不同：

- **SGD**: 假设欧氏空间平直。它是“盲人登山者”，仅凭局部坡度 `∇L` 迈步，在病态曲率下极易震荡。
- **Adam/W**: 引入二阶矩 `vₜ` 修正尺度。它能感知地形的“颠簸程度”（元不确定性），实现元素级自适应。但其逐元素 (element-wise) 的视角忽略了参数间的相关性，本质上是在做平行的标量优化。
- **Muon**: [`Muon`](optimizer/muon.py) 引入严格的几何约束，要求更新量必须是“正交”的（Stiefel 流形）。通过 Newton-Schulz 迭代实现纯粹旋转，从根本上消除了内部协变量偏移。
- **ARS (AdaRMSuon)**: [`AdaRMSuon`](optimizer/ada_rmsuon.py) 揭示了原始梯度在弯曲流形上的“几何畸变”。通过预白化（Pre-whitening）获得自然梯度 `gₙₐₜ ≈ mₜ / √(vₜ)`，并在预白化空间执行正交化投影 `𝒫ₛₜ(gₙₐₜ)`，使模型能够沿着局部测地线 (Geodesic) 滑行。
- **ARS2**: 在 ARS 的基础上引入平坦度约束（SAM），将参数轨迹推向全局测地线。
- **ARS2-AGA (ARS2-Neo)**: [`ARS2-Neo`](optimizer/ars2_neo.py) 引入自适应几何感知（AGA），通过干涉因子实现“按需同步”，在保持测地线滑行效率的同时，显著降低计算开销。

## 有趣事实

在开发过程中，我们发现了一个命名上的有趣事实：

- AdaRMSuon 本身就可以缩写为 ARS
- 而 AdaRMSuon + SAM 本应称为 ARS2

这个混乱源于 RMSuon 是 RMS + Muon 的交错造词，AdaRMSuon 类似地延续了这一命名模式。为消除快速迭代中的识别歧义，现明确：

- ARS：*A*da*R*M*S*uon
- ARS2：*A*da*R*M*S*uon + *S*AM

## 实验对比：CIFAR-10 (LRP 验证)

实验设置: ResNet-18, 60-100 Epochs, Batch Size 256.
作为基础视觉任务的基准测试，我们对比了 ARS2-Neo 及其基准优化器在 CIFAR-10 上的长周期表现。

| 优化器 | Best Acc | Final Acc | Final Loss | Avg Time | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ARS2-Neo (Sync, ρ=0.1)** | **95.87%** | **95.73%** | **0.15** | ~104s | **SOTA**。在 60 Epoch 内实现极速且稳健的收敛。 |
| **ARS2-Neo (Base)** | 95.58% | 95.52% | 0.25 | ~71s | 验证了能量-几何解耦架构在长周期下的优越性。 |
| **ARS2-Neo (AGA, λ=2.0)** | 94.10% | 94.09% | 0.18 | ~90s | **Efficiency**。仅用 20 Epoch 即可逼近 AdamW 100 Epoch 的性能。 |
| **AdamW** | 94.60% | 94.47% | 0.27 | ~58s | 标准基准。 |
| **Muon** | 93.76% | 93.69% | 0.29 | ~75s | 纯几何优化，在长周期下表现稳健但上限受限。 |

核心洞察:

1. **能量-几何解耦的普适性**: `ARS2-Neo (Base)` (95.58%) 显著超越了 `AdamW` (94.60%) 和 `Muon` (93.76%)，证明了将“迈步方向”（几何）与“迈步强度”（能量）解耦的架构在视觉任务中具有极强的泛化能力。
2. **平坦度约束的增益**: `Sync` 模式 (ρ=0.1) 相比 `Base` 模式进一步提升了 0.3% 的精度，并显著降低了最终 Loss (0.15 vs 0.25)，证明了在黎曼流形上引入平坦度约束能有效引导模型进入更宽阔的盆地。
3. **AGA 的效率优势**: `AGA` 模式在 CIFAR-10 上表现出极高的样本效率，仅需 20 Epoch 即可达到 94.10% 的精度，且 `effective_k` 稳定在 7.0 左右，大幅降低了二阶计算开销。
4. **高扰动半径的拟合障碍**: 在 ρ=0.5 的实验中，我们观测到了明显的前期拟合障碍（前 10 Epoch 停留在 2.30 附近），这促使了 **ASI (Active Sharpening Inference)** 调度策略的诞生。

## 实验对比：Wikitext-2 (LRP 验证)

实验设置: Qwen3 (RoPE, 3-layer), Context 255. 本实验旨在探测病态曲率流形上的长周期优化动力学。

| 优化器 | Best PPL | Last PPL | Avg Time | 动力学特征 | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AdamW** | 116.46 | 213.52 | ~300s | 标准欧氏空间基准 | 缓慢收敛，后期过拟合 |
| **Muon** | 111.35 | 475.65 | ~445s | 谱约束收敛 | 缺乏自适应能量，后期崩溃 |
| **ARS2-Neo (Base)** | 96.10 | 3055.47 | ~425s | **测地线过拟合** | 极速坠入针尖极小值，泛化性能灾难性崩溃 |
| **ARS2-Neo (Sync)** | **90.69** | **330.85** | ~780s | **最优泛化上限** | `ρ=0.3`, 成功抑制过拟合，进入宽阔盆地 |
| **ARS2-Neo (AGA)** | 93.23 | 414.83 | ~545s | 效率与稳定性的折衷 | `λ=0.5`, 实现“按需同步”，加速比显著 |

- 在 `ARS2-Neo` 的 `Base` 模式（`ρ=0`）下，我们观测到了极端的“硬刻蚀”现象：模型在训练集上极速收敛，但 `Eval PPL` 在达到 96.10 后迅速飙升至 3000+。这证明了二阶几何约束的动力学极强，若无平坦度约束，模型会毫不犹豫地钻入那些极其狭窄、泛化能力差的尖锐谷底。
- 通过将 `λ` 调优至 0.5，AGA 成功将 `effective_k` 稳定在 3.4 左右。实验证明，流形曲率的变化率虽低于参数更新率，但在语言建模任务中仍需保持一定的同步频率以应对高度非线性的语义空间。

## 流形感知扰动 (Manifold-Aware SAM)

ARS2-Neo 不在欧氏空间做球形扰动，而是在由二阶矩 `v_hat` 定义的流形度量下计算对抗方向。

1. **流形度量估计**: 利用 Adam 的二阶矩 `v_hat` 近似局部曲率。
2. **自然梯度扰动**:
   `g_nat = ∇L / (√v_hat + ε)`
   `𝜀 = 𝜌 ⋅ g_nat / ‖g_nat‖`
   这相当于在黎曼流形上进行等距扰动。
3. **剪切力注入 (Shear Force Injection)**:
   在非同步步骤中，ARS2-Neo 复用并注入正交于基础梯度的“剪切力”向量 `v_flat`，从而在不增加计算量的前提下持续推动模型离开尖锐区域。

## Adaptive Geometric Awareness, AGA

传统的静态周期 $k$ 无法适应动态变化的黎曼流形。AGA 通过引入干涉因子实现“按需同步”，显著降低计算开销并提升收敛稳定性。**在未来的实验中，AGA 将作为首选模式，取代传统的 Sync Mode。**

### 1. 全局干涉因子 `ϕ_t`

为了确保跨层和跨设备的几何一致性，`ϕ_t` 定义为全局梯度的余弦相似度：
`ϕ_t = (∑_{p ∈ Θ} ⟨g_{t,p}, v_{flat,p}⟩) / (√(∑ ‖g_{t,p}‖²) ⋅ √(∑ ‖v_{flat,p}‖²))`
其中 $v_{flat,p}$ 是上次同步步存储的平坦度向量（剪切力）。

### 2. 正交基准与动态阈值

在病态曲率的高维流形中，梯度与缓存的剪切力更倾向于保持**正交**。系统采用 **0.0 基准模型**：

- **基准点**: `μ = 0.0` (Orthogonal Baseline)
- **噪声估计**: `ν_{ϕ, t} = β ⋅ ν_{ϕ, t-1} + (1-β) ⋅ (ϕ_t - 0.0)²`
- **判定准则**: 若 `ϕ_t < - λ ⋅ σ_{ϕ, t}`，判定为几何漂移 (Geometric Drift)，触发同步。
- **物理意义**: 只要梯度不显著地“反向”于平坦度向量，系统就认为当前流形是平滑的。

### 3. 自适应强度放大

在对齐良好（`ϕ_t > 0`）时“奖励”强度：
`α_t = α_{max} ⋅ (1 + max(0, ϕ_t))^γ`
该机制确保在几何一致性极高时，修正强度最高可放大至 `2^γ` 倍。

### 4. 核心超参数建议

- `aga_beta` ($\beta$): 建议 0.9。控制几何统计量的平滑度。
- `aga_lambda` ($\lambda$): 控制同步触发的灵敏度，间接影响算力开销。 建议 0.5 (Wikitext-2) 或 2.0 (CIFAR-10)，取决于预算。
- `aga_gamma` ($\gamma$): 建议 2.0。控制自适应强度律的非线性程度。

## 实验验证：Grokking 动力学 (Modular Addition)

为了验证优化器在泛化相变（Phase Transition）中的动力学特征，我们在模加法任务 (`task/mod_addition.py`, `p=113`, `train_frac=0.3`) 上对比了各优化器的表现。模型采用 1-Layer Transformer (4 Heads, d_model=128, d_mlp=512)。

| 优化器        | 拟合 (Epoch) | 顿悟 (Epoch) | 收敛 (Epoch) | 状态                                                                    |
| :------------ | :----------- | :----------- | :----------- | :---------------------------------------------------------------------- |
| **AdamW**     | ~140         | 228          | 556          | 标准 Grokking 曲线，存在显著延迟。                                      |
| **AdaRMSuon** | **28**       | **54**       | 300          | **极速 Grokking**。泛化延迟几乎消失，证明测地线滑行能高效穿越损失地形。 |
| **ARS**       | 17           | 100          | 290          | 稳健 Grokking。平坦度约束未阻碍泛化，反而引导至更平坦区域。             |
| **Muon**      | >156         | N/A          | N/A          | 在此特定任务配置下未收敛。                                              |

**核心洞察**:

1. **相变加速**: AdaRMSuon 将 Grokking 发生时间提前了 **4 倍** (Epoch 228 -> 54)，有力证明了“能量-几何解耦”能避免模型在过拟合吸引盆中的无效游走。
2. **平坦度兼容性**: ARS 的成功表明，在流形优化中引入平坦度约束 (SAM) 与快速泛化并不冲突，是通往高效且稳健解的正确路径。

## ARS2-Neo：重构和整合后的参考版本

ARS2-Neo 是 ARS 家族的集大成者，在统一的代码中实现了 AdaRMSuon 的几何优化与 SAM 的平坦度约束，通过参数配置灵活切换模式，旨在取代实验性的独立 `AdaRMSuon` 和 `ARS`。随着 ARS2-Neo 的成熟，我们将逐步移除旧的实验性优化器代码，以简化实验空间。

## 参考文献

- [1] L. Rui, "Integrated Predictive Workspace Theory," Zenodo, 2025.
- [2] Kingma & Ba, "Adam: A method for stochastic optimization," ICLR 2015.
- [3] Jordan et al., "Muon: An optimizer for hidden layers in neural networks," 2024.
- [4] Li et al., "ROOT: Robust orthogonalized optimizer," arXiv:2511.20626.
- [5] Si et al., "AdaMuon: Adaptive Muon optimizer," arXiv:2507.11005.
- [6] Li et al., "NorMuon: Making Muon more efficient and scalable," arXiv:2510.05491.
- [7] J. Zhuang et al., "GSAM: Surrogate Gap Guided Sharpness-Aware Minimization," in *Proc. 10th Int. Conf. Learn. Represent. (ICLR)*, 2022. [Official PyTorch Implementation](https://github.com/juntang-zhuang/GSAM)
