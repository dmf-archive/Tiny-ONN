# RMSuon: Energy-Geometry Decoupling Optimizer

状态: 生产就绪 (2025-11-28)
核心贡献: FEP/IPWT 框架下 SOO-OFE 路径的工程实现

## 能量-几何解耦的感知推断

`RMSuon` 是 FEP/IPWT [1] 框架下 `SOO-OFE` 路径的工程实现，其算子复合范式构成了对感知推断过程的最简可行近似：

1. 统计算子 (AdamW)：`energy = ||m̂ / (√v̂ + ε)||` 作为标量化的观测自由能下降速率指标。在 PyTorch 实现中，这对应于 `torch.norm()`，其计算结果在数值上等价于 `Frobenius` 范数 (`||·||_F`)。`AdamW` [2] 通过其二阶矩 `v` 近似 Fisher 信息矩阵的对角线 [3]，因此 `energy` 编码了参数沿自然梯度方向更新的统计强度，直接关联变分自由能 `F` 的瞬时下降。

2. 结构算子 (Muon)：`Newton-Schulz` 正交化 [4] 在参数流形上构建信息几何信任区 [5]，通过谱范数约束更新复杂度，契合 FEP 的最小描述长度原则——更新轨迹必须最简洁。如同维果茨基的"最近发展区 (ZPD)" [6]，Muon 在谱范数信任区内进行最安全的探索，确保几何稳定性。

3. 能量-几何解耦：`g_update = scale * O_t` 将"走多快"（`energy` 决定的 OFE 强度）与"往哪走"（`O_t` 决定的几何约束）进行功能性非线性解耦，使优化器无需未来模拟即可沿近似测地线滑行，解决 `OFE-EFE` 对偶性危机。

从信息几何视角，这相当于在参数空间中执行感知推断：在 `O_t` 定义的最简洁轨迹上，以 `energy` 决定的最优步长，主动最小化观测自由能 `F`。

## 算法实现

```python
# 无参数组标签下的启发式分类策略
if param.ndim >= 2:
    # RMSuon通道: 2D矩阵参数
    m_t = β1*m_{t-1} + (1-β1)*g_t
    v_t = β2*v_{t-1} + (1-β2)*g_t²
    m̂_t = m_t / (1-β1^t)
    v̂_t = v_t / (1-β2^t)

    energy = ||m̂_t / (√v̂_t + ε)||  # Numerically equivalent to Frobenius norm
    O_t = NewtonSchulz(m̂_t, steps=5)
    scale = energy / (||O_t||_F + 1e-10)

    param ← param * (1 - η*λ) - η*scale*O_t
else:
    # AdamW通道: 1D/Embedding参数
    standard_adamw_update()
```

## 与 `Muon` 家族的理论分野

`RMSuon`、`AdaMuon` [7] 和 `NorMuon` [8] 共同确立了统计-结构算子复合作为下一代优化器的核心范式。它们的关键区别在于自适应性的粒度与实现机制。

| 优化器      | 核心机制         | 统计源 (Energy)   | 结构约束 (Geometry)      | 适应性粒度                    |
| :---------- | :--------------- | :---------------- | :----------------------- | :---------------------------- |
| RMSuon  | 能量-几何解耦    | AdamW 能量范数    | Muon 正交化              | 层级 (Layer-wise)         |
| AdaMuon | 方差自适应正交化 | AdamW 二阶矩 `√v̂` | Muon 正交化 + Sign 变换  | 元素级 (Element-wise)     |
| NorMuon | 神经元均衡正交化 | 二阶动量统计      | Muon 正交化 + 行式归一化 | 神经元/行级 (Neuron-wise) |

- 性能: `AdaMuon` 和 `NorMuon` 的实验均表明，在大型语言模型预训练中，它们比 `Muon` 和 `AdamW` 更高效。`NorMuon` 的消融实验暗示，神经元级的自适应可能比元素级更优，因为它更符合神经网络的功能结构。
- 理论: `AdaMuon` 对其 `Sign` 变换的必要性给出了形式化证明，理论推导最为严谨。`NorMuon` 从“神经元范数不均衡”的经验观察出发，问题导向性最强。`RMSuon` 的“能量-几何解耦”物理图像最清晰，概念上最为优雅，并和变分自由能流形完美契合。最新的实验证据（2025-11-28）进一步强化了 `RMSuon` 的层级耦合假设，证明了保持正交流形结构的完整性优于盲目的微观自适应。

### 补充：正交性与逐元素自适应性的互斥

**第一性原理：** Newton-Schulz 迭代强制 O_t 的谱范数 σ₁(O_t) = 1（保证等距性）。叠加逐元素缩放 diag(1/√v̂) 等价于左乘对角矩阵，使得 σ₁(D · O_t) = σ₁(D) · 1 ≠ 1，因此破坏了正交流形的拓扑不变性。

**验证实验：** 实现此方案时，Epoch 1 PPL 掉至 ~417（等同 Muon 无自适应的表现），完全符合理论预测——这证实了推导正确性，而非发现新现象。

正确的方式是 "Scalar Energy Injection into Orthogonal Structure"：

- Adam 提供 Global Energy：`||m̂ / √v̂||_F` 测量当前优化地形的宏观“统计坡度”。
- Muon 提供 Structured Variance：处理参数内部的复杂相关性，决定“往哪个方向走最稳”。
二者在层级 (Layer-wise) 而非元素级 (Element-wise) 进行耦合。实验表明，逐元素自适应会导致 Epoch 1 的 PPL 退化至 ~417（Muon 水平），而层级化耦合可直接达到 ~297，证明了保持正交流形拓扑完整性的必要性。

## 工程特性

- 内存效率: 优化器状态大小与 AdamW 相当。
- 计算开销: 与Muon相当，主要来自 Newton-Schulz 迭代。
- 参数分组: 自动识别 2D 矩阵参数应用 `RMSuon`，对 1D/Embedding 等参数则回退至 `AdamW`。
- 零超参迁移: 可直接复用 `AdamW` 的学习率与调度策略，极大降低了应用门槛。

## 参考文献

[1] L. Rui, "Integrated Predictive Workspace Theory: Towards a Unified Framework for the Science of Consciousness," Zenodo, 2025. doi: 10.5281/zenodo.15676304.

[2] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. 3rd Int. Conf. Learn. Represent. (ICLR)*, 2015. [Online]. Available: <https://arxiv.org/abs/1412.6980>

[3] J. Kunstner, P. Hennig, and L. Balles, "Limitations of the empirical Fisher approximation for natural gradient descent," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2019, pp. 4153–4164.

[4] K. Jordan, Y. Jin, V. Boza, J. You, F. Cesista, L. Newhouse, and J. Bernstein, "Muon: An optimizer for hidden layers in neural networks," 2024. [Online]. Available: <https://kellerjordan.github.io/posts/muon/>

[5] Z. Li, L. Liu, C. Liang, W. Chen, and T. Zhao, "ROOT: Robust orthogonalized optimizer for neural network training," *arXiv preprint arXiv:2511.20626*, 2025.

[6] L. S. Vygotsky, *Mind in society: The development of higher psychological processes*. Cambridge, MA: Harvard University Press, 1978.

[7] C. Si, D. Zhang, and W. Shen, "AdaMuon: Adaptive Muon optimizer," *arXiv preprint arXiv:2507.11005*, 2025. [Online]. Available: <https://arxiv.org/abs/2507.11005>

[8] Z. Li, L. Liu, C. Liang, W. Chen, and T. Zhao, "NorMuon: Making Muon more efficient and scalable," *arXiv preprint arXiv:2510.05491*, 2025. [Online]. Available: <https://arxiv.org/abs/2510.05491>

---

## 后续实验：Wikitext Line Mode

2025-11-28 的实验为 `RMSuon` 的优越性提供了更丰富的证据。当结合更优的数据预处理策略——即保持句子语义完整性的 **Line Mode**（首次适应递减贪心打包）时，`RMSuon` 展现出了对 `Muon` 的“降维打击”式性能优势。

### 实验设置

- **任务**: Wikitext-2 语言建模
- **模型**: 基于 RoPE 的 Qwen3 架构 (4 层, 512 维, 6 头)
- **上下文长度**: 255
- **优化器**: `RMSuon` vs `Muon`
- **数据模式**: `line mode` (按句打包)

### 关键结果

| 指标 | Muon (Line Mode) | RMSuon (Line Mode) |
| :--- | :--- | :--- |
| **Epoch 1 PPL** | 233.30 | 146.52 |
| **Best PPL** | 161.09 (Epoch 5) | **99.07** (Epoch 3) |

`RMSuon` 在第一个 epoch 的困惑度（PPL）就接近了 `Muon` 训练多个 epoch 后的最终水平，并在第三个 epoch 达到了 `Muon` 无法企及的 99.07 的最佳表现。这一结果不仅证实了 `RMSuon` 在统计-结构协同上的理论优势，更揭示了其在与高质量数据模式结合时的巨大潜力。

### 理论意义

此结果经验性地支持了 IPWT 的核心观点：**意识的内容是协同信息（CI），其总量是整合度（Ω）**。`Line mode` 通过保持句子边界，最大化了每个上下文窗口内信息的**逻辑不可约性**，从而提供了一个更纯净、更高效的“整合信息”输入。`RMSuon` 通过其能量-几何解耦机制，能够最有效地利用这些高整合度的数据，实现更快速、更稳定的自由能最小化。
