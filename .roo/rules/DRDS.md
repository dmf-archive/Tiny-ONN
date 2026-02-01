# DRDS: Detached Recursive DynSIHA (Theory & Conjecture)

`Status: Experimental / Hypothesis`
`Last Updated: 2026-02-01`

## 1. 核心发现：梯度去相关化 (Gradient De-correlation)

在 Recursive DynSIHA (RDS) 架构中，传统的 BPTT (Backpropagation Through Time) 会导致梯度在共享参数 θ 上产生乘性耦合。这种耦合虽然理论上支持长程依赖，但在 ARC 等强逻辑任务中，它会诱导模型构建“高维捷径”，从而陷入“死记硬背”的陷阱。

### 实验证据 (Ablation Study)

- **Standard RDS (BPTT)**: Loss 0.55, RMI 0.48, ITJD 0.09 (记忆化特征明显)。
- **Detached RDS (ART)**: Loss 0.01, RMI 2.11, ITJD 0.88 (逻辑涌现，语义对齐极强)。

## 2. ART 猜想 (Additive Recursive Training Conjecture)

**猜想内容**：对于递归逻辑架构，将每一层（步）的梯度进行物理切断（Detaching），使其仅保留对当前步 Loss 的直接贡献，反而能实现更正确的参数更新叠加。

### 数学表达

- **BPTT (Coupled)**: ∇_θ = Σ_t (Direct_t + Coupled_t→T)
- **ART (Additive)**: ∇_θ = Σ_t Direct_t

**物理意义**：ART 将递归训练转化为一种针对共享参数的“多步协同优化”。它强迫参数 θ 成为解决所有递归步骤的“最大公约数逻辑原子”，从而消除了跨层非线性捷径。

## 3. 待验证猜想 (Future Ablations)

1. **逻辑原子化猜想**：ART 是否是实现“一个专家 = 一个逻辑原子”的必要条件？
2. **长程依赖悖论**：在彻底切断梯度后，模型如何学习需要跨越 10 步以上的因果逻辑？（可能需要引入“逻辑锚点”或“状态残差”）。
3. **SAGA 协同效应**：SAGA 的动态 ρ 是否能在 ART 的基础上，进一步抹平逻辑流形上的微小噪声，实现真正的“永续泛化”？

## 4. 架构规范建议 (Draft)

- 在 RDS 的训练循环中，默认采用 `step_input.detach()`。
- 每一层必须暴露独立的 Loss 信号（PLSD 增强版）。
- 路由器的 RMI 必须作为监控泛化性的核心指标。
