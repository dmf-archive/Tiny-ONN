# 今日会议笔记 26-04-09

## 1. 讨论背景

本次头脑风暴围绕 **PISD (Predictive Integrity Self-Distillation)** 的本质及其在信息几何层面对冲梯度“剪切力”的能力展开。核心目标是探讨 PISD 能否作为一种更轻量、且具备持续学习潜力的算子，在不引入大规模采样（贝叶斯）或高昂物理位移（SAM）的情况下实现泛化提升。

## 2. 核心议题 (Issues/Theses)

### 2.1 PISD 的本质定位

- **状态**：议题中
- **内容**：PISD 首先被视为一种**自蒸馏 (Self-Distillation)** 机制。其核心特征是混合目标 $q_{mix} = (1 - \lambda)y + \lambda p_\theta$。与传统自蒸馏不同，其混合权重 $\lambda$ 受**预测完整性 (PI)** 动态驱动。
- **关联项**：通常与 RLVR (Reinforcement Learning with Verifiable Rewards) 在稳定训练方面存在交叉点（如 SDFT 框架）。

### 2.2 PISD 与 SAM 的空间对偶性假设

- **状态**：议题中
- **对比**：
  - **SAM**：参数空间 $\Theta$ 的扩散算子（物理位移），改变**训练路径**。
  - **PISD**：概率单纯形 $\Delta$ 的扩散算子（目标混合），改变**训练目标**。
- **核心假设**：在信息几何层面，PISD 对目标的扩散（模糊化）可能在效果上等价于对损失景观的平坦化，从而可能取代 SAM 的物理探测步（$k=0$ 实验组）。

### 2.3 贝叶斯方法的“降维映射”

- **状态**：待澄清
- **内容**：探讨 PISD 是否可以被解读为一种“无采样贝叶斯”对齐。
- **参考**：VBLL (Variational Bayesian Last Layers) 和 Martingale Posterior (MGP) 提示了将贝叶斯推断降维至预测规则自洽性的可能性。
- **警告**：需警惕强行建立关联的风险，PISD 的有效性可能源于更基础的正则化动力学。

### 2.4 持续学习中的 p ∩ q 问题

- **状态**：议题中
- **内容**：SGD/GD 面对非平稳数据流时，无法直接优化联合分布。PISD 通过 PI 审计，在“系统级冲突”剧烈时下调外部证据的引力，从而保护已有的协同参数组。

## 3. 实验设计参数 (传感器模式)

目前计划在 Modulo Addition 任务上进行验证：

- **传感器模式**：启用 ARS2-Neo 的 **AGA (Adaptive Geometric Awareness)**，但设置 `alpha=0`。
- **信号源**：利用 `phi_t`（梯度一致性）作为 $C_{geo}$ 的代理。
- **控制项**：排除 ASI（Active Sharpening Inference），维持静态 `rho`。

## 4. 后续待办

- [ ] 形式化描述 PISD 与 SAM 在梯度流层面的数学差异。
- [ ] 决定是否将 VBLL 的确定性不确定性估计引入 PI 审计公式。
- [ ] 启动 `exp/pi_weighted_self_distillation_spike.py` 的重构。

---
*记录人：Tiny-Ouroboros 代理*
*最后更新：2026-04-09*
