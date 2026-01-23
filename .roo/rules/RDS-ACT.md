# RDS-ACT: Recursive DynSIHA with Adaptive Computation Time

> **状态**: 实验性 (Experimental)
> **核心目标**: 实现全局资源共享的动态递归推理，并通过自监督学习实现精准的提前停止（Early Exit）。

## 1. 核心理念

RDS-ACT (Recursive DynSIHA with ACT) 是 DynSIHA 架构的终极形态。不同于 `FlatDynSIHA` 的层级堆叠，RDS-ACT 在物理上仅由一个共享的递归块（Recursive Block）组成。模型通过在时间维度上的展开，根据任务复杂度动态决定计算步数。

## 2. RDS-ACT 训练范式：Per-Layer Speculative Decode, PLSD

RDS-ACT 采用一种独特的自监督对齐策略。在训练阶段，模型按预设的最大递归深度 $T_{max}$ 展开（类似于 `FlatDynSIHA`， 但所有层共享单一专家组），在每一个递归步（层）的输出端都挂载一个共享解码头（LM Head）进行即时解码。系统计算每一层输出相对于目标的 Loss，形成一个 Loss 序列 $[L_1, L_2, ..., L_{T_{max}}]$。取其中最小的loss作为反向传播路径，并让独立的 ACT 预测头通过学习该序列，实现自监督对齐——即预测在何处达到 Loss 收益递减点或全局最小值，从而在推理时实现精准的提前停止（Early Exit）。

## 3. 与 `ref/trm` (ACT-V1) 的对比分析

| 维度 | ACT-V1 (Q-learning/Binary) | RDS-ACT (Loss-Trend/Speculative) |
| :--- | :--- | :--- |
| **核心逻辑** | 预测当前输出是否“正确” (`seq_is_correct`) | 预测 Loss 的演进趋势与边际效用 |
| **信号质量** | 信号稀疏且剧烈（全对或全错） | 信号连续且平滑（Loss 曲线） |
| **鲁棒性** | 在逻辑密集型任务中易训练不稳定 | 对计算收益有更本质的理解，泛化性强 |
| **推理策略** | 二分类判定是否停止 | 识别 Loss 收益递减点，实现最优 Early Exit |

## 4. 待验证目标

- [ ] 在 `RecursiveDynSIHA` 中实现基于 `halt_logits` 的 ACT 逻辑。
- [ ] 验证递归深度与任务复杂度的相关性。
- [ ] 评估 PLSD 范式在 ARC 任务上的收敛稳定性。
