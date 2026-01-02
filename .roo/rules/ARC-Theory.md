# ARC 理论备忘录

`Latest update: 2025-10-09`

## H₀: Emergent Differentiable Program Search

Gradient Descent ≡ Program Search.
模型作为通用计算基质 (Universal Computational Substrate)，通过在连续参数空间中的可微优化，发现解决离散符号任务的复杂算法。最小化预测误差驱动网络拓扑自组织，最终在功能上等价于特定任务的求解程序。

## H₁: Loss Masking (样本对规则推断)

定义: ARC ≡ 规则推断 ≫ 输入输出映射。
策略: 全量预测。模型预测所有输出（训练集 + 测试集），而非仅`Test_Output`。参考 LLM 使用成对聊天记录进行 SFT。
机制: 提供多点监督信号，迫使模型提取样本对间的共享抽象规则，而非记忆孤立映射。

## H₂: Adaptive Differential Loss (自适应差分损失)

目标: 聚焦变换 (Transformation)，抑制复制 (Copying)。
机制: 1. Diff Mask: 辅助交叉熵损失 `diff_loss` 仅在变化 Token (x_ij ≠ y_ij) 上计算。 2. Dynamic λ: 权重 λ ∝ N_identity / N_diff。
效果: 等价于秩-1 注意力机制。系统自动放大稀疏的“变化”信号（核心规则），抑制主导的“不变”背景噪声。
