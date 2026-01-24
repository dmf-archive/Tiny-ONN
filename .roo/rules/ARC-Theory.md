# ARC 任务备忘录

`Latest update: 2026-01-25`

## H₀: Emergent Differentiable Program Search

Gradient Descent ≡ Program Search.
模型作为通用计算基质 (Universal Computational Substrate)，通过在连续参数空间中的可微优化，发现解决离散符号任务的复杂算法。最小化预测误差驱动网络拓扑自组织，最终在功能上等价于特定任务的求解程序。

## H₁: Loss Masking

定义: ARC ≡ 规则推断 ≫ 输入输出映射。
策略: 全量预测。模型预测所有输出（训练集 + 测试集），而非仅`Test_Output`。参考 LLM 使用成对聊天记录进行 SFT。
机制: 提供多点监督信号，迫使模型提取样本对间的共享抽象规则，而非记忆孤立映射。

## H₂: Adaptive Differential Loss

目标: 聚焦变换 (Transformation)，抑制复制 (Copying)。
机制: 1. Diff Mask: 辅助交叉熵损失 `diff_loss` 仅在变化 Token (x_ij ≠ y_ij) 上计算。 2. Dynamic λ: 权重 λ ∝ N_identity / N_diff。
效果: 等价于秩-1 注意力机制。系统自动放大稀疏的“变化”信号（核心规则），抑制主导的“不变”背景噪声。

## DatasetAnalysis

Source:`data/ARC-AGI-2/data/training`,`data/ARC-AGI-2/data/evaluation`;Script:`exp/analyze_arc_data.py`

## TokenLength(L)Distribution

|Set|N|Min|Max|Mean|P50|P80|P90|P95|P98|P99|
|:----|:--|:--|:---|:---|:---|:---|:---|:---|:---|:---|
|Train|1k|68|9.3k|1.5k|1.1k|2.2k|3.4k|4.3k|5.6k|7.0k|
|Eval|120|691|9.3k|3.0k|2.8k|4.1k|4.7k|5.6k|7.3k|8.1k|

## GridSize(HxW)Topology(Top15)

|Metric|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|
|:-------------|:----|:----|:----|:----|:----|:----|:----|:----|:----|:-----|:-----|:-----|:-----|:-----|:-----|
|TrainSize|10x10|3x3|9x9|16x16|15x15|12x12|5x5|6x6|7x7|11x11|8x8|4x4|20x20|13x13|30x30|
|Train%|11.6%|7.6%|4.1%|3.6%|3.3%|2.8%|2.7%|2.5%|2.3%|2.3%|2.3%|2.2%|2.2%|2.1%|1.9%|
|EvalSize|30x30|20x20|16x16|10x10|12x12|22x22|18x18|29x29|26x26|8x8|15x15|25x25|30x22|11x11|27x27|
|Eval%|8.7%|6.9%|5.8%|5.3%|4.8%|4.3%|2.5%|1.7%|1.6%|1.5%|1.4%|1.4%|1.4%|1.3%|1.3%|
