# 技术规格与架构文档：DFS 解码器 (ARC-Specific)

**文档状态**: 提案
**适用范围**: 仅限 Tiny-ONN-ARC 模型

## 1.0 概述

本文档为在 `Tiny-ONN-ARC` 项目中实现一个基于深度优先搜索（DFS）的解码器提供技术规格。该解码器旨在取代现有的 `argmax` 贪婪搜索，以克服局部最优陷阱，从而更充分地发掘模型解决 ARC 任务的内在推理能力。其核心设计思想借鉴了 `ARChitect` 团队在 Kaggle ARC Prize 2024 竞赛中使用的 `turbo_dfs` 算法。

如提案人所指出的，此解码器是为 ARC 这类具有**唯一、精确解**的**推理任务**而专门设计的。对于通用的、开放式的 `Omni-LLM` 模型，传统的采样方法（如 `top-p`, `top-k`）在理论上更为合适。

## 2.0 核心理论

该解码器将 token 生成过程建模为一个在决策树上的搜索问题。它通过**概率阈值剪枝（Probability Threshold Pruning）**的深度优先搜索，系统性地探索所有“足够可能”的生成路径，以保证在给定的概率预算内找到全局最优或近似最优的完整序列。

### 2.1 理论优势

- **克服局部最优陷阱**: `argmax` 的根本缺陷在于其“鼠目寸光”，在每一步都选择局部概率最高的 token。这可能会导致因一个早期的、看似合理但实际错误的决策，而错失需要“深谋远虑”的全局最优解。
- **保证找到最优解**: DFS 方案通过探索所有累积概率在预设阈值内的路径，能够**保证**找到模型在该概率预算内所能生成的、全局概率最高的序列。这是 `argmax` 和随机采样都无法保证的。
- **为 ARC 任务“量身定做”**: 对于 ARC 这类需要精确、多步逻辑推理的任务，一个正确的解决方案可能包含一些在局部看起来概率不高的步骤。DFS 能够发现这些“非直觉”的路径，从而显著提升求解能力。

## 3.0 架构与实现细节

我们将在 `exp/arc/evaluation.py` 的 `ArcGenerator` 类中实现一个新的 `dfs_search` 方法。

### 3.1 核心函数签名

```python
# 位于 ArcGenerator 类中
def dfs_search(
    self,
    input_ids: torch.Tensor,
    config: GenerationConfig,
    **kwargs
) -> list[tuple[float, torch.Tensor]]:
    # ...
```

- **返回**: 一个列表，其中每个元素是一个元组 `(score, sequence)`，按 `score` 从优到劣排序。

### 3.2 核心递归逻辑 (`_recursive_dfs`)

将实现一个私有的递归辅助函数，其状态传递将遵循 `turbo_dfs` 的核心思想。

```python
def _recursive_dfs(
    self,
    current_tokens: torch.Tensor,
    current_score: float,
    past_key_values: tuple | None,
    # --- config params ---
    min_prob_threshold: float,
    max_new_tokens: int,
    eos_token_id: int,
) -> list[tuple[float, list[int]]]:
    # ...
```

### 3.3 关键实现机制

1. **分数计算**:
   - 分数将使用**负对数概率（Negative Log-Likelihood, NLL）**进行累加。这在数学上等价于概率的乘积，但在数值上更稳定。
   - `current_score = prev_score + nll(next_token)`
2. **概率阈值剪枝**:
   - 将 `config.min_prob` 转换为最大允许的累积 NLL 分数 `max_score = -log(min_prob)`。
   - 在递归的每一步，如果 `current_score > max_score`，则该路径被剪枝，立即返回。
3. **KV 缓存管理**:
   - 这是性能的关键。在每次递归调用模型生成下一个 token 的 `logits` 时，必须传入当前的 `past_key_values`。
   - `unsloth` 库可能需要手动进行缓存修剪。在回溯（backtracking）时，必须将 KV 缓存的状态恢复到进入当前递归层之前的状态。`ARChitect` 团队的实现通过在递归前记录缓存长度 `pos`，并在返回后将其截断 `cache[:, :, :pos]` 来实现这一点。我们将采用相同的机制。
4. **回溯与序列聚合**:
   - 当一个序列遇到 `eos_token_id` 或达到 `max_new_tokens` 时，递归终止，返回 `(final_score, completed_sequence)`。
   - 上层调用将收集所有子路径返回的完整序列，并将其进一步向上传递，最终在顶层调用中聚合成一个完整的候选列表。

## 4.0 配置与集成

1. **`GenerationConfig`**: 在 [`exp/arc/config.py`](exp/arc/config.py) 的 `GenerationConfig` 类中，我们将添加新的参数，如 `use_dfs: bool = False` 和 `min_prob: float = 0.01`。
2. **`ArcGenerator.generate`**: `generate` 方法将被修改，通过检查 `config.use_dfs` 来决定是调用 `dfs_search` 还是现有的 `greedy_search`。

此文档为后续的开发工作提供了清晰、可执行的蓝图。
