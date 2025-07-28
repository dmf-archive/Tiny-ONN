# 显存占用分析：`retain_graph=True` vs. `aux_outputs` 方案

本分析旨在定量计算两种不同训练循环方案在处理MoE层时的显存占用差异。

## 1. 核心假设与参数

根据项目文件 `configs/E2E_test_train.yaml` 和 `tiny_onn/` 目录下的模型结构，我们确定以下计算参数：

- **Batch Size (`B`)**: 1
- **Sequence Length (`S`)**: 512
- **Hidden Size (`H`)**: 1024 (基于 Qwen3-0.5B 结构)
- **MoE Intermediate Size (`I`)**: 2816 (通常为 H * 2.75)
- **Number of Experts per Layer (`E`)**: 8
- **Number of MoE Layers (`L`)**: 24
- **Data Type**: `bfloat16` (每元素 2 字节)
- **Average Activated Experts (`k`)**: 2 (用于估算稀疏激活)

## 2. 显存占用计算

### A. `retain_graph=True` 方案 (旧方案)

此方案的核心成本在于 `main_loss.backward(retain_graph=True)` 会保留整个计算图，以备 `gating_loss` 的二次反向传播。这导致所有中间激活值（activations）必须驻留显存。

- **单层MoE层保留的激活值大小**:
  - `flat_hidden_states`: `B*S*H` = 1 * 512 * 1024 = 524,288
  - `gate` 输出 (`raw_logits`, `activated_scores`, `routing_weights`): `3 * B*S*E` = 3 * 512 * 8 = 12,288
  - `expert` 内部激活 (估算): `B*S*k*I` = 1 * 512 * 2 * 2816 = 2,883,584
  - **单层总计**: 524,288 + 12,288 + 2,883,584 = 3,420,160 个元素

- **所有MoE层总激活值**:
  - `L * (单层总计) * 2 bytes`
  - `24 * 3,420,160 * 2` = **164,167,680 字节 ≈ 156.6 MB**

**注意**: 这仅仅是MoE层激活值的最小估算。实际开销远大于此，因为它还包括所有Attention、LayerNorm等层的激活，以及PyTorch autograd引擎自身的巨大开销。一个经验法则是，保留的完整计算图大小约等于模型参数大小的2到3倍。

### B. `aux_outputs` + `.detach()` 方案 (新方案)

此方案在 `main_loss.backward()` 后立即释放计算图，仅为门控损失的计算保留几个被 `.detach()` 的小张量。

- **保留的张量总大小**:
  - `router_logits`: `L*B*S*E` = 24 * 512 * 8 = 98,304
  - `routing_weights`: `L*B*S*E` = 24 * 512 * 8 = 98,304
  - `activated_scores`: `L*B*S*E` = 24 * 512 * 8 = 98,304
  - `surprise_matrix` (从cache构建): `L*B*S*E` = 24 * 512 * 8 = 98,304
  - **总计**: 4 * 98,304 = 393,216 个元素

- **总保留张量显存**:
  - `393,216 * 2 bytes` = **786,432 字节 ≈ 0.75 MB**

## 3. 结论

| 方案 | 主要显存开销 (理论计算) | 核心原理 |
| :--- | :--- | :--- |
| **`retain_graph=True`** | **> 156 MB** (仅MoE激活，实际可达GB级) | 保留整个模型的计算图和所有中间激活值。 |
| **`aux_outputs`** | **< 1 MB** | 立即释放计算图，仅保留计算所需的小尺寸、detached的张量。 |

**量化对比**: 从理论上计算，新方案相比旧方案，在处理MoE相关的梯度计算时，**至少能节约超过 99.5% 的激活值显存**。这从根本上解决了因 `retain_graph=True` 导致的显存爆炸问题，是实现高性能训练的核心。