# Tiny-ONN 剪枝调试总结

**日期：** 2025-07-11
**当前问题：** 剪枝后的 Qwen3-1.7B 模型推理输出乱码。

---

## 1. 已排除的可能性和尝试的修复

### 1.1 残差连接和 `forward` 返回值问题 (已排除)

- **问题描述：** 最初怀疑是 `PrunedQwen3DecoderLayer.forward` 中残差连接处理不当或返回值结构与原始 `Qwen3DecoderLayer` 不一致导致。
- **尝试修复：**
  - 第一次修复：确保 `position_embeddings` 参数正确传递给 `self_attn`。
  - 第二次修复：重构 `forward` 方法，使其返回值严格遵循原始 `Qwen3DecoderLayer` 的 `(hidden_states, self_attn_weights)` 元组结构，并正确处理 `output_attentions` 和 `use_cache` 标志。
- **结果：** 乱码问题依然存在，形式略有变化。这表明问题不在此。

### 1.2 `nn.Identity()` 物理剪枝与 `transformers` 初始化冲突 (已排除，但揭示了深层问题)

- **问题描述：** 怀疑 `PrunedQwen3DecoderLayer` 在 `__init__` 中将模块替换为 `nn.Identity()` 导致 `transformers.from_pretrained` 在加载权重时，无法正确初始化或加载被替换模块的权重，从而导致模型内部状态不一致。
- **尝试修复：**
  - 将 `PrunedQwen3DecoderLayer` 从 `Qwen3DecoderLayer` 继承改为继承 `nn.Module`，并手动复刻 `Qwen3DecoderLayer` 的 `__init__` 和 `forward` 逻辑，在 `__init__` 中根据剪枝阈值物理地创建 `Qwen3Attention`/`Qwen3MLP` 或 `nn.Identity()`。
  - 修复了由此引入的相对导入问题 (`ImportError`)。
  - 修复了 `PrunedQwen3DecoderLayer` 未定义 `attention_type` 属性的问题。
- **结果：** 乱码问题依然存在，形式再次变化。虽然这个方向的修复没有直接解决乱码，但它揭示了 `transformers.from_pretrained` 在加载模型时，如果模型骨架与 `state_dict` 不完全匹配，会静默忽略缺失的权重。这导致了模型内部的数值不稳定。

---

## 2. 已检查的参考资料/代码

- **`common/models/pruned_layers.py`**：我们自定义的剪枝层定义文件。
- **`.venv/Lib/site-packages/transformers/models/qwen3/modeling_qwen3.py`**：Qwen3 模型的原始实现，特别是 `Qwen3DecoderLayer` 和 `Qwen3Attention` 的 `__init__` 和 `forward` 方法。
- **`.venv/Lib/site-packages/transformers/modeling_utils.py`**：`PreTrainedModel.from_pretrained` 的核心实现，包括其如何处理 `config.json`、`auto_map`、`_init_weights`、`post_init` 以及权重加载 (`load_state_dict`) 的逻辑。
- **`weights/Qwen--Qwen3-1.7B-pruned/config.json`**：确认 `architectures` 和 `auto_map` 配置正确。
- **`pruner/prune.py`**：剪枝脚本，负责生成剪枝后的模型权重和配置。
- **`scanner/app.py`**：加载剪枝模型并进行推理的 Gradio 应用。
- **DeepWiki 查询结果**：关于 `Qwen3DecoderLayer.forward` 签名和 `transformers` 初始化机制的权威信息。
- **Git commit `8309a66` 和 `9d13894`**：旧的、能正常工作的“计算旁路”剪枝实现，其关键在于模型结构未被物理修改，仅在 `forward` 中逻辑跳过计算。

---

## 3. 待进一步调查的问题和方向

- **`transformers.from_pretrained` 的深层行为**：尽管 `config.json` 和 `auto_map` 看起来正确，但 `from_pretrained` 在加载我们物理剪枝后的模型时，仍然导致乱码。这可能意味着：
  - `load_state_dict(..., strict=False)` 在处理缺失权重时的默认行为（静默忽略）可能导致未被正确初始化的参数。
  - `Qwen3` 模型可能存在某些隐式依赖或初始化顺序，是我们通过简单复制和修改无法完全模拟的。
  - `nn.Identity()` 在某些情况下可能不是一个完全透明的替代品，尤其是在涉及 `torch.compile` 或其他优化时。
- **“计算旁路”的真正优势**：旧版本之所以能工作，是因为它加载了**完整的**模型权重，然后才在运行时选择性地跳过计算。这避免了 `from_pretrained` 在加载不完整 `state_dict` 时的复杂性。
- **下一步计划（初步设想）**：
  - **回归“计算旁路”**：将 `pruned_layers.py` 恢复到继承 `Qwen3DecoderLayer` 的版本，并在 `forward` 中使用布尔标志跳过计算。
  - **修改 `prune.py` 以物理剪枝 `state_dict`**：让 `prune.py` 加载一个完整的模型，然后手动从其 `state_dict` 中删除被剪枝模块的键，再保存这个物理剪枝后的 `state_dict`。
  - **修改 `scanner/app.py` 加载逻辑**：让 `scanner/app.py` 加载一个**完整的 Qwen3 模型骨架**，然后使用 `model.load_state_dict(pruned_state_dict, strict=False)` 来加载我们手动剪枝过的 `state_dict`。这将确保模型结构和加载的权重文件是匹配的。
