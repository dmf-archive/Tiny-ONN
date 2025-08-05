# Native Sparse Attention (NSA) 实现报告

## 1. 核心概念与技术亮点

- **稀疏注意力模式**: NSA 是一种高效的稀疏注意力机制，旨在提高 Transformer 模型的计算效率。它通过三种策略（滑动窗口、压缩注意力、精细选择注意力）组合实现稀疏性。
- **Triton 加速**: 核心实现利用 NVIDIA Triton 库编写 CUDA 内核，以实现高性能的稀疏注意力计算。`triton_native_sparse_attention.py` 包含了 Triton 实现的前向和后向核心函数 `native_sparse_attend`。
- **Flex Attention 支持**: `native_sparse_attention.py` 中提供了与 `Flex Attention` 的兼容性，并通过 `create_sliding_mask` 和 `create_fine_mask` 等函数生成 Flex Attention 所需的块掩码。
- **Auto-regressive Inference**: 支持高效的自回归推理模式，通过缓存 K/V 值来加速生成。

## 2. 架构与关键组件

NSA 的实现主要围绕以下核心组件：

- **`SparseAttention` 类** (`native_sparse_attention.py`):
  - 这是主要的注意力模块，它封装了稀疏模式的逻辑。
  - **`sliding_window`**: 使用 `local-attention` 库实现局部滑动窗口注意力。
  - **压缩策略**: 通过 `k_compress` 和 `v_compress` 对 K/V 进行压缩。
  - **精细选择策略**: 根据压缩注意力计算出的重要性分数 (`importance_scores`) 选择最相关的 K/V 块进行注意力计算。
  - **策略组合**: 通过学习到的权重 (`to_strategy_combine`) 组合滑动窗口、压缩注意力、精细选择三种策略的输出。
  - **关键配置参数**:
    - `sliding_window_size`: 滑动窗口大小。
    - `compress_block_size`: 压缩块大小。
    - `compress_block_sliding_stride`: 压缩块滑动步长。
    - `selection_block_size`: 精细选择块大小。
    - `num_selected_blocks`: 精细选择的块数量。
    - `use_diff_topk`: 是否使用可微分的 TopK。
    - `query_heads_share_selected_kv`: 查询头是否共享选择的 KV 块。
    - `compress_mlp`: 用于 K/V 压缩的 MLP 模块。
- **Triton 内核** (`triton_native_sparse_attention.py`):
  - 提供了 `native_sparse_attn_forward` 和 `native_sparse_attn_backward` 函数，通过 Triton 实现稀疏注意力的前向和后向传播。
  - `NSA` 类（`torch.autograd.Function` 的子类）封装了 Triton 内核的调用。
  - 验证 (`test_triton_nsa.py`) 证明了 Triton 实现与 PyTorch 纯实现之间的数值一致性（输出和梯度），增强了可靠性。
- **压缩网络** (`compress_networks.py`):
  - 提供了 `ConvLinearCompress`、`AttentionPool`、`GroupedMLP`、`SingleProjection` 和 `CompressTransformer` 等多种用于压缩 K/V 的网络结构。`SparseAttention` 类的 `compress_mlp` 参数可以使用这些网络。
- **Transformer 整体结构** (`transformer.py`):
  - `Transformer` 类集成了 `SparseAttention` 或常规 `Attention` (基于 `F.scaled_dot_product_attention`) 作为注意力层。
  - 包含了 token embedding、多层注意力与前馈网络 (`FeedForward`)、RMSNorm 等标准 Transformer 组件。
  - `sample` 方法支持自回归生成。

## 3. 与 Tiny-ONN 的潜在集成点

鉴于 Tiny-ONN 项目下一阶段的目标是“全栈稀疏化”，NSA 提供了一个直接可用的稀疏注意力实现。其基于 Triton 的高性能特性与 Tiny-ONN 的需求高度契合。可以考虑将其集成到 `TinyOnnAttention` 模块中，替换或增强现有的注意力机制，从而将稀疏化扩展到注意力层。

## 4. 依赖项

- `einx`
- `einops`
- `jaxtyping`
- `local-attention`
- `rotary-embedding-torch`
- `torch>=2.5`
