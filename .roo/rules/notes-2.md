# DynNSA: 动态原生稀疏注意力架构备忘录 (v1)

## 1. 核心思想：结构化稀疏与内容驱动稀疏的融合

本架构旨在将 **Native Sparse Attention (NSA)** 的多策略、结构化稀疏框架，与我们自研的 **DynSMHA** 的非结构化、内容驱动的专家选择机制进行深度融合。其核心思想是，将 `DynSMHA` 作为一个更强大、更灵活的**动态稀疏注意力策略**，去**取代** `NSA` 中原有的、基于静态 top-k 的“精细选择（Fine）”策略。

通过这种方式，我们创建了一个统一的注意力层，它能够同时利用：

1. **结构化先验**: 通过固定的滑动窗口和全局压缩，高效处理局部信息和全局上下文。
2. **内容动态选择**: 通过 `DynSMHA` 机制，根据输入内容的语义，自适应地选择和组合一组专门化的“专家”注意力头，以处理非结构化的、长距离的依赖关系。

## 2. `DynNSA` (v1): 最终架构

```mermaid
graph TD
    subgraph "输入"
        IN[hidden_states (B,T,D)] --> NORM[LayerNorm]
    end

    subgraph "DynNSA Layer"
        subgraph "策略 1: 滑动窗口注意力 (Sliding)"
            NORM --> S_ATTN["LocalAttention"] --> S_OUT[sliding_output]
        end

        subgraph "策略 2: 全局压缩注意力 (Compressed)"
            NORM --> C_ATTN["Compression + Attention"] --> C_OUT[compressed_output]
            C_ATTN -- "importance_scores (q @ ck^T)" --> D_ATTN
        end

        subgraph "策略 3: 动态专家注意力 (DynSMHA)"
            NORM --> D_ATTN["DynSMHALayer"] --> D_OUT[dynsmha_output]
        end

        subgraph "最终组合"
            S_OUT & C_OUT & D_OUT --> STACK[Stack Outputs (3,B,H,T,D_h)]
            NORM --> COMBINE_GATE["to_strategy_combine MLP"] -- weights (B,H,T,3) --> EINSUM
            STACK -- "einsum" --> EINSUM[Weighted Sum] --> MERGE[Merge Heads]
        end
    end

    subgraph "输出"
        MERGE --> FINAL_OUT[final_output]
    end
```

### 3. 实现准则 (SOP)

1. **三策略并行**: 整个 `DynNSALayer` 必须并行计算三个独立的注意力策略分支：
    - **Sliding**: 直接使用 `local-attention` 库实现。
    - **Compressed**: 借鉴 `native-sparse-attention` 的实现，包括 `k,v` 的压缩窗口和压缩 MLP。
    - **DynSMHA**: 使用我们已在 `exp/dyn_smha_poc` 中验证的、基于熵亲和度和动态投影的 `DynSMHALayer` 实现。

2. **信号共享 (核心机制)**:
    - `Compressed` 策略分支计算出的**重要性分数** (`csim`, 即 `q @ ck^T`) **必须**被用作 `DynSMHA` 策略分支中**门控网络的内容信号源**。
    - 这将取代 `DynSMHA` 原本基于 `softmax(QK^T)` 的熵计算，为其提供一个更稳定、更具全局视野的路由信号。

3. **动态策略组合**:
    - 必须复用 `native-sparse-attention` 中的 `to_strategy_combine` 模块（一个作用于输入的 `nn.Linear` + `nn.Sigmoid`）。
    - 该模块的输出将作为权重，通过 `torch.einsum` 对三个策略分支的输出进行动态加权求和。

4. **统一的训练目标**:
    - **主损失 (`main_loss`)**: 照常计算。
    - **辅助损失 (`aux_loss`)**: 至少包含来自 `DynSMHA` 分支的 `SurpriseMin` 门控损失。
    - **可选**: 可以为 `to_strategy_combine` 门控的输出权重引入额外的稀疏性或熵正则化损失，以鼓励策略选择的稀疏性。

5. **模块化实现**:
    - 应创建一个新的 PoC 目录 `exp/dyn_nsa_poc/`。
    - 在 `model.py` 中，创建一个 `DynNSALayer` 模块，其内部封装 `SlidingAttention`, `CompressedAttention`, 和 `DynSMHALayer` 三个子模块。
