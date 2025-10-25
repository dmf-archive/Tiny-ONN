# DynTRM 实施蓝图 (v1.0)

`日期`: 2025-10-25  
`作者`: Roo (Architect AI)  
`状态`: Draft  
`基于`: ADR-0005, DFC-Theory, SARS 原理与最新审计结论

## 1. 目标与范围

本文档旨在为 `DynTRM` (Dynamic Tiny Recursive Model) 提供一个**唯一且权威**的工程实施蓝图。它取代了 `plan-DynTRM` 目录下所有之前的设计文档。

**核心目标**:

1. 将模型从固定的、堆叠式的 Transformer，演进为一个**动态的、递归的通用计算引擎**。
2. 通过**全局动态专家库**和**自适应计算时间 (ACT)**，实现极致的参数效率和计算深度自适应。
3. 集成并正确实现 `SARS` (Surprise-Aware Routing Shaping) 元学习机制，以驱动专家的自组织。

## 2. 关键设计修正 (基于最新审计)

### 2.1 位置编码：回归一维 RoPE

- **决策**: 移除 [`exp/arc_dyntrm/circle_rope_arc.py`](exp/arc_dyntrm/circle_rope_arc.py) 中所有 `CircleRoPE` 相关代码。
- **原因**: `CircleRoPE` 强制注入绝对坐标，阻碍了模型对位移不变性的学习。空间关系应作为可学习的“内容”，而非不可变的“偏置”。
- **实现**: 采用标准的、一维的 RoPE。空间线索将通过 `<row_sep>` (换行符) token 隐式地提供给模型，使其能从序列模式中推断几何结构。

### 2.2 SARS 核心机制：手工链式法则重计算

- **决策**: 重构 [`exp/arc_dyntrm/train.py`](exp/arc_dyntrm/train.py) 中的 [`_calculate_goodness`](exp/arc_dyntrm/train.py:31) 函数。
- **原因**: 当前实现仅是简单的后处理，未能为路由参数提供学习信号。
- **实现**: 必须移植 `exp/arc/train.py` 中的 [`_calculate_goodness_jit`](exp/arc/train.py:75) 函数。该函数的核心是：
  1. 利用捕获的 `captured_masked_grad_outputs` (输出梯度) 和 `captured_spl_inputs` (输入)。
  2. **手工执行链式法则**，重新计算 `mu_grad_norm` (学习成本)。
  3. 基于理论公式 `goodness = norm(routing_logits) - norm(mu_grad_norm)` 计算 `goodness_logits`。
  4. 使用 `goodness_logits` 构建 `meta_loss`，并通过独立的 `backward()` 调用为路由参数生成梯度。

## 3. 架构与工程实现

### 3.1 顶层架构 (`ArcTransformer`)

与 `ADR-0005` 设计保持一致：

```python
class ArcTransformer(nn.Module):
    # ... (其他组件)
    self.attn_expert_library: nn.ModuleList  # 全局共享的注意力专家库
    self.ffn_expert_library: nn.ModuleList   # 全局共享的FFN专家库
    self.block: DynTRMBlock                  # 单一的、可重入的计算块
    # ... (其他组件)

    def forward(...):
        # ... (前处理)
        while current_step < self.config.num_layers: # 递归循环
            x, ... = self.block(x, self.attn_expert_library, self.ffn_expert_library, ...)
            # ... (ACT 停止判断)
        # ... (后处理)
```

### 3.2 核心计算块 (`DynTRMBlock`)

功能上等同于一个标准的 Transformer Block，但其内部组件是动态构建的。

#### 3.2.1 DynSIHA+ (动态稀疏无限头注意力)

- **职责**: 为每个物理头，从全局专家库中**动态合成** Q, K, V 向量。
- **路由**: 采用**稠密合成 (Dense Composition)** 范式。计算所有专家的输出，再通过 `einsum` 进行加权求和，这有利于 GPU 并行化。随着硬件支持的改进，可以考虑替换到更典型的 MoE 式路由后合成。
- **专家库**: 64 个 `ExpertMLP`，每个处理 `d_head` 维的输入输出。

#### 3.2.2 DynMoE (动态专家混合 FFN)

- **职责**: 为每个 token，从全局专家库中**动态选择**一个或少数几个 FFN 专家。
- **路由**: 采用**稀疏选择 (Sparse Selection)** 范式。使用 `ReLU` 进行自然稀疏门控，仅对被激活的专家进行计算，以促进专家功能分化。
- **专家库**: 16 个 `ExpertMLP`，每个是标准的 `d_model -> 4*d_model -> d_model` 结构。

#### 3.2.3 ExpertMLP (专家 MLP)

提供具体的非线性变换能力，并同时用于`Latent Head`和`FFN Expert`：

```python
class ExpertMLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x))) # 标准 SwiGLU 变体
```

### 3.3 自适应计算时间 (ACT)

在 `DynTRMBlock` 末尾集成一个轻量级的 `halt_predictor`。

- **训练**: 采用**最大步数上限展开**。计算图最高可展开为 `N` 步，梯度通过所有步骤反向传播。
- **推理**: 使用 `halt_predictor` 输出的 `q_halt` 和 `q_continue` logits，当 `sigmoid(q_halt - q_continue)` 超过阈值时，提前终止递归循环。
