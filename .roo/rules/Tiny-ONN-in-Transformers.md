# 技术规范与实施蓝图：构建独立的 `TinyOnnForCausalLM` 模型

**版本**: 1.0
**日期**: 2025-09-11
**作者**: Ω Researcher

## 摘要

本报告旨在为构建一个全新的、独立的 `TinyOnnForCausalLM` 模型架构提供一份详尽、可执行的技术规范与分步实施蓝图。该架构将原生于 Hugging Face Transformers 生态系统，旨在彻底解决“模型手术”方案带来的集成复杂性与维护难题。

设计将以 `Qwen3` 架构为高级参考，借鉴其成熟的 `RotaryEmbedding` 和 `RMSNorm` 实现，但模型的核心计算单元将完全被我们自有的 `DynamicInfiniteHeadAttention` (DynSIHA) 和 `DynamicInfiniteExpert` (MoIE) 模块取代。训练范式将通过一个自定义的 `TinyOnnTrainer` 实现，以完全兼容我们独特的、基于 `Surprise Minimization Loss (SML)` 和 `KL Divergence Loss` 的双重辅助损失函数。

本蓝图综合了对项目现有代码库的形式化分析，以及与 `DeepWiki` 的多次迭代技术询证，旨在成为后续工程实施阶段的唯一、权威的指导性文件。

---

## 第一部分：架构注册与文件结构

为确保模型能被 `AutoModelForCausalLM` 等工厂类无缝识别，必须遵循 Transformers 的标准模块化规范。

### 1.1. 文件结构

在 Transformers 的安装目录（或本地开发副本）的 `src/transformers/models/` 路径下，创建新的模型目录 `tiny_onn`：

```
src/transformers/models/
└── tiny_onn/
    ├── __init__.py               # 注册与模块暴露
    ├── configuration_tiny_onn.py # 定义 TinyOnnConfig
    └── modeling_tiny_onn.py      # 定义模型核心架构
```

### 1.2. 配置文件 (`configuration_tiny_onn.py`)

此文件定义模型的超参数。

- **`TinyOnnConfig` 类**: 必须继承自 `transformers.PretrainedConfig`。
- **`model_type` 属性 (关键)**: 必须定义一个类属性 `model_type = "tiny_onn"`。这个唯一的字符串是连接配置、模型和自动映射的核心标识符。
- **`__init__` 方法**: 包含所有模型相关的超参数，如 `vocab_size`, `hidden_size`, `num_hidden_layers`, `d_ffn_factor` 等，并必须接受 `**kwargs` 传递给父类构造函数。

```python
# configuration_tiny_onn.py
from transformers import PretrainedConfig

class TinyOnnConfig(PretrainedConfig):
    model_type = "tiny_onn"
    attribute_map = {"num_attention_heads": "num_heads"} # 兼容性映射

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=2048,
        d_ffn_factor=4,
        num_hidden_layers=24,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        # ... 其他参数
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.d_ffn_factor = d_ffn_factor
        self.num_hidden_layers = num_hidden_layers
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        super().__init__(**kwargs)
```

### 1.3. 注册文件 (`__init__.py`)

此文件负责将新模型组件暴露给 Transformers 的命名空间，并执行自动注册。

```python
# __init__.py
from typing import TYPE_CHECKING

from transformers.utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

_import_structure = {
    "configuration_tiny_onn": ["TinyOnnConfig"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tiny_onn"] = [
        "TinyOnnModel",
        "TinyOnnForCausalLM",
        "TinyOnnPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_tiny_onn import TinyOnnConfig
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tiny_onn import (
            TinyOnnForCausalLM,
            TinyOnnModel,
            TinyOnnPreTrainedModel,
        )
else:
    import sys
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)

# 在 __init__.py 的顶层或一个专门的注册函数中
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from .configuration_tiny_onn import TinyOnnConfig
from .modeling_tiny_onn import TinyOnnModel, TinyOnnForCausalLM

AutoConfig.register("tiny_onn", TinyOnnConfig)
AutoModel.register(TinyOnnConfig, TinyOnnModel)
AutoModelForCausalLM.register(TinyOnnConfig, TinyOnnForCausalLM)
```

---

## 第二部分：模型实现 (`modeling_tiny_onn.py`)

这是架构的核心，所有计算逻辑在此定义。

### 2.1. 核心组件

将 `exp/RoPE_SBL/model.py` 中经过验证的 `SparseBayesianLinear` 模块代码迁移至此文件，作为所有动态层的基础。

### 2.2. 借鉴与集成

遵循最佳实践，通过**继承**而非复制代码的方式集成 `Qwen3` 的组件。

```python
# modeling_tiny_onn.py
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding

# 通过直接继承获得 Qwen3 的优化实现
class TinyOnnRMSNorm(Qwen3RMSNorm):
    pass

class TinyOnnRotaryEmbedding(Qwen3RotaryEmbedding):
    pass
```

### 2.3. 解码器层 (`TinyOnnDecoderLayer`)

这是 Transformer Block 的实现，组合了注意力、前馈网络和层归一化。

```python
import torch.nn as nn
from transformers.cache_utils import Cache
from typing import Optional

# DynamicInfiniteHeadAttention 和 DynamicInfiniteExpert (基于 SBL) 也在此文件中定义

class TinyOnnDecoderLayer(nn.Module):
    def __init__(self, config: TinyOnnConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = DynamicInfiniteHeadAttention(config=config, layer_idx=layer_idx)
        self.mlp = DynamicInfiniteExpert(config)

        self.input_layernorm = TinyOnnRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = TinyOnnRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Cache] = None,
        # ... 其他 generate() 所需参数
    ) -> tuple:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # 自注意力模块
        attn_output, self_attn_weights, present_key_value, attn_masked, attn_kl = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            # ...
        )
        hidden_states = residual + attn_output

        # 前馈网络模块
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        ffn_output, ffn_masked, ffn_kl = self.mlp(hidden_states)
        hidden_states = residual + ffn_output
        
        masked_outputs = attn_masked + ffn_masked
        total_kl = attn_kl + ffn_kl

        return (hidden_states, present_key_value, masked_outputs, total_kl)
```

### 2.4. 主模型 (`TinyOnnModel` 与 `TinyOnnForCausalLM`)

- **基类 `TinyOnnPreTrainedModel`**: 继承自 `transformers.PreTrainedModel`，并设置 `config_class = TinyOnnConfig`。
- **`TinyOnnModel`**: 包含词嵌入层、`TinyOnnRotaryEmbedding` 实例、`nn.ModuleList` 包装的 `TinyOnnDecoderLayer` 列表和最终的 `TinyOnnRMSNorm`。其 `forward` 方法负责：
    1. 计算词嵌入。
    2. 调用 `rotary_emb` 生成 `position_embeddings`。
    3. 循环遍历所有 `decoder_layer`，正确传递 `hidden_states`、`position_embeddings` 和 `past_key_value`。
    4. 收集每一层的 `masked_outputs` 和 `kl_loss`。
    5. 返回一个自定义的 `ModelOutput` 对象，包含 `last_hidden_state`, `past_key_values`, `all_masked_outputs`, `total_kl_loss`。
- **`TinyOnnForCausalLM`**: 包装 `TinyOnnModel`，并在其上添加一个 `lm_head` (`nn.Linear`)。其 `forward` 方法调用 `TinyOnnModel`，获取其所有输出，然后计算 `logits`。**最重要的是，它需要将底模型返回的 `all_masked_outputs` 和 `total_kl_loss` 传递出去**，以供自定义 `Trainer` 使用。

---

## 第三部分：训练集成 (`TinyOnnTrainer`)

为了实现我们独特的 `SML+KL` 训练范式，必须创建一个自定义的 `Trainer`。

### 3.1. 创建 `TinyOnnTrainer`

```python
from transformers import Trainer
import torch
import torch.nn.functional as F

class TinyOnnTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 1. 从输入中分离标签
        labels = inputs.pop("labels")

        # 2. 执行前向传播
        #    模型 forward 必须返回包含 logits, masked_outputs, kl_loss 的对象
        outputs = model(**inputs)
        
        logits = outputs.logits
        masked_outputs = outputs.all_masked_outputs
        kl_loss = outputs.total_kl_loss

        # 3. 计算主损失
        # 注意：此处应有对 labels 的 shift 操作
        main_loss = F.cross_entropy(logits.view(-1, self.model.config.vocab_size), labels.view(-1))

        # 4. 计算 Surprise 和 Gate Loss
        gate_loss = torch.tensor(0.0, device=model.device)
        if masked_outputs:
            surprise_grads = torch.autograd.grad(
                main_loss, 
                masked_outputs, 
                retain_graph=True, # 关键！
                allow_unused=True
            )
            # ... 此处为基于 surprise_grads 计算 gate_loss 的完整逻辑 ...
            # (参考 exp/RoPE_SBL/train.py)

        # 5. 计算总损失
        # (权重 w_gate, w_kl 可根据需要调整或动态计算)
        total_loss = main_loss + CONFIG["W_GATE"] * gate_loss + CONFIG["W_KL"] * kl_loss

        return (total_loss, outputs) if return_outputs else total_loss
```

---

## 第四部分：核心功能兼容性与约束

### 4.1. KV 缓存

- **`forward` 签名**: `TinyOnnDecoderLayer` 和 `DynamicInfiniteHeadAttention` 的 `forward` 方法必须接受 `past_key_value: Optional[Cache] = None` 和 `use_cache: bool = False`。
- **状态管理**: 注意力层内部将使用 `past_key_value.update(new_key, new_value, self.layer_idx)` 来更新缓存。
- **返回值**: `forward` 方法必须返回更新后的 `present_key_value`，最终由 `TinyOnnForCausalLM` 包装在 `CausalLMOutputWithPast` 对象中。

### 4.2. `generate()` 方法

只要 `forward` 方法正确实现了 KV 缓存的接口，并返回包含 `logits` 和 `past_key_values` 的 `CausalLMOutputWithPast` 对象，`model.generate()` 即可无缝工作。

### 4.3. 梯度检查点 (关键约束)

- **不兼容性确认**: 基于历史实验 (`exp/RoPE_SBL`) 和 `DeepWiki` 的信息，`torch.utils.checkpoint.checkpoint` 与我们的 SML+KL 训练范式（特别是依赖 `retain_graph=True` 的 SML）存在根本性冲突，会导致 KL 损失的梯度流中断。
- **当前决策**: **在找到可行的替代方案之前，必须禁用梯度检查点功能。** 这意味着在工程实现阶段，需要接受比标准 Transformer 模型更高的激活值内存占用。此约束是本次技术预研的关键发现，必须在资源规划中予以考虑。
