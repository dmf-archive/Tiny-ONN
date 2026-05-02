# 常用资源路径

为了方便快速查找常用依赖库的文档或咨询 Deepwiki，以下是其对应的 GitHub Repository 地址：

- `pytorch/pytorch`
- `huggingface/transformers`
- `huggingface/accelerate`
- `bitsandbytes-foundation/bitsandbytes`
- `da-fr/arc-prize-2024`
- `michaelhodel/arc-dsl`
- `lose4578/CircleRoPE`
- `SamsungSAILMontreal/TinyRecursiveModels`

---

## 关键源代码文件

以下是重要的参考源代码在工作区内的路径：

- `ARS2-Neo`: `ref/F3EO/optimizer/ars2_neo.py`
- `Qwen3`: `.venv/Lib/site-packages/transformers/models/qwen3/modeling_qwen3.py`
- `DynMoE`的动态 MoE：`ref\DynMoE\DeepSpeed-0.9.5\deepspeed\moe\sharded_moe.py`
- `DynMoE`的 SDL ：`ref\DynMoE\DeepSpeed-0.9.5\deepspeed\moe\loss.py`

---

## 项目架构索引

- `/.roo/rules` - 设计文档和项目Agent指令
- `/docs/spikes` - 技术预研文档
- `/exp/pisd` - PISD 技术预研实验区
- `/exp/plsd` - PLSD 技术预研实验区
- `/exp/dynsiha` - DynSIHA 技术预研实验区
- `/src/arc2` - 各子目标定稿后，实施demo模型和 ARC-AGI-2 的训练代码
- `/src/arc3` - 后续迭代架构模型及针对ARC-AGI-3 的训练代码
