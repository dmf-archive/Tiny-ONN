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

## 关键源代码文件

以下是重要的参考源代码在工作区内的路径：

- `ARS2-Neo`: `ref/F3EO/optimizer/ars2_neo.py`
- `Qwen3`: `.venv/Lib/site-packages/transformers/models/qwen3/modeling_qwen3.py`
- `DynMoE`的动态 MoE：`ref\DynMoE\DeepSpeed-0.9.5\deepspeed\moe\sharded_moe.py`
- `DynMoE`的 SDL ：`ref\DynMoE\DeepSpeed-0.9.5\deepspeed\moe\loss.py`

## 重要架构设计文档

> 注意：以下文档因篇幅过长，为保持规则库简洁，仅在此提供路径索引，不直接并入 `.roo/rules`。

- `ARS-Series`: `ref\F3EO\.roo\rules\ARS-Series.md`
  - 简介：探讨在黎曼流形上滑行的优化算法家族（ARS/ARS2-Neo），核心在于能量-几何解耦（Energy-Geometry Decoupling）与测地线优化。
- `FARS`: `ref/FARS.md`
  - 简介：Fisher-Aware Routing Shaping。利用优化器二阶矩（Fisher 近似）量化专家认知代价，实现自发的功能分化与稀疏路由。
- `RDS-ACT`: `ref/RDS-ACT.md`
  - 简介：Recursive DynSIHA 与 Adaptive Computation Time。从 Q-Learning 演进至 PLSD（逐层推测解码）范式，实现动态递归推理。
