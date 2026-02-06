# ARC-2 极简训练框架

> 核心目标: 实现 模型架构 (Model) 与 训练流程 (Pipeline) 的彻底解耦，并完成 DynSIHA 的高性能原子化重构。
> 核心原则: 任务逻辑与优化器（ARS2-Neo）深度耦合以换取简洁性，但模型必须可插拔。
> 生产级对齐: 模型基础设施（RoPE, RMSNorm, KV Cache）必须向 `transformers` 生产级标准对齐，确保推理效率与外推稳定性。

## 目录结构与模块功能设计

```text
Tiny-ONN/
└── src/
    ├── models/
    │   └── dynsiha/
    │       ├── shared/
    │       ├── flat/
    │       │   ├── configuration_flat_dynsiha.py
    │       │   └── modeling_flat_dynsiha.py
    │       └── recursive/
    │           ├── configuration_recursive_dynsiha.py
    │           └── modeling_recursive_dynsiha.py
    ├── tasks/
    │   └── arc/
    │       ├── data.py
    │       ├── trainer.py
    │       ├── shaper.py
    │       └── observer.py
    └── optimizers/
        └── ars2_neo.py
```
