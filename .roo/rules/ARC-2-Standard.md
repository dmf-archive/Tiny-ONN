# ARC-2 极简训练框架标准规范 (v2.2)

> **核心目标**: 实现 **模型架构 (Model)** 与 **训练流程 (Pipeline)** 的彻底解耦，并完成 DynSIHA 的高性能原子化重构。
> **核心原则**: 任务逻辑与优化器（ARS2-Neo）深度耦合以换取简洁性，但模型必须是可插拔的“纯净计算基座”。
> **生产级对齐**: 模型基础设施（RoPE, RMSNorm, KV Cache）必须向 `transformers` 生产级标准对齐，确保推理效率与外推稳定性。

## 1. 目录结构与模块功能设计

```text
Tiny-ONN/
└── src/
    ├── models/                # 【纯粹模型层】定义神经网络拓扑，严禁包含训练逻辑
    │   ├── dynsiha/           # DynSIHA架构命名空间
    │   │   ├── capr.py        # 核心路由模块：高性能向量化实现，消除 Python 循环
    │   │   ├── flat.py        # 生产形态：Block 级组装的 FlatDynSIHA
    │   │   └── recursive.py   # 实验形态：全局递归再入实现 (Recursive DynSIHA)
    │   └── baseline/          # 基线模型：标准 Transformer 实现
    ├── tasks/                 # 【任务逻辑层】封装特定任务的训练与评估流程
    │   └── arc/               # ARC-AGI 任务专用目录
    │       ├── data/          # 数据增强、Loss Masking (H1) 与 ADL (H2) 逻辑
    │       ├── trainer.py     # 核心训练器：锁定 ARS2-Neo，执行原子重塑工作流
    │       ├── shaper.py      # 路由塑造器：封装 FARS/SARS 逻辑，计算 meta_loss
    │       └── observer.py    # 观察者：专注于 Rich 可视化、指标格式化与日志
    └── optimizers/            # 【优化算子层】
        └── ars2_neo.py        # SOTA 优化器：能量-几何解耦与 AGA 机制
```

## 2. 核心协议规范

### 2.1 模型协议 (Model Protocol)

- [x] **纯粹性**: 模型必须是标准的 `nn.Module`，严禁依赖特定的训练器状态或全局变量。
- [x] **元数据暴露**: 动态路由模型必须通过 `return_dict` 模式暴露 `routing_logits` 或 `routing_info`。
- [x] **生产级适配**: 必须集成 `RoPE` 位置编码与 `RMSNorm`，并支持 `transformers.cache_utils.Cache` 协议。
- [ ] **内生 ASI**: 模型应集成基于 Surrogate Gap 的自适应调度 (Active Sharpening Inference) 作为默认特性。
- [ ] **FARS 支持**: 模型应可选地暴露 `active_features` 以支持精确的梯度计算。

### 2.2 路由塑造器 (RoutingShaper)

- [x] **职责分离**: 负责计算 `meta_loss` (FARS/SARS) 和路由相关的诊断指标（如专家利用率、熵）。
- [ ] **非侵入式注入**: 通过 `shaper.wrap_model(model)` 注入必要的 hooks，严禁修改模型主干代码。
- [x] **优化器协同**: 允许访问优化器的 `state_dict`（特别是 Adam 的二阶矩 `v_t`）以实现无量纲的 FARS 成本计算。

### 2.3 训练与优化规范

- [x] **优化器锁定**: 默认锁定使用 **ARS2-Neo**，充分利用其测地线滑行与平坦度约束能力。
- [x] **原子重塑**: 训练脚本必须支持 `optimizer.step(closure)` 模式，确保 SAM 扰动逻辑的正确执行。
- [x] **Just in Fail**: 严禁在训练流程中使用 `try...except` 静默捕获异常。

## 4. 原子化重构路线图 (Module Surgery)

### Phase 1: 生产级算子注入 (Production Alignment)

- [ ] 实现 `DynSIHARotaryEmbedding` 与 `DynSIHARMSNorm`。
- [ ] 重构 `DynSIHAAttention` 以支持 `past_key_values` (Cache) 协议。
- [ ] 标准化 `FlatDynSIHA` 接口，返回 `CausalLMOutputWithPast`。

### Phase 2: 统一模型库构建 (Standardization)

- [x] 建立 `src/models/dynsiha/` 统一命名空间。
- [x] 实现标准的 `FlatDynSIHA` 基类，移除所有任务特定的硬编码逻辑。
- [x] 确保模型支持标准的 `transformers` 风格调用接口。

### Phase 3: 解耦训练器实现 (Decoupled Trainer)

- [x] 编写符合 ARC-2 规范的极简 `src/tasks/arc/trainer.py`。
- [x] 实现任务逻辑（Grid 处理、Loss Masking）与模型架构的彻底分离。
- [x] 集成 `Shaper` 与 `Observer` 协议，完成闭环。

---
**签发人**: Roo (AI Architect)
**日期**: 2026-01-24
