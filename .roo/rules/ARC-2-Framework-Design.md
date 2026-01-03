# ARC-2 极简训练框架设计规范 (v2.0)

> **核心目标**: 实现 **模型架构 (Model)** 与 **训练流程 (Pipeline)** 的解耦。
> **核心原则**: 任务逻辑（ARC-2）与优化器（ARS2-Neo）可以深度耦合以换取简洁性，但必须确保模型是可插拔的，且路由元学习（FARS/SARS）不侵入主流程。

## 1. 核心架构原则

### 1.1 模型协议 (Model Protocol)

- **纯粹性**: 模型必须是标准的 `nn.Module`，不依赖特定的训练器状态。
- **元数据暴露**: 动态路由模型必须通过 `return_dict` 暴露 `routing_logits`。
- **FARS 支持**: 为了支持 SARS/FARS 等路由元学习框架，模型应可选地暴露 `active_features`（用于精确梯度计算，若使用 Fisher 近似则非必须）。

### 1.2 路由塑造器 (RoutingShaper)

- **职责**: 负责计算 `meta_loss` (SARS/FARS) 和路由相关的诊断指标。
- **解耦**: `Shaper` 作为一个独立的策略类，通过 `shaper.wrap_model(model)` 注入必要的 `hooks`（如果需要捕获中间梯度）。
- **优化器协同**: `Shaper` 能够访问优化器的 `state_dict`（特别是 Adam 的二阶矩 `v_t`）以实现无量纲的 FARS 成本计算。

### 1.3 观察者协议 (Observer Protocol)

- **职责分离**: `Observer` 不再负责计算核心指标，仅负责 **格式化输出** 和 **可视化**。
- **数据源**: `Observer` 接收由 `Trainer` 和 `Shaper` 汇总后的 `metrics` 字典。

### 1.4 优化器锁定

- 默认锁定使用 **ARS2-Neo**。
- 训练脚本直接调用 `ARS2-Neo` 的 `step(closure)`，利用其内置的 SAM 和 AGA 逻辑。

## 2. 极简目录结构

```text
src/
├── models/                # 纯粹的模型定义
│   ├── dynsiha/           # 包含 CAPR 路由的模型
│   └── baseline/          # 标准 Transformer
└── tasks/
    └── arc/
        ├── data/          # 数据处理与增强
        ├── trainer.py     # 核心训练循环 (耦合任务逻辑与 ARS2-Neo)
        ├── shaper.py      # FARS/SARS 路由塑造逻辑 (可插拔)
        └── observer.py    # 专注于 Rich 可视化与日志
```

## 3. 关键工作流：解耦的 `train_step`

```python
def train_step(model, batch, optimizer, shaper=None):
    def closure():
        optimizer.zero_grad()
        # 1. 前向传播
        outputs = model(batch["input_ids"], labels=batch["labels"])
        main_loss = outputs.loss
        
        # 2. 路由塑造 (静默注入)
        meta_loss = 0
        if shaper is not None and "routing_logits" in outputs:
            # Shaper 内部可能使用 optimizer.state 获取 Fisher 信息
            meta_loss = shaper.compute_meta_loss(outputs, optimizer)
        
        total_loss = main_loss + meta_loss
        total_loss.backward()
        return total_loss, outputs

    # ARS2-Neo 的 step 会处理 SAM 扰动和闭包调用
    loss, outputs = optimizer.step(closure)
    return outputs, loss
```

## 4. 演进路线

1. **Phase 1**: 剥离 `CAPR` 模块，使其成为 `src/models/dynsiha` 的核心组件。
2. **Phase 2**: 实现 `src/tasks/arc/shaper.py`，封装 FARS 逻辑。
3. **Phase 3**: 编写极简 `trainer.py`，支持传入任意符合协议的模型。
4. **Phase 4**: 在 `exp/arc_flat_dynsiha/` 中调用新框架进行实验。

---
**设计者**: Roo (AI Architect)
**日期**: 2026-01-22
