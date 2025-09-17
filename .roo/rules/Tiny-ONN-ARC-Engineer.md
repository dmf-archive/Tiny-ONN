# Tiny-ONN-ARC 工程备忘录

`Latest update: 2025-09-18`

## Single-View Grokking

目标: 最大化梯度信号纯度，强制 SAPS 自组织

训练循环结构: `Task Loop → View Loop → Grokking Loop`

收敛条件:

- `loss < 0.01`
- `token_accuracy > 1.00`
- 达到 `MAX_INNER_STEPS` 上限

### 原理

- **梯度纯度最大化**: 通过单视角过拟合避免梯度平均，确保梯度信号精确指向特定锁孔结构
- **选择压力**: 暴力过拟合单个视角，强制学习可分离的知识表示
- **批次约束**: `batch_size = 1`（禁止跨任务梯度累积和跨视角批量处理，防止破坏余弦路由的非线性流形）
- **双优化器配置**:
  - `optimizer_meta`: 负责 `proto_weight`, `gate_param`，学习率 3e-2，快速学习路由策略
  - `optimizer_main`: 负责 `mu_weight`，学习率 3e-3，稳定学习计算核心
