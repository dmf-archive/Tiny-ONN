# Tiny-ONN-ARC 训练框架设计蓝图 (v1.0)

## 1. 设计目标
构建一个高度模块化、声明式配置驱动的 ARC (Abstraction and Reasoning Corpus) 任务训练框架。该框架旨在将 `Tiny-ONN` 的核心研究成果（如 `DynSIHA`, `RMSuon`, `SARS`）以标准化的方式集成到 `transformers` 生态中。

## 2. 核心架构

### 2.1 声明式配置系统
采用 `YAML` 作为实验定义的唯一来源，通过 `pydantic` 或 `dataclasses` 进行解析和校验。

- **Config 层次**:
  - `ModelConfig`: 定义模型拓扑（层数、隐藏维度、路由机制类型）。
  - `OptimizerConfig`: 定义优化器策略（ARS 扰动强度、RMSuon 正交化频率）。
  - `DataConfig`: 定义 ARC 序列化参数与增强策略。
  - `TrainingConfig`: 定义训练循环控制（PI 阈值早停、日志频率）。

### 2.2 模型插拔机制 (Model Registry)
所有模型架构（如 `FlatDynSIHA`, `RecursiveDynSIHA`）需封装为 `transformers.PreTrainedModel`。

- **Registry**: 维护一个 `MODEL_TYPE_TO_CLS` 映射。
- **Wrapper**: 提供一个统一的 `ArcModelWrapper`，负责处理 ARC 特有的输入（如 `coords`）并将其映射到标准 `transformers` 接口。
- **原子重塑**: 确保模型内部组件（如 `SPL` 模块）可以独立替换而不影响整体计算图。

### 2.3 优化器集成方案 (Optimizer Registry)
复用并扩展 `F3EO` 的优化器管理逻辑。

- **声明式注册**: 在 `optimizer/__init__.py` 中定义优化器元数据。
- **自动参数分组**: 框架根据参数名称或形状自动分配优化器组：
  - `2D Weights (Linear/Conv)` -> `RMSuon` / `Muon`
  - `1D Params (Bias/LayerNorm)` -> `AdamW`
  - `Routing Params` -> 专用低学习率 `AdamW`

### 2.4 ARC 数据流水线 (Data Pipeline) - 向量化重构
为了解决 Python 循环带来的性能瓶颈和 IPC 开销，数据流水线将进行**全向量化**重构。

#### A. 架构设计
- **原则**: "Tensor In, Tensor Out"。Worker 进程与主进程之间仅传输 `torch.Tensor`，利用共享内存实现零拷贝通信。
- **`ArcTaskDataset`**:
  - 职责：加载任务，执行**向量化增强**（几何变换+颜色置换）。
  - 输出：返回 `dict[str, torch.LongTensor]`，严禁返回 Python List。
- **`VectorizedSerializer`**:
  - 替代现有的 `GridSerializer`。
  - 实现：摒弃 Python `for` 循环，使用 `torch.flatten`, `torch.cat`, `torch.repeat_interleave` 等算子在 Tensor 层面完成网格到 Token 流的转换。
- **`TokenBucketSampler`**:
  - 职责：基于预计算的序列长度进行动态分桶，生成 Batch 索引，确保 GPU 负载均衡。

#### B. 性能对比
| 阶段 | 旧方案 (Current) | 新方案 (Proposed) |
| :--- | :--- | :--- |
| **增强** | Trainer 主线程串行执行 | Worker 进程并行执行 (Tensor Ops) |
| **序列化** | Python 双重循环 (极慢) | PyTorch 向量化操作 (极快) |
| **通信** | 无 (单进程) | 共享内存 Zero-Copy (极快) |
| **瓶颈** | CPU 序列化阻塞 GPU | 几乎无 (IO/GPU Bound) |

## 3. 可观测性与 PI 指标引擎

### 3.1 PI (Predictive Integrity) 计算
PI 是框架的核心反馈信号，用于驱动早停和课程学习。
- `PI = exp(-(CrossEntropyLoss + GradNorm))`
- 框架需支持在每个 step 实时计算并记录 PI。

### 3.2 实时可视化 (Rich Observer)
集成 `rich` 库，提供以下功能：
- **彩色栅格渲染**: 在终端直接显示模型预测结果。
- **指标面板**: 实时显示 Loss, Acc, PI, τ (Entropy) 和梯度范数。

## 4. 实施路线图 (Implementation Roadmap)

1. **Phase 1: 基础骨架**: 建立 `configs/`, `models/`, `optimizers/`, `data/` 目录结构及 Registry 机制。
2. **Phase 2: 模型重写**: 使用 `transformers` 规范重写 `DynSIHA` 参考实现。
3. **Phase 3: 训练器集成**: 实现支持 PI 早停和动态 Batch 的 `UnifiedTrainer`。
4. **Phase 4: 验证**: 在 ARC-AGI-2 训练集上进行基线对齐测试。

---
*Document End.*
