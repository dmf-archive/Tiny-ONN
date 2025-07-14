# ΔSC 扫描器数据流序列图

**版本:** 1.0
**日期:** 2025-07-15

## 1. 概述

本序列图详细描述了在一次完整的扫描任务中（以 `scripts/batch_scanner.py` 为例），数据从输入准备到最终写入 `.mscan` 文件的完整生命周期。它清晰地展示了调度器、扫描核心、PyTorch 模型及 IO 模块之间的交互顺序和数据流动路径。

## 2. 序列图

```mermaid
sequenceDiagram
    actor User as User/CLI
    participant BS as Batch Scanner (Orchestrator)
    participant SE as ScannerEngine (Core)
    participant Model as PyTorch Model
    participant IO as MScanWriter (I/O)

    User->>+BS: run batch_scanner.py --params
    BS->>Model: load_model() & tokenizer()
    BS->>IO: MScanWriter(filepath, metadata)
    BS->>SE: ScannerEngine(model)

    loop For each sample in dataset
        BS->>BS: Prepare input_ids & labels
        
        BS->>+SE: with ScannerEngine(model) as scanner:
        SE->>Model: attach_hooks()
        Note right of SE: Registers forward & tensor hooks
        
        BS->>+Model: loss = model(inputs, labels)
        Model-->>SE: forward_hook(output)
        Note right of SE: Captures and stores activations
        Model-->>-BS: returns loss
        
        BS->>+Model: loss.backward()
        Model-->>SE: tensor_hook(grad)
        Note right of SE: Captures and stores gradients
        Model-->>-BS: backward pass complete
        
        BS->>SE: get_collected_data()
        SE-->>-BS: returns {activations}, {gradients}
        
        SE->>-Model: remove_hooks()
        Note right of SE: Hooks are detached
        
        BS->>BS: Process data into NumPy records
        BS->>+IO: writer.append_records(records, seq_info)
        IO->>IO: Write binary data to .mscan file
        IO-->>-BS: Records appended
    end
    
    BS->>+IO: writer.close()
    IO->>IO: Update header with final counts
    IO-->>-BS: File closed
    BS-->>-User: Scan Complete
```

## 3. 流程详解

1. **初始化 (Initialization)**:
    - 用户通过命令行启动 `batch_scanner.py`。
    - **调度器 (`Batch Scanner`)** 加载 PyTorch 模型和分词器。
    - 调度器初始化 `MScanWriter`，创建 `.mscan` 文件并写入初始元数据。
    - 调度器初始化 `ScannerEngine`，并将其与模型关联。

2. **扫描循环 (Scan Loop)**:
    - 调度器遍历数据集中的每一个样本。
    - **进入 `with` 上下文**: `ScannerEngine` 被激活，它立即调用 `attach_hooks()`，在模型的所有目标模块和参数上注册**前向钩子**和**张量钩子**。

3. **前向传播 (Forward Pass)**:
    - 调度器调用 `model()` 执行前向传播。
    - 在计算过程中，每当一个被钩住的模块（如 `nn.Linear`）完成计算，其**前向钩子**被触发。
    - `ScannerEngine` 捕获该模块的**输出激活值**，并将其存储在内部的 `self.activations` 字典中。

4. **反向传播 (Backward Pass)**:
    - 调度器调用 `loss.backward()` 执行反向传播。
    - 当梯度计算回传到每一个被钩住的参数时，其**张量钩子**被触发。
    - `ScannerEngine` 捕获该参数的**最终梯度**，并将其存储在内部的 `self.gradients` 字典中。

5. **数据收集与持久化 (Data Collection & Persistence)**:
    - **离开 `with` 上下文**: `ScannerEngine` 的 `remove_hooks()` 方法被自动调用，清除所有钩子，确保模型恢复到干净状态。
    - 调度器调用 `get_collected_data()` 从 `ScannerEngine` 中取回包含所有激活值和梯度数据的字典。
    - 调度器将这些原始张量数据处理、量化，并打包成符合 `METRICS_DTYPE` 规范的 NumPy 记录数组。
    - 调度器调用 `MScanWriter` 的 `append_records()` 方法，将 NumPy 记录以二进制形式追加到 `.mscan` 文件末尾。

6. **完成 (Finalization)**:
    - 循环结束后，调度器调用 `writer.close()`。
    - `MScanWriter` 最后一次更新文件头，写入最终的记录总数，然后关闭文件。
    - 任务完成。
