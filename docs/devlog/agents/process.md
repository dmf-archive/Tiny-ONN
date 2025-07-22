# 调试移交与最终策略报告

## 1. 任务目标

将一个标准的 `Qwen/Qwen3-0.6B` 模型，通过“模型手术”的方式，将其 MLP 层替换为我们自定义的 `DynMoE` 层，并最终通过一个蒸馏训练测试。

## 2. 调试过程总结与根本原因分析

我在执行此任务时遇到了一系列严重且连锁的错误，最终导致了致命的访问冲突（`access violation`）和 VSCode 崩溃。这表明问题比最初预想的更为复杂。

1. **`FileNotFoundError`**: 任务开始时，核心文件如 `tiny_onn/config.py`, `tiny_onn/model.py` 和 `train.py` 缺失。我根据项目规范，从头开始创建了这些文件。

2. **`TypeError` 与 `AttributeError`**: 在 `tiny_onn/modular.py` 的 `forward` 方法中，对 `past_key_values` 的处理不当，导致了与 `DynamicCache` 相关的一系列类型和属性错误。我对 `forward` 方法的签名和逻辑进行了多次迭代修复。

3. **`ValueError` (Hugging Face Auto\* Class Registration)**:

   - **模型注册**: `AutoModelForCausalLM.register` 失败，因为 `TinyOnnForCausalLM` 继承的 `config_class` 与注册时传入的 `TinyOnnConfig` 不匹配。通过在 `TinyOnnForCausalLM` 中明确覆写 `config_class = TinyOnnConfig` 解决了此问题。
   - **Tokenizer 注册**: `AutoTokenizer.register` 失败，因为我错误地将一个 `FastTokenizer` 作为 `slow_tokenizer` 传递。通过为 `AutoTokenizer.register` 同时提供 `Qwen2Tokenizer` (slow) 和 `Qwen2TokenizerFast` (fast) 解决了此问题。

4. **`Windows fatal exception: access violation`**: 这是最严重的问题，它出现在 `pytest` 运行期间，即使在解决了所有静态检查和注册问题之后依然存在。堆栈跟踪指向了 PyTorch 底层的 `torch.nn.modules.linear.Linear.__init__`，该过程在 `perform_surgery` 函数中初始化模型时被调用。

## 3. 最终诊断与移交建议

致命的访问冲突错误强烈暗示问题源于以下一个或多个方面：

- **环境/依赖冲突**: 可能是 CUDA、`bitsandbytes`、PyTorch 或 `transformers` 之间的版本不兼容或安装损坏。
- **模型初始化失败**: 尽管类型检查通过，但在 `TinyOnnForCausalLM` 的初始化过程中，传递给底层 `Qwen3Moe` 模型的参数可能存在逻辑错误，导致了内存损坏。
- **硬件问题**: 虽然可能性较小，但不排除底层硬件或驱动程序存在问题。

**移交后的建议步骤**:

接手此工作的工程师应放弃继续在现有代码上进行小修小补的策略，并采取更系统的方法：

2. **隔离问题**:

   - **暂停 `test_distillation.py`**: 首先集中精力让最基本的 `test_surgery.py` 通过。
   - 在 `scripts/perform_surgery.py` 中添加详细的打印语句，跟踪 `TinyOnnConfig` 在传递给 `TinyOnnForCausalLM` 之前的所有属性值，以验证其正确性。
   - 尝试在不加载 `base_model.state_dict()` 的情况下初始化 `TinyOnnForCausalLM`，看是否仍然发生崩溃。这将有助于确定问题是在模型结构定义阶段还是在权重加载阶段。

3. **代码审查**:
   - 仔细审查 `tiny_onn/modular.py` 中 `DynMoE` 和 `TinyOnnDecoderLayer` 的 `__init__` 方法。确认所有从 `config` 中读取的参数都被正确地传递给了底层的 `Qwen3MLP` 和 `Qwen3MoeAttention`。

我对未能解决这个深层次问题表示歉意，并希望这份详细的报告能为下一位工程师提供一个清晰的起点。
