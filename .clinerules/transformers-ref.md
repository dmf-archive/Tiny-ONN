# Hugging Face `transformers` 库详细文档

Hugging Face `transformers` 库提供了一个全面的框架，用于处理最先进的机器学习模型，为各种架构提供统一的 API，为模型处理提供核心组件，以及强大的训练和优化基础设施。该库集中了模型定义，使其与不同的训练框架和推理引擎兼容。它强调易用性和效率，允许用户以最少的努力加载和微调预训练模型。

## 模型架构

`Transformers` 库支持各种模型架构，主要分为仅编码器、仅解码器和编码器-解码器模型。

### 仅编码器模型 (Encoder-Only Models)

仅编码器模型通常用于需要理解输入的任务，例如文本分类、token 分类和问答。BERT 是仅编码器模型的一个示例。

### 仅解码器模型 (Decoder-Only Models)

仅解码器模型专为生成任务而设计，模型根据输入和先前生成的 token 顺序生成输出。GPT2 是仅解码器模型的一个示例，常用于文本生成。

### 编码器-解码器模型 (Encoder-Decoder Models)

编码器-解码器模型，也称为序列到序列模型，适用于摘要和翻译等任务，其中输入序列被转换为输出序列。BART 是编码器-解码器模型的一个示例。`VisionEncoderDecoderModel` 类允许使用预训练的视觉模型作为编码器和预训练的语言模型作为解码器来初始化图像到文本模型。类似地，`TFEncoderDecoderModel` 可用于使用任何预训练的自编码模型作为编码器和任何预训练的自回归模型作为解码器来初始化序列到序列模型。

## 核心组件

该库的核心组件包括 `Tokenizer`、`Config`、`Model`、`Pipeline` 和 `Auto` 类，它们协同工作，为各种任务提供简化的工作流程。

### 分词器 (Tokenizer)

`Tokenizer` 负责通过将原始文本转换为模型可以处理的数字表示（token）来为模型准备文本数据。分词器的基类是 `PreTrainedTokenizerBase`，它为编码、解码和处理特殊 token 提供通用接口。还有用于慢速实现的 `PreTrainedTokenizer` 和用于快速实现的 `PreTrainedTokenizerFast`。

### 配置 (Config)

`Config`（由 `PretrainedConfig` 表示）管理模型配置，包括 `vocab_size`、`hidden_size`、`num_attention_heads` 和 `num_hidden_layers` 等参数。它处理序列化并与 Hugging Face Hub 集成。配置文件可以加载和保存到磁盘，但它只影响模型的配置，不加载模型权重。

### 模型 (Model)

`Transformers` 库中的模型通常是 `PreTrainedModel`（用于 PyTorch）、`TFPreTrainedModel`（用于 TensorFlow）或 `FlaxPreTrainedModel`（用于 Flax）的子类。这些基类提供通用功能，例如加载/保存模型、管理设备放置和定义前向传递接口。

### Pipeline

`Pipeline` 是一个高级 API，旨在通过预训练模型在各种模态和任务中进行轻松快速的推理。它抽象了模型加载、tokenization 和后处理的复杂性，允许用户通过单个函数调用执行文本生成、图像分割和自动语音识别等任务。例如，可以使用 `pipeline(task="sentiment-analysis")` 创建文本分类 pipeline。

### Auto 类 (Auto Classes)

`Auto` 类（`AutoConfig`、`AutoModel`、`AutoTokenizer` 等）提供统一简化的 API，用于根据模型类型和配置动态加载适当的实现类。该系统根据从 `config.json` 文件中提取的 `model_type` 字段自动选择正确的模型、分词器或配置。例如，`AutoModel.from_pretrained("google-bert/bert-base-cased")` 将实例化一个 `BertModel`。`Auto` 类使用 `OrderedDict` 映射（例如 `CONFIG_MAPPING_NAMES`、`MODEL_MAPPING_NAMES`、`TOKENIZER_MAPPING_NAMES`）将模型类型链接到其各自的实现类名。

## 训练基础设施

`Transformers` 库提供强大的训练基础设施，主要通过 `Trainer` 类和 `TrainingArguments`。

### Trainer

`Trainer` 类是 PyTorch 模型的完整训练和评估循环，针对 🤗 Transformers 进行了优化。它抽象了样板代码，允许用户专注于模型、数据集和训练设计选择。`Trainer` 的关键参数包括 `model`、`args`（`TrainingArguments` 的实例）、`data_collator`、`train_dataset`、`eval_dataset`、`processing_class`、`compute_metrics` 和 `optimizers`。

### TrainingArguments

`TrainingArguments` 是一个类，用于配置各种超参数和训练功能，例如 `output_dir`、`learning_rate`、`per_device_train_batch_size`、`num_train_epochs`、`eval_strategy`、`save_strategy` 和 `push_to_hub`。它支持分布式训练、混合精度训练（`bf16` 或 `fp16`）以及将模型保存到 Hub 等功能。

## 内存优化

该库提供多种内存优化技术，包括量化和缓存。

### 量化 (Quantization)

量化涉及降低模型权重和激活的精度，以节省内存并加快推理速度。与 `Transformers` 集成的 `torchao` 库支持各种量化技术，如 A16W8 Float8 动态量化、A8W8 Int8 动态量化和 A16W4 Int4 仅权重量化。它还支持用于长上下文推理的 `KV Cache Quantization`，以降低内存。

### 缓存 (Caching)

缓存，特别是 KV（Key-Value）缓存，对于优化生成过程中的自回归模型性能至关重要。它存储先前计算的键和值向量，以避免为每个新 token 重新计算它们，从而减少计算时间并提高响应速度。`transformers` 库提供多种 `Cache` 类，例如 `DynamicCache`、`Static Cache`、`Offloaded Cache` 和 `Quantized Cache`，每种都针对不同的内存或速度要求进行了优化。大多数模型的默认缓存是 `DynamicCache`。可以通过在 `~GenerationMixin.generate` 中设置 `use_cache=False` 来禁用缓存。
