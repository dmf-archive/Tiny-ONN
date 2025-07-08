# Tiny-ONN 开发计划与技术参考

## 摘要

本开发计划详细阐述了 Tiny-ONN (Tiny-Optimized Neural Network) 的构建、训练与评估策略。Tiny-ONN 旨在通过一种创新的“知识炼金术”工作流，从大型预训练语言模型（如 Qwen3）中外科手术式地提取并重组核心知识单元，形成一个由稀疏专家模块和动态路由系统组成的紧凑、高效且可解释的模型。该计划涵盖了从模型加载、激活与梯度分析、专家提取，到持续预训练、路由系统适配，以及最终的性能评估与可解释性分析的全过程。我们旨在实现一个在性能上接近原始大型模型，但在计算效率和可解释性上具有显著优势的下一代稀疏模型范式。

## 1. 引言

### 1.1 背景与动机

当前，大型语言模型（LLMs）在各种自然语言处理任务中展现出卓越的能力。然而，其庞大的参数量带来了高昂的计算成本、巨大的内存占用以及部署的复杂性，这限制了它们在资源受限环境中的应用。同时，LLMs 内部的知识表示和决策过程往往被视为“黑箱”，缺乏透明度和可解释性，这在关键应用领域构成了挑战。

为了解决这些问题，稀疏专家模型（Sparse Mixture-of-Experts, SMoE）提供了一个有前景的方向。SMoE 模型通过将模型能力分散到多个专家网络中，并在推理时仅激活其中一小部分专家，从而在保持甚至提升性能的同时，有效降低了计算量。然而，现有 SMoE 模型通常从随机初始化开始训练专家，或者在预训练阶段就引入稀疏性，这可能导致专家分化不充分或训练过程复杂。

本计划提出 Tiny-ONN 架构，旨在通过一种“知识炼金术”的方法，从一个已充分预训练的稠密 LLM 中，外科手术式地提取并重组其核心知识单元。这种方法利用了大型模型在预训练过程中已经形成的强大知识表示，通过识别和分离其中最活跃、最具贡献的参数群，将其转化为 Tiny-ONN 的初始专家模块。这种“动态选择性抽样”策略有望显著提升初始专家模块的质量，加速后续的持续训练和专家分化过程。

### 1.2 Tiny-ONN 核心理念

Tiny-ONN 的核心理念在于将大型模型的密集智能，转化为由精巧“概念细胞”组成的**数字生命**。这包括以下几个关键点：

- **知识提取 (Knowledge Extraction)**：通过对大型预训练模型（如 Qwen3）在特定任务（如 SFT）上的完整激活模式和梯度强度进行精细化分析，计算每个参数的“预测完整性微分”（PIDiff）或“协同贡献度”，并动态提取高密度因果网络。这些网络被视为模型中承载特定知识或功能的“概念细胞”。
- **动态选择性抽样 (Dynamic Selective Sampling)**：区别于传统的随机初始化或预设稀疏性，Tiny-ONN 的专家模块是根据其在实际任务中的“高激活 + 低梯度 = 高协同”原则动态选择和组装而来的。这确保了初始专家模块的质量和相关性，并实现了极致的参数压缩。
- **稀疏专家重组 (Sparse Expert Recomposition)**：提取出的“干细胞”被重组为 Tiny-ONN 的专家模块。这些专家模块在后续的持续预训练中，通过稀疏混合专家（SMoE）机制进行进一步的功能分化和优化。
- **自适应路由 (Adaptive Routing)**：设计一个全新且独立的路由系统，能够根据输入内容和任务需求，高效、精确地将请求路由到最合适的专家组合。该路由系统将与专家模块协同进化，实现智能的知识调度。
- **可解释性 (Interpretability)**：通过对专家激活模式的可视化和语义标注，揭示模型内部的知识分区和决策机制，从而提升模型的透明度和可信度。

## 2. 核心概念与技术基础

Tiny-ONN 的构建依赖于多个前沿的深度学习概念和 PyTorch 框架的强大功能。本章将详细介绍这些核心概念及其在 Tiny-ONN 中的应用。

### 2.1 Dynamic Selective Sampling

动态选择性抽样是 Tiny-ONN 知识提取阶段的核心策略。它摒弃了传统 SMoE 模型中专家模块随机初始化或预设稀疏性的做法，转而从一个已充分预训练的稠密大型模型中，根据其在特定任务上的实际表现，动态地识别和提取最具价值的参数子集。

**核心思想：**
一个大型预训练模型在经过海量数据训练后，其内部的参数已经形成了复杂的知识表示。并非所有参数在所有任务上都同等重要。动态选择性抽样旨在通过以下步骤实现：

1. **行为观察**：在特定任务（如 SFT）上运行大型模型，观察其内部各参数群的激活模式和对任务损失的贡献。
2. **价值评估**：基于激活值、梯度信息和激活频率等指标，量化每个参数群的“活跃度”和“重要性”。
3. **智能提取**：根据评估结果，选择并提取那些在目标任务上表现出高活跃度和高贡献度的参数群，作为 Tiny-ONN 的初始专家模块。

**优势：**

- **高质量初始化**：提取的专家模块继承了大型模型已有的知识，避免了从零开始训练的冷启动问题，显著缩短了收敛时间。
- **任务相关性**：专家模块的提取是基于其在目标任务上的实际表现，确保了其与 Tiny-ONN 预期功能的紧密相关性。
- **效率提升**：通过聚焦于模型中最“有用”的部分，减少了不必要的参数量，从而提升了训练和推理效率。

### 2.2 Knowledge Alchemy

“知识炼金术”是 Tiny-ONN 项目的宏观愿景，它将整个开发过程视为一个将“巨石”（大型稠密模型）转化为“黄金”（高效、可解释的稀疏专家模型）的炼金过程。

**炼金步骤：**

1. **分解 (Decomposition)**：通过精细化数据收集与“预测完整性微分”（PIDiff）计算，深入到隐藏维度的内部，识别模型中每个参数的协同贡献度。
2. **提纯 (Purification)**：通过构建特征向量和聚类分析，动态识别并分离出功能上高度相关、协同工作的参数集合，形成“高密度因果网络”。
3. **重组 (Recomposition)**：从聚类结果中智能选择和组装参数，形成专家模块。这些模块可能由来自Qwen3-4B模型不同层、不同矩阵的非连续参数组成，需要稀疏矩阵表示和自定义计算核。
4. **催化 (Catalysis)**：通过持续预训练（特别是 SMK 策略），催化专家模块进一步分化，并优化路由系统的调度能力。
5. **升华 (Sublimation)**：最终形成一个在性能、效率和可解释性上均达到新高度的 Tiny-ONN 模型。

“知识炼金术”强调了从现有智能中提取、精炼和重组知识的理念，而非简单地从头开始构建。这不仅是一种技术方法，更是一种对模型智能本质的探索。

### 2.3 Sparse Mixture-of-Experts, SMoE

稀疏专家模型是 Tiny-ONN 的基础架构。它通过引入多个专家网络和一个门控网络（路由器）来实现计算的稀疏性。

**基本结构：**

- **专家网络 (Expert Networks)**：通常是前馈网络 (FFN) 或其他子模块，每个专家负责处理输入的不同方面或特定类型的知识。Tiny-ONN 的专家模块将从 Qwen3 中提取。
- **门控网络/路由器 (Gating Network/Router)**：一个小型神经网络，根据输入决定激活哪些专家。在稀疏 SMoE 中，路由器通常会选择 `top-k` 个专家进行激活，其中 `k` 远小于专家总数。

**工作原理：**
对于给定的输入，路由器会计算每个专家的“分数”或“权重”，然后选择分数最高的 `k` 个专家。输入数据只通过这 `k` 个被选中的专家进行计算，从而显著减少了总的计算量。未被激活的专家在当前前向传播中不参与计算，也不更新梯度。

**SMoE 的优势：**

- **计算效率**：在推理时只激活部分专家，降低了 FLOPs。
- **模型容量**：尽管计算稀疏，但模型总参数量可以非常大，从而提升了模型容量和学习复杂模式的能力。
- **并行性**：专家模块可以并行计算，有利于分布式训练和推理。
- **潜在的可解释性**：理论上，不同的专家可以学习到不同的功能或知识领域，为模型的可解释性提供了基础。

### 2.4 PyTorch 在 Tiny-ONN 中的作用

PyTorch 作为当前主流的深度学习框架，为 Tiny-ONN 的开发提供了强大的支持和灵活性。其动态计算图、丰富的张量操作、自动求导机制以及模块化设计，是实现“知识炼金术”工作流的关键。

#### 2.4.1 张量与自动求导

- **张量 (Tensors)**：PyTorch 的核心数据结构，类似于 NumPy 的 `ndarray`，但支持 GPU 加速。在 Tiny-ONN 中，Qwen3 的模型权重、激活值、梯度以及所有中间计算结果都以张量形式存在。我们将利用 PyTorch 提供的张量操作（如切片、索引、数学运算、线性代数）来对 Qwen3 的参数进行提取和重组。
- **自动求导 (Autograd)**：`torch.autograd` 是 PyTorch 的自动微分引擎，它能够自动计算张量上所有操作的梯度。这是 Tiny-ONN 知识提取阶段评估参数“活跃度”和“贡献度”的基础。通过 `loss.backward()`，我们可以方便地获取模型参数的梯度，从而量化其对 SFT 任务损失的影响。对于更精细的梯度分析，`torch.autograd.grad` 提供了计算任意输出对任意输入梯度的能力，这对于精确评估特定参数的贡献至关重要。

#### 2.4.2 PyTorch Hooks 机制

PyTorch Hooks 是实现“动态选择性抽样”的核心技术。它们允许我们在不修改模型源代码的情况下，在模型的前向或反向传播过程中插入自定义逻辑，从而捕获中间激活值或梯度。

- **`module.register_forward_hook(hook_fn)`**：
  - **用途**：在指定模块（如 Qwen3 的 FFN 层或注意力输出层）的前向传播完成后执行 `hook_fn`。`hook_fn` 的签名通常为 `hook_fn(module, input, output)`，其中 `input` 是模块的输入张量，`output` 是模块的输出张量。
  - **应用**：在 Tiny-ONN 的知识提取阶段，我们将利用前向 Hook 来捕获 Qwen3 模型在 SFT 推理过程中特定层的激活值。这些激活值是评估参数活跃度的直接依据。
- **`module.register_backward_hook(hook_fn)`**：
  - **用途**：在指定模块的反向传播完成后执行 `hook_fn`。`hook_fn` 的签名通常为 `hook_fn(module, grad_input, grad_output)`，其中 `grad_input` 是模块输入张量的梯度，`grad_output` 是模块输出张量的梯度。
  - **应用**：虽然主要通过 `tensor.grad` 属性获取梯度，但反向 Hook 提供了更细粒度的控制，可以用于捕获特定模块的梯度流，辅助分析参数贡献。
- **`tensor.register_hook(hook_fn)`**：
  - **用途**：直接在张量上注册 Hook，用于捕获该张量的梯度。`hook_fn` 的签名通常为 `hook_fn(grad)`，其中 `grad` 是该张量的梯度。
  - **应用**：在 Tiny-ONN 中，我们将为 Qwen3 模型中每个参数张量注册 Hook，以便在 `loss.backward()` 后直接捕获这些参数的梯度，用于活跃度评估。

通过这些 Hook，我们可以在不侵入 Qwen3 模型结构的前提下，实现对其内部动态行为的全面监控和数据收集。

#### 2.4.3 模型量化与精度管理

在处理 Qwen3 这样的大型模型时，显存是一个关键限制。PyTorch 提供了量化功能，而 Hugging Face `transformers` 库则集成了 `bitsandbytes` 等工具，支持低精度（如 INT4、INT8）模型加载，以显著降低显存占用。

- **量化加载**：
  - **需求**：在显存受限的情况下，加载 Qwen3-4B 的量化版本进行推理，以便进行活跃参数的检测。
  - **PyTorch/Transformers 功能**：Hugging Face `transformers` 库的 `from_pretrained` 方法支持 `load_in_8bit` 或 `load_in_4bit` 参数，底层会调用 `bitsandbytes` 等库进行量化加载。
  - **重要性**：量化加载使得我们能够在有限的 GPU 资源上运行 Qwen3，从而进行知识提取阶段的推理和数据收集。
- **精度管理**：
  - **量化不影响参数映射**：量化只是改变了参数的数值表示（例如从 BF16 变为 INT8），但参数在模型结构中的逻辑位置和索引是保持不变的。这意味着即使模型是量化的，我们仍然可以通过 Hook 和梯度分析来识别“活跃参数”。
  - **反量化**：在提取这些活跃参数时，我们将它们从低精度（如 INT4）反量化回更高精度（如 BF16 或 FP32），以确保 Tiny-ONN 的专家模块拥有足够的精度，避免因精度损失而影响性能。PyTorch 的 `torch.quantization` 模块提供了相关的反量化工具，或者我们可以利用 `bitsandbytes` 提供的反量化功能。

#### 2.4.4 参数操作与重组

Tiny-ONN 的知识提取和专家重组过程涉及对模型参数张量进行精细的操作。PyTorch 提供了丰富的张量操作函数，能够满足这些需求。

- **`torch.nn.Parameter`**：模型参数的类型，是 `torch.Tensor` 的子类，但会被 `nn.Module` 自动注册为可训练参数。
- **`tensor.clone().detach()`**：
  - **用途**：复制参数值，并从计算图中分离。这对于在不影响原始模型计算图的情况下，对参数进行操作（如切分、合并）至关重要。
- **`torch.split()` / `torch.chunk()`**：
  - **用途**：用于将大型模型的权重张量拆分成小块，以实现参数的“网格化”。`split` 允许按指定大小或数量进行拆分，`chunk` 允许按指定块数进行拆分。
- **`torch.cat()`**：
  - **用途**：用于将选定的专家权重重新拼接起来，形成 Tiny-ONN 的专家模块。
- **`torch.zeros_like()`**：
  - **用途**：创建与现有张量形状相同且填充零的张量。这可以用于初始化非活跃或新创建的参数，或者作为占位符。

这些操作将使我们能够灵活地对 Qwen3 的参数进行切分、选择、提取和重组，从而构建 Tiny-ONN 的专家模块。

#### 2.4.5 PyTorch 编译与优化

虽然在知识提取阶段可能不会直接使用，但 PyTorch 的编译系统（如 TorchDynamo 和 TorchInductor）以及分布式训练功能，对于 Tiny-ONN 后续的持续预训练和性能优化至关重要。

- **`torch.compile` (TorchDynamo, TorchInductor)**：
  - **用途**：将 PyTorch 代码编译为优化的内核，显著提升模型训练和推理速度。
  - **应用**：在 Tiny-ONN 的持续预训练阶段，可以对专家模块和路由系统应用 `torch.compile`，以最大化训练效率。TorchDynamo 负责图捕获，将 PyTorch 操作序列提取到 FX 图中；TorchInductor 则将 FX 图编译为优化的后端特定代码（如 Triton GPU 内核）。
- **编译自动求导 (Compiled Autograd)**：
  - **用途**：通过跟踪和编译自动求导图来优化反向传播计算，消除 Python 开销。
  - **应用**：对于 Tiny-ONN 的稀疏训练，特别是当 `act_k` 动态变化时，编译自动求导可以进一步提升梯度计算的效率。
- **分布式训练 (`torch.distributed`)**：
  - **用途**：支持在多 GPU 或多节点环境下进行模型训练。
  - **应用**：考虑到 Tiny-ONN 专家模块的并行特性，分布式训练将是扩展其训练规模和加速训练过程的关键。`torch.distributed.tensor` 和 `torch.distributed.optim` 等模块将提供高效的分布式解决方案。

### 2.5 Hugging Face Transformers 库的应用

Hugging Face `transformers` 库是加载、管理和操作大型语言模型（如 Qwen3）的行业标准。它为 Tiny-ONN 的开发提供了便捷的接口和强大的功能。

- **模型加载与管理**：
  - `AutoModelForCausalLM.from_pretrained()`：用于加载 Qwen3 预训练模型，并支持 `load_in_4bit` 等量化参数。
  - `AutoTokenizer.from_pretrained()`：加载与模型对应的分词器，用于文本的编码和解码。
- **模型配置 (`PretrainedConfig`)**：
  - 提供对模型架构参数的访问，例如层数、隐藏维度、注意力头数等。这对于理解 Qwen3 的内部结构，并确定 Hook 注册点和参数提取策略至关重要。
- **数据处理**：
  - `transformers` 库与 Hugging Face `datasets` 库紧密集成，方便加载和预处理 SFT 数据集。
- **训练支持 (`Trainer`)**：
  - 虽然 Tiny-ONN 的持续预训练将采用定制的 SMK 策略，但 `Trainer` 类可以作为参考，或者在某些通用训练阶段提供便利。

通过结合 PyTorch 的底层灵活性和 `transformers` 库的高级抽象，我们将能够高效地实现 Tiny-ONN 的整个开发流程。

## 3. Qwen3 模型选择与预处理

本章将详细阐述 Tiny-ONN 项目中 Qwen3 模型的选择标准、量化加载策略以及 SFT (Supervised Fine-Tuning) 数据集的准备工作。这些是“知识炼金术”工作流的基石。

### 3.1 Qwen3 模型选择

Qwen3 系列模型因其在多语言、多模态和通用能力上的出色表现而被选为 Tiny-ONN 的知识源。我们将选择 **Qwen3-4B** 模型。

- **Qwen3-4B**：
  - **参数量**：40 亿参数。
  - **显存需求**：在 BF16 精度下，约 8GB 显存（仅模型权重）。加上 KV Cache、激活值和优化器状态，实际显存占用会更高，可能超过 8GB。
  - **适用场景**：如果目标硬件具备 16GB 或更高显存的 GPU，可以考虑直接加载 BF16 精度进行知识提取。否则，必须采用量化加载。

**选择原则**：
选择 Qwen3-4B 以获取最丰富的知识源，并通过量化加载有效管理显存。

### 3.2 Qwen3 量化加载策略

为了在有限的 GPU 显存上运行 Qwen3 模型并进行知识提取，量化加载是必不可少的步骤。我们将主要使用 Hugging Face `transformers` 库提供的集成量化功能。

#### 3.2.1 量化类型选择

- **INT4 量化**：
  - **优势**：显存占用最低，可以将 4B 模型压缩到约 2GB 显存。这为后续的激活和梯度数据收集提供了宝贵的显存空间。
  - **劣势**：精度损失相对较大，可能对知识提取的准确性产生轻微影响。
  - **实现**：通过 `transformers` 库的 `load_in_4bit=True` 参数，结合 `bitsandbytes` 库实现。
- **INT8 量化**：
  - **优势**：显存占用适中，精度损失小于 INT4。
  - **劣势**：显存占用仍高于 INT4。
  - **实现**：通过 `transformers` 库的 `load_in_8bit=True` 参数，结合 `bitsandbytes` 库实现。

**推荐策略**：
鉴于知识提取阶段对显存的严格要求，**优先采用 INT4 量化加载 Qwen3 模型**。在提取活跃参数时，再将其反量化回 BF16 精度。这种策略在保证显存效率的同时，确保了最终 Tiny-ONN 专家模块的精度。

#### 3.2.2 量化加载实现细节

使用 `transformers` 库加载 Qwen3 模型并进行 INT4 量化的示例代码骨架：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 1. 定义量化配置
# Qwen3 推荐使用 bfloat16 作为计算数据类型
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # 使用 NF4 量化类型
    bnb_4bit_compute_dtype=torch.bfloat16, # 计算数据类型
    bnb_4bit_use_double_quant=True, # 启用双重量化
)

# 2. 加载模型和分词器
model_name = "Qwen/Qwen1.5-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto", # 自动分配到可用设备
    torch_dtype=torch.bfloat16 # 模型加载时指定数据类型，尽管量化会改变实际存储精度
)

# 确认模型已量化
print(f"Model loaded in 4-bit: {model.is_loaded_in_4bit()}")
print(f"Model device: {model.device}")

# 3. 验证参数映射
# 即使量化，参数的逻辑结构和名称路径保持不变
# 我们可以遍历模型参数，验证其类型和形状
for name, param in model.named_parameters():
    if "weight" in name or "bias" in name:
        # 量化后的参数通常是 bitsandbytes.nn.Params4bit 类型
        # 但其逻辑形状和在模型中的位置与原始模型一致
        print(f"Parameter: {name}, Type: {type(param)}, Shape: {param.shape}")
        # 可以在这里验证参数是否为量化类型，例如 isinstance(param, bnb.nn.Params4bit)
```

**关键确认**：
量化加载后，模型参数的数值表示会变为低精度（如 INT4），但其在模型结构中的逻辑位置、名称路径和原始形状信息是**保持不变的**。这意味着我们可以通过 `model.named_parameters()` 遍历所有参数，并根据其名称或模块结构来识别和操作它们，这对于后续的参数提取和 Hook 注册至关重要。在提取时，这些量化参数可以被反量化回 BF16 精度。

### 3.3 SFT 数据集选择与准备

SFT (Supervised Fine-Tuning) 数据集是 Tiny-ONN 知识提取阶段的“探针”。通过在这些数据集上运行 Qwen3 模型，我们可以观察其在处理真实世界任务时的激活模式和知识贡献。数据集的质量和多样性将直接影响提取出的“干细胞”的纯度和通用性。

#### 3.3.1 数据集选择标准

选择 SFT 数据集时，应遵循以下原则：

- **多样性**：数据集应覆盖 LLM 的核心能力，包括但不限于：
  - **指令遵循**：如 Alpaca、ShareGPT 等，用于评估模型理解和执行指令的能力。
  - **多轮对话**：如 OpenAssistant Conversations Dataset (OASST1)，用于评估模型在对话上下文中的连贯性和响应能力。
  - **代码生成与理解**：如 CodeAlpaca、HumanEval 等，用于评估模型在编程任务上的知识。
  - **数学推理**：如 GSM8K、MATH 等，用于评估模型的逻辑和计算能力。
  - **常识问答**：如 CommonsenseQA、ARC 等，用于评估模型的常识知识。
  - **多语言能力**：如果 Qwen3 的多语言特性是关注点，应包含多语言 SFT 数据集。
- **高质量**：数据集应经过严格筛选，确保数据标注准确、无偏见、无重复，且指令清晰、响应合理。低质量数据可能导致提取的知识单元不纯或具有误导性。
- **适中规模**：数据集规模不宜过大，以免知识提取阶段的推理时间过长；也不宜过小，以确保能够充分激活 Qwen3 模型中的各种知识单元。建议选择数万到数十万条样本的数据集。
- **代表性**：数据集应能代表 Tiny-ONN 最终预期应用场景的典型任务和数据分布。

**推荐数据集示例**：

- `HuggingFaceH4/ultrachat_200k`：高质量多轮对话数据集。
- `tatsu-lab/alpaca`：指令遵循数据集。
- `WizardLM/WizardLM_evol_instruct_V2_196k`：进化指令数据集。
- `Open-Orca/OpenOrca`：高质量指令微调数据集。
- `bigcode/starcoderdata` (部分采样)：代码相关数据。

#### 3.3.2 数据集预处理

数据集预处理包括加载、分词、格式化和批处理。

1. **加载数据集**：
   使用 Hugging Face `datasets` 库加载选定的 SFT 数据集。

   ```python
   from datasets import load_dataset

   dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
   # dataset = dataset["train"].select(range(10000)) # 可选：采样部分数据进行测试
   ```

2. **分词 (Tokenization)**：
   使用与 Qwen3 模型对应的 `AutoTokenizer` 对数据集进行分词。需要将对话或指令-响应对格式化为模型可接受的输入序列。

   ```python
   # 假设 tokenizer 已经加载
   def preprocess_function(examples):
       # 根据数据集的具体格式，将文本转换为对话格式或指令-响应格式
       # 例如，对于 Ultrachat，可能需要将 turns 列表转换为单个字符串
       # 这里仅为示例，实际处理需根据数据集结构调整
       texts = []
       for conversation in examples["messages"]:
           formatted_text = tokenizer.apply_chat_template(
               conversation,
               tokenize=False,
               add_generation_prompt=False # 知识提取阶段不需要生成提示
           )
           texts.append(formatted_text)

       # 对文本进行分词，并截断或填充到最大长度
       tokenized_inputs = tokenizer(
           texts,
           max_length=tokenizer.model_max_length,
           padding="max_length",
           truncation=True,
           return_tensors="pt" # 返回 PyTorch 张量
       )
       return tokenized_inputs

   tokenized_dataset = dataset.map(
       preprocess_function,
       batched=True,
       remove_columns=dataset["train"].column_names # 移除原始列
   )
   ```

3. **格式化为模型输入**：
   确保分词后的数据集包含 `input_ids` 和 `attention_mask`。对于 SFT 任务，还需要 `labels`，通常是 `input_ids` 的副本，但在计算损失时会屏蔽掉输入部分。

   ```python
   def prepare_labels(examples):
       # 对于因果语言模型，labels 通常是 input_ids 的副本
       # 但在计算损失时，需要屏蔽掉输入部分的损失
       examples["labels"] = examples["input_ids"].clone()
       # 假设我们只计算响应部分的损失，需要找到响应的起始位置并屏蔽之前的部分
       # 这部分逻辑依赖于具体的对话模板和SFT数据格式
       # 简单示例：如果所有 token 都参与损失计算，则直接复制
       return examples

   tokenized_dataset = tokenized_dataset.map(prepare_labels, batched=True)
   ```

4. **数据加载器 (DataLoader)**：
   使用 `torch.utils.data.DataLoader` 创建数据加载器，以便在知识提取阶段进行批处理推理。

   ```python
   from torch.utils.data import DataLoader

   # 确保数据集是 PyTorch 格式
   tokenized_dataset.set_format("torch")

   train_dataloader = DataLoader(
       tokenized_dataset["train"],
       batch_size=4, # 根据显存调整批次大小
       shuffle=True
   )
   ```

**注意事项**：

- **对话模板**：Qwen3 模型有特定的对话模板（如 `tokenizer.apply_chat_template`）。在预处理 SFT 数据时，务必遵循该模板，以确保模型能够正确理解输入。
- **最大长度**：根据 Qwen3 模型的 `model_max_length` 设置分词器的 `max_length`，并决定截断或填充策略。
- **批次大小**：在量化加载模型后，批次大小仍需根据实际显存情况进行调整。较小的批次大小可以减少单次推理的显存峰值。

通过以上步骤，我们将为 Tiny-ONN 的知识提取阶段准备好高质量的 Qwen3 模型和 SFT 数据集。

## 4. 知识提取阶段：从巨石中外科手术式分离『干细胞』

本章详细阐述 Tiny-ONN 的核心——“知识提取”阶段。此阶段的目标是识别 Qwen3 模型中在 SFT 任务上最活跃、最核心的参数，并将其重组为 Tiny-ONN 的初始专家模块。

### 4.1 精细化数据收集与“预测完整性微分”（PIDiff）计算

**目标**：为Qwen3-4B模型中的**每一个参数**（或最小逻辑单元，如单个神经元权重）计算其在SFT数据集上的“预测完整性微分”（PIDiff）或“协同贡献度”。

**实现**：

- **记录原始数据**：在SFT数据集上运行Qwen3-4B模型，捕获**所有参数的完整激活模式**（例如，在不同输入下的激活值序列）和**梯度强度**。这将是海量数据，需要极度优化内存管理（例如，分批处理、实时聚合到CPU/磁盘、仅存储统计摘要）。
- **定义PIDiff/协同贡献度**：
  - 对于每个参数 `p`，我们定义一个**协同贡献分数 `S_p`**，作为其PIDiff的代理。`S_p` 将综合考虑：
    - **平均激活强度**：`Avg_Act(p)`
    - **平均梯度范数的倒数**：`1 / Avg_Grad_Norm(p)` (代表其稳定性/收敛性)
    - **激活频率**：`Freq_Act(p)` (代表其通用性或重要性)
  - **公式示例**：`S_p = w1 * Avg_Act(p) + w2 * (1 / Avg_Grad_Norm(p)) + w3 * Freq_Act(p)`。权重 `w1, w2, w3` 将通过实验调整，以最大化提取出的专家在后续训练中的性能。
  - **ΣPI工具包应用**：我们可以利用ΣPI工具包来计算这些底层指标，并将其整合到 `S_p` 中。

#### 4.1.1 PyTorch Hook 注册

在 Qwen3 模型的每个 Transformer 块的 FFN 层和注意力输出层上，注册 `forward_hooks` 以捕获激活值。同时，为每个参数张量注册 `register_hook` 以捕获梯度。

**Hook 注册实现**：

```python
activation_data = {} # 存储激活值
gradient_data = {}   # 存储梯度范数

def forward_hook_fn(module, input, output):
    act_val = output[0] if isinstance(output, tuple) else output
    activation_data[module] = act_val.detach().cpu().float()

def backward_hook_fn(grad):
    grad_val = grad[0] if isinstance(grad, tuple) else grad
    return grad_val

# 遍历 Qwen3 模型的所有模块
for name, module in model.named_modules():
    if "mlp.down" in name or "self_attn.o_proj" in name:
        module.register_forward_hook(forward_hook_fn)

# 为所有参数注册梯度 Hook
for name, param in model.named_parameters():
    if param.requires_grad:
        param.register_hook(lambda grad, p_name=name: gradient_data.setdefault(p_name, []).append(grad.norm().item()))
```

**注意**：

- **显存管理**：激活值和梯度数据量巨大，必须及时将它们 `detach().cpu()` 并存储，以避免 GPU 显存溢出。
- **Hook 签名**：`forward_hook_fn` 接收 `(module, input, output)`，`tensor.register_hook` 接收 `(grad)`。
- **梯度 Hook**：`tensor.register_hook` 的 `hook_fn` 必须返回梯度，否则反向传播会中断。这里我们只是在 Hook 内部记录范数，不修改梯度。
- **Qwen3 模块名称**：需要仔细检查 Qwen3 模型的实际模块名称（例如 `model.transformer.h.0.mlp.down_proj`）来精确注册 Hook。

### 4.2 SFT 集上的推理与数据收集

在注册好 Hook 后，我们将在选定的 SFT 数据集上对量化后的 Qwen3 模型进行前向推理，并收集激活值、梯度范数和激活频率。

**数据收集流程**：

1. **迭代 SFT 数据集**：使用之前准备好的 `DataLoader` 迭代 SFT 数据集。
2. **前向推理**：将 `input_ids` 和 `attention_mask` 输入到 Qwen3 模型。
3. **计算损失**：计算模型输出与 `labels` 之间的损失。
4. **反向传播**：执行 `loss.backward()` 以计算所有参数的梯度。
5. **数据捕获**：在每次迭代中，通过 Hook 捕获的激活值和梯度范数将被存储。同时，记录每个参数的激活频率（即在多少个样本中其激活值非零或梯度非零）。
6. **清零梯度**：在每次迭代后，必须清零模型参数的梯度 (`model.zero_grad()`)，以避免梯度累积。

```python
from tqdm import tqdm

# 存储每个参数的活跃度指标
param_metrics = {name: {"activations": [], "grad_norms": [], "activation_counts": 0} for name, _ in model.named_parameters()}

model.eval()
with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Collecting Activation Data")):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        for module, act_val in activation_data.items():
            # 假设激活值与特定参数相关联，需要建立映射
            # 这里简化处理，实际需要根据模块结构和参数名称进行更精细的关联
            pass

model.train()
for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Collecting Gradient Data")):
    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)
    labels = batch["labels"].to(model.device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    loss.backward()

    for name, param_grad_norms in gradient_data.items():
        if param_grad_norms:
            param_metrics[name]["grad_norms"].extend(param_grad_norms)
            if any(gn > 0 for gn in param_grad_norms):
                param_metrics[name]["activation_counts"] += 1
            param_grad_norms.clear()

    model.zero_grad()
```

**重要提示**：

- **两次前向传播**：为了同时收集激活值和梯度，可能需要进行两次前向传播。第一次在 `torch.no_grad()` 下收集激活值，第二次在启用梯度计算的情况下收集梯度。或者，更高效的方法是在一次前向传播中，通过 Hook 同时捕获激活和梯度（如果 Hook 设计得当）。
- **数据存储**：收集到的激活值和梯度数据量可能非常大，需要高效的存储策略（例如，只存储统计量如平均值、峰值、方差，而不是所有原始数据）。
- **激活频率定义**：激活频率可以定义为参数中至少一个参数的激活值或梯度非零的样本比例。

### 4.3 构建特征向量与聚类分析

**目标**：识别出在功能上高度相关、协同工作的参数集合，形成“高密度因果网络”。

#### 4.3.1 协同贡献分数 (S_p) 计算

为Qwen3-4B模型中的**每一个参数**计算其“预测完整性微分”（PIDiff）或“协同贡献度”分数 `S_p`。

- **核心理念**：“高激活 + 低梯度 = 高协同”是核心指导原则。
  - **高激活**：参数在处理特定输入时被频繁且强烈地使用。
  - **低梯度**：参数对当前任务的损失贡献已小，其内部参数相对稳定和优化。
  - **结合**：两者结合代表高预测完整性（PI），是高协同信息在功能层面的最佳代理。
- **公式示例**：`S_p = w1 * Avg_Act(p) + w2 * (1 / Avg_Grad_Norm(p)) + w3 * Freq_Act(p)`。权重 `w1, w2, w3` 将通过实验调整。
- **ΣPI工具包应用**：利用ΣPI工具包计算底层指标并整合到 `S_p` 中。

#### 4.3.2 特征向量构建与聚类分析

为Qwen3-4B中的**每一个参数**（或其所在的最小逻辑块，如FFN矩阵中的一行/列）构建一个高维特征向量，并进行聚类分析。

1. **特征向量构建**：
   向量将包含：
   - 其计算出的 `S_p` 分数。
   - 其在不同类型SFT样本（如代码、对话、推理）上的**行为模式摘要**（例如，平均激活、平均梯度范数、稀疏性等）。
   - 其**结构位置信息**（层索引、模块类型如`gate_proj`、矩阵内的行/列索引）。

2. **降维与聚类**：
   对这些参数的特征向量进行降维（如UMAP/t-SNE用于可视化，PCA用于特征选择），然后应用聚类算法（如K-means、DBSCAN、层次聚类）。
   - **目标**：聚类将揭示哪些参数在功能上是紧密关联的，它们共同构成了一个“高密度因果网络”或“概念细胞”。这些网络可能跨越不同的层和矩阵。
   - **动态性**：聚类的结果将**动态定义**我们的“片”，而不是预设的。

3. **可视化**：
   使用降维技术（如 t-SNE 或 UMAP）将聚类结果可视化，以直观地展示参数群之间的关联性和潜在的功能分区。这将帮助我们验证聚类结果的合理性。

   ```python
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.manifold import TSNE
   import umap.umap_ as umap

   # 假设 all_param_features 是一个 (num_params, feature_dim) 的 NumPy 数组
   # 其中 feature_dim 是每个参数的特征向量维度

   # 数据标准化
   scaler = StandardScaler()
   scaled_features = scaler.fit_transform(all_param_features)

   # 执行 K-Means 聚类
   num_clusters = 100 # 目标专家数量
   kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
   clusters = kmeans.fit_predict(scaled_features)

   # clusters 数组将包含每个参数所属的簇 ID (0 到 99)

   # 使用 t-SNE 进行降维
   tsne = TSNE(n_components=2, random_state=42)
   tsne_results = tsne.fit_transform(scaled_features)

   plt.figure(figsize=(10, 8))
   sns.scatterplot(
       x=tsne_results[:, 0], y=tsne_results[:, 1],
       hue=clusters, palette=sns.color_palette("hsv", num_clusters),
       legend="full", alpha=0.7
   )
   plt.title("t-SNE Visualization of Parameter Clusters")
   plt.xlabel("t-SNE Dimension 1")
   plt.ylabel("t-SNE Dimension 2")
   plt.show()

   # 使用 UMAP 进行降维
   reducer = umap.UMAP(n_components=2, random_state=42)
   umap_results = reducer.fit_transform(scaled_features)

   plt.figure(figsize=(10, 8))
   sns.scatterplot(
       x=umap_results[:, 0], y=umap_results[:, 1],
       hue=clusters, palette=sns.color_palette("hsv", num_clusters),
       legend="full", alpha=0.7
   )
   plt.title("UMAP Visualization of Parameter Clusters")
   plt.xlabel("UMAP Dimension 1")
   plt.ylabel("UMAP Dimension 2")
   plt.show()
   ```

**目标**：
识别出功能上高度相关、协同工作的参数集合，形成“高密度因果网络”，并最终组装成 100 个专家模块。

### 4.4 动态提取与专家组装

根据聚类和协同贡献分数结果，从 Qwen3 模型中智能地选择和组装参数，形成100个专家模块，总参数量控制在1B左右。

**提取流程**：

1. **选择策略**：
   根据聚类中参数的平均 `S_p` 分数、簇的紧凑性、以及簇所代表的语义功能（通过后续语义标注）来选择最有价值的簇。

2. **组装专家**：
   一个专家模块将由一个或多个选定的“高密度因果网络”（即参数簇）组成。这些参数可能来自Qwen3-4B模型的不同层、不同矩阵，甚至是非连续的内存区域。

3. **工程挑战与实现**：
   这将是最大的工程挑战。组装过程不再是简单的矩阵拼接，而是需要：
   - **稀疏矩阵表示**：将这些非连续的参数高效地存储为稀疏矩阵。
   - **自定义计算核（Kernel）**：可能需要编写自定义的PyTorch操作或CUDA Kernel，以高效地执行这些由非连续参数组成的专家模块的前向和反向传播。`torch.compile`的定制后端将在这里发挥关键作用。
   - **映射与索引**：维护一个精确的映射，记录每个专家模块由Qwen3-4B中哪些原始参数组成。

   ```python
   # 假设 selected_clusters 是一个列表，包含要提取的簇 ID
   # 假设 param_to_cluster_map 是一个从 param_name 到 cluster_id 的映射
   # 假设 original_qwen_model 是原始的 Qwen3 模型

   extracted_experts_weights = [] # 存储提取出的专家权重

   for cluster_id in selected_clusters:
       expert_weights_for_this_cluster = []
       # 遍历属于当前簇的所有参数
       for name, param_tensor in original_qwen_model.named_parameters():
           if param_to_cluster_map.get(name) == cluster_id:
               # 反量化为 BF16 精度
               if hasattr(param_tensor, 'to_bfloat16'):
                   dequantized_param = param_tensor.to_bfloat16().clone().detach()
               elif hasattr(param_tensor, 'dequantize'):
                   dequantized_param = param_tensor.dequantize().clone().detach()
               else:
                   dequantized_param = param_tensor.clone().detach().to(torch.bfloat16)
               expert_weights_for_this_cluster.append(dequantized_param)

       # 这里需要更复杂的逻辑来将这些非连续的参数组装成一个专家模块
       # 可能涉及创建稀疏张量或自定义层
       # 示例：简单地将所有参数列表作为专家权重
       extracted_experts_weights.append(expert_weights_for_this_cluster)

   # extracted_experts_weights 现在包含了 100 个专家模块的 BF16 精度权重
   # 这些权重将用于初始化 Tiny-ONN 的专家模块
   ```

**反量化注意事项**：

- **`bitsandbytes` 库**：`bitsandbytes` 库在加载量化模型时，会将参数包装成 `bnb.nn.Params4bit` 或 `bnb.nn.Params8bit` 类型。这些类型通常提供了 `to_bfloat16()` 或 `to_float()` 等方法来获取反量化后的张量。
- **精度选择**：选择 BF16 精度作为 Tiny-ONN 专家模块的初始精度，因为它在保持训练稳定性的同时，比 FP32 节省一半显存。
- **`clone().detach()`**：在提取参数时，务必使用 `clone().detach()` 来创建参数的独立副本，避免对原始 Qwen3 模型造成意外修改，并将其从计算图中分离。

至此，“知识提取”阶段完成。我们已经从 Qwen3 模型中成功分离并提纯了高质量的“高密度因果网络”，它们将作为 Tiny-ONN 的初始专家模块，承载着 Qwen3 的核心知识。

## 5. 持续预训练阶段：专家分化与路由适配

本章将详细描述 Tiny-ONN 的第二阶段——“持续预训练”。此阶段的目标是在提取出的专家组上进行 SMK (Sparse Mixture-of-Experts with Knowledge distillation) 持续预训练，促进其功能分化，并训练全新的路由系统以高效调度这些专家。

### 5.1 Tiny-ONN 架构初始化

在开始持续预训练之前，需要初始化 Tiny-ONN 的整体架构，包括加载提取出的专家集群和初始化全新的路由组。

#### 5.1.1 专家集群加载

加载第一阶段提取出的专家集群。这些专家将作为 Tiny-ONN 的核心计算单元，总参数量预计在0.6B左右。

- **实现**：
  - 定义一个 `TinyONNExpert` 类，继承自 `torch.nn.Module`。每个 `TinyONNExpert` 实例将封装一个或多个从 Qwen3 提取出的参数。
  - 在 `TinyONNExpert` 的 `__init__` 方法中，加载对应的 BF16 精度权重。
  - 将 100 个 `TinyONNExpert` 实例组织成一个 `torch.nn.ModuleList`。

```python
import torch.nn as nn

class TinyONNExpert(nn.Module):
    def __init__(self, expert_weights_list):
        super().__init__()
        # expert_weights_list 是一个列表，包含该专家模块的所有参数张量
        # 例如，可以是一个 FFN 层的两个线性层权重
        self.linear1_weight = nn.Parameter(expert_weights_list[0])
        self.linear2_weight = nn.Parameter(expert_weights_list[1])
        # 根据实际提取的参数结构进行调整
        # 也可以直接将整个 FFN 模块作为专家

    def forward(self, x):
        # 示例：一个简单的FFN结构
        x = torch.matmul(x, self.linear1_weight.T) # 假设是转置的权重
        x = nn.functional.relu(x)
        x = torch.matmul(x, self.linear2_weight.T)
        return x

# 假设 extracted_experts_weights 是第一阶段提取出的 100 个专家权重列表
# 每个元素是一个列表，包含该专家模块的多个参数张量
tiny_onn_experts = nn.ModuleList([
    TinyONNExpert(weights) for weights in extracted_experts_weights
])
```

**注意**：`TinyONNExpert` 的具体实现将取决于从 Qwen3 中提取的参数的粒度。如果提取的是完整的 FFN 模块，那么 `TinyONNExpert` 可以直接封装该模块。如果提取的是更细粒度的权重，则需要重新构建对应的层。

#### 5.1.2 路由组初始化

初始化一个**全新且独立的路由组**，包括 **1 个主路由**和 **10 个子路由**（总参数量约 0.5B）。路由组的参数应从头开始训练。

- **主路由 (Main Router)**：负责根据输入初步判断，将请求分配给一个或多个子路由。
- **子路由 (Sub-Routers)**：每个子路由负责在主路由的指导下，从 100 个专家中选择 `top-k` 个专家进行激活。

**路由组设计考量**：

- **参数量**：总参数量约 0.5B，确保路由系统足够复杂以进行精确调度，同时又不会过于庞大。
- **结构**：可以是简单的线性层、小型 MLP 或更复杂的 Transformer 编码器。
- **训练**：路由组的参数将从随机初始化开始，并在持续预训练阶段与专家组协同训练。

```python
class MainRouter(nn.Module):
    def __init__(self, input_dim, num_sub_routers):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_sub_routers) # 输出到子路由的数量

    def forward(self, x):
        return self.linear(x) # 输出每个子路由的 logits

class SubRouter(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_experts) # 输出到专家数量

    def forward(self, x):
        return self.linear(x) # 输出每个专家的 logits

# 假设输入维度与 Qwen3 的隐藏维度一致
hidden_dim = 4096 # 示例，需根据Qwen3实际维度调整
num_sub_routers = 10
num_experts = 100

main_router = MainRouter(hidden_dim, num_sub_routers)
sub_routers = nn.ModuleList([
    SubRouter(hidden_dim, num_experts) for _ in range(num_sub_routers)
])

# 将专家组和路由组集成到 Tiny-ONN 的整体架构中
class TinyONN(nn.Module):
    def __init__(self, experts, main_router, sub_routers):
        super().__init__()
        self.experts = experts
        self.main_router = main_router
        self.sub_routers = sub_routers
        # 其他组件，如 LayerNorm, Embedding 等，可以从Qwen3复制或重新初始化
        # 这里仅展示核心的专家和路由部分

    def forward(self, x):
        # 路由逻辑：
        # 1. 输入 x 经过主路由，选择一个或多个子路由
        # 2. 选定的子路由根据 x 选择 top-k 专家
        # 3. 输入 x 经过选定的专家进行计算
        # 4. 专家输出进行加权求和

        # 这是一个简化的路由逻辑示例
        main_router_logits = self.main_router(x)
        # 选择激活的子路由 (例如，使用 top-1 或 top-k)
        activated_sub_router_idx = torch.argmax(main_router_logits, dim=-1)

        expert_logits = self.sub_routers[activated_sub_router_idx](x)
        # 选择 top-k 专家
        top_k_values, top_k_indices = torch.topk(expert_logits, k=10, dim=-1) # 示例 k=10

        # 根据 top_k_indices 激活专家并计算加权和
        # ... (复杂的专家调度和加权逻辑)

        return x # 示例返回
```

### 5.2 SMK 持续预训练流程

在初始化 Tiny-ONN 架构后，将在与第一阶段相同的 SFT 数据集上启动 Tiny-ONN 的**两阶段 SMK 持续预训练**。SMK (Sparse Mixture-of-Experts with Knowledge distillation) 是一种旨在促进专家分化和路由适配的训练策略。

#### 5.2.1 阶段 1 (强制分化)

- **目标**：让专家组在通用 SFT 任务上开始**强制分化**，形成初步的专业领域。
- **关键参数**：
  - `act_k=100`：所有专家都计算梯度。这意味着在反向传播时，所有 100 个专家都会接收梯度更新。这有助于在训练初期快速激活和调整所有专家。
  - `top_k=10`：路由尝试选择 10 个专家。尽管所有专家都计算梯度，但路由器的目标是学习选择最相关的 10 个专家。
  - `min_k=10`：SMK 策略强制只更新 `surprise` 最低的 10 个专家。这里的 `surprise` 可以理解为专家对当前输入的“意外程度”或“不确定性”。选择 `surprise` 最低的专家进行更新，意味着我们鼓励专家在它们已经相对“熟悉”的领域进行更精细的优化，从而促进专业化。

**训练细节**：

- **损失函数**：SFT 任务的交叉熵损失。
- **知识蒸馏 (Knowledge Distillation)**：可以引入知识蒸馏损失，以 Qwen3 原始模型作为教师模型，Tiny-ONN 作为学生模型。蒸馏损失可以帮助 Tiny-ONN 保持 Qwen3 的通用能力。
- **路由损失**：除了 SFT 损失，还可以引入路由损失，例如负载均衡损失 (Load Balancing Loss)，以确保专家被均匀地激活，避免某些专家被过度使用而另一些专家被闲置。
- **优化器**：AdamW 或其他适用于 LLM 训练的优化器。
- **学习率调度**：Cosine 学习率衰减或其他调度策略。

**强制分化机制**：
通过 `act_k=100` 和 `min_k=10` 的结合，我们强制所有专家在训练初期都参与到梯度计算中，但只对那些“表现良好”（surprise 低）的专家进行更集中的更新。这鼓励专家在它们擅长的领域变得更加专业。

#### 5.2.2 阶段 2 (自适应涌现)

- **目标**：让路由系统学会**高效调度**这些正在分化的专家，并让专家功能进一步**精细化**。
- **关键参数**：
  - `top_k=dynamic`：路由根据 `surprise` 动态激活更多专家。这意味着路由器不再固定选择 10 个专家，而是根据输入和专家自身的“surprise”程度，动态决定激活的专家数量。这允许模型在需要时利用更多专家，在不需要时保持稀疏。
  - `act_k=top_k`：只计算被激活专家的梯度。这与传统的 SMoE 训练一致，只有被路由器选中的专家才参与计算和梯度更新，从而实现计算稀疏性。
  - `min_k = f(PI, act_k)`：学习率和更新数量根据预测完整性 (PI, Prediction Integrity) 动态调整。PI 可以衡量模型对当前输入的预测置信度或完整性。当 PI 较低时，可能需要激活更多专家或以更高的学习率更新专家，以提升模型能力。

**训练细节**：

- **预测完整性 (PI)**：PI 的具体定义需要实验探索。它可以是模型输出的熵、与教师模型输出的一致性、或者基于某些内部激活模式的指标。
- **动态学习率/更新数量**：根据 PI 动态调整学习率和每个专家的更新步数。例如，当 PI 较低时，增加学习率或允许更多专家参与更新。
- **路由系统优化**：此阶段路由系统将更加专注于学习如何根据输入内容和任务需求，精确地将请求路由到最合适的专家组合。

**自适应涌现机制**：
通过动态 `top_k` 和基于 PI 的 `min_k` 调整，Tiny-ONN 能够自适应地调整其稀疏性，并在需要时利用更多专家来处理复杂或不确定的输入。这使得模型在保持效率的同时，能够涌现出更强大的能力。

### 5.3 路由系统适配

在 SMK 训练过程中，路由组将与专家组协同进化，学会根据输入内容和任务需求，精确地将请求路由到最合适的专家组合。

- **路由损失设计**：
  - **负载均衡损失**：鼓励路由器均匀地使用所有专家，避免某些专家被过度使用而另一些专家被闲置。
  - **专家特异性损失**：鼓励专家在功能上进行分化，例如，通过惩罚专家在不相关输入上的激活。
  - **知识蒸馏损失**：路由器的输出（例如，专家选择的概率分布）可以与教师模型（Qwen3）的某些内部表示进行对齐，以引导路由器学习更有效的调度策略。
- **路由器的输入**：
  路由器的输入可以是原始的 token embedding，也可以是 Transformer 块的输出，或者是经过特定聚合的上下文表示。选择合适的输入对于路由器学习有效的调度策略至关重要。
- **路由器的输出**：
  路由器的输出是每个专家的 logits。这些 logits 经过 softmax 或 top-k 门控机制，决定哪些专家被激活。
- **主路由与子路由的协同**：
  主路由负责粗粒度的专家组选择，子路由负责细粒度的专家选择。这种分层路由机制可以提高调度的效率和准确性。主路由可以学习识别输入的大致类别（例如，代码、数学、对话），然后将任务传递给擅长该类别的子路由，子路由再从其管辖的专家池中选择最合适的专家。

**PILF-2 路由系统在 LLM 训练上的差异**：
正如原始笔记所指出的，PILF-2 在图片分类任务上遇到的路由问题，在 LLM 训练中可能得到缓解。

- **文本数据连续性**：文本数据在语义和概念上通常比图像数据更连续，这使得路由器更容易学习到稳定的映射关系。
- **概念复用性**：LLM 中的概念细胞（专家）在不同文本中会反复被激活，这为路由器提供了更丰富的学习信号，有助于强化正确的路由路径。
- **自监督信号**：LLM 的自监督训练（预测下一个词）提供了极其丰富和连续的监督信号，这有助于路由器更稳定地学习。
- **预分化专家**：Tiny-ONN 的专家模块是从 Qwen3 中提取的“干细胞”，它们在开始训练时就已具备初步的语言理解能力和专业分化倾向。这意味着路由器不需要从零开始学习如何分化专家，它只需要学习如何高效地“调度”这些已经预分化的专家，大大降低了路由器的学习难度和出错概率。

通过精心设计的 SMK 策略和路由系统，Tiny-ONN 将在持续预训练阶段实现专家功能的精细分化和高效的知识调度。

## 6. 评估、可解释性与发布：展示智能的未来

本章将规划 Tiny-ONN 的最终阶段——性能评估、可解释性分析方法以及模型发布策略。这是验证“知识炼金术”工作流有效性，并向社区展示 Tiny-ONN 价值的关键步骤。

### 6.1 性能评估

全面评估 Tiny-ONN 的性能，并与原始 Qwen3-4B 模型进行直接对比，以验证其在效率和能力上的优势。

#### 6.1.1 基准测试集选择

在多个基准测试集上评估 Tiny-ONN，确保其在不同任务和能力维度上的表现。

- **通用语言理解**：
  - **MMLU (Massive Multitask Language Understanding)**：评估模型在 57 个学科上的多任务理解能力。
  - **C-Eval**：中文综合能力评估基准。
  - **GSM8K (Grade School Math 8K)**：小学数学问题，评估数学推理能力。
  - **HumanEval**：代码生成能力评估。
- **推理能力**：
  - **ARC (AI2 Reasoning Challenge)**：常识推理。
  - **HellaSwag**：常识推理，区分合理与不合理。
- **长上下文能力**：
  - 使用长文本摘要、长文档问答等任务，评估模型处理长序列的能力。
- **多语言能力**：
  - 如果 Qwen3 的多语言特性是关注点，应在 XNLI、XSum 等多语言基准上进行评估。
- **效率指标**：
  - **吞吐量 (Tokens/sec)**：在不同批次大小下的推理速度。
  - **显存占用 (GB)**：模型加载和推理时的显存峰值。
  - **FLOPs (Floating Point Operations)**：每 token 的计算量。

#### 6.1.2 评估指标与对比

- **核心指标**：
  - **准确率/F1 分数**：针对分类、问答等任务。
  - **BLEU/ROUGE**：针对生成任务（如摘要、翻译）。
  - **Pass@k**：针对代码生成任务。
  - **困惑度 (Perplexity)**：在未见过的文本数据上的语言建模能力。
- **对比对象**：
  - **Qwen3-4B (BF16 精度)**：作为性能基线，直接对比 Tiny-ONN 在相同任务上的表现。
  - **Qwen3-4B (INT4 量化)**：对比量化后的 Qwen3，以评估 Tiny-ONN 在保持效率的同时，是否能超越量化模型的性能。
- **性能预期**：
  验证“性能损耗小于 10% 甚至比肩”的预期。这意味着 Tiny-ONN 在激活参数量远小于 Qwen3 的情况下，仍能保持相似甚至更好的性能。
- **稀疏性与性能权衡**：
  评估 Tiny-ONN 在不同激活参数量（例如，从 0.6B 到 1.1B，通过调整 `top_k` 或 `min_k`）下的性能表现。这将展示 Tiny-ONN 在效率和能力之间的灵活权衡。

### 6.2 可解释性分区与专家标注

Tiny-ONN 的一个核心优势是其潜在的可解释性。我们将通过可视化和语义标注来揭示模型内部的知识分区和决策机制。

#### 6.2.1 可视化技术

- **激活散点图与热力图**：
  - **用途**：展示专家在不同任务或概念上的激活模式。
  - **实现**：
    - 在推理过程中，记录每个专家在不同输入样本上的激活强度（例如，FFN 层的输出范数）。
    - 将这些激活强度数据进行降维（如 PCA、t-SNE、UMAP），并在二维或三维空间中绘制散点图，每个点代表一个专家，颜色或大小表示其激活强度。
    - 生成热力图，行代表专家，列代表不同的任务类别或语义概念，单元格颜色表示专家在该任务/概念上的平均激活强度。
  - **示例**：类似 `expert_dashboard` 的可视化，直观展示哪些专家在处理特定类型输入时最为活跃。
- **K-聚类可视化**：
  - **用途**：对最终分化后的专家权重或激活模式进行 K-聚类，并在低维空间中可视化，以确认其功能分区的清晰性。
  - **实现**：与知识提取阶段的聚类类似，但应用于持续预训练后的专家。这将验证专家是否形成了清晰的功能簇。

#### 6.2.2 语义标注与概念细胞识别

- **目标**：结合 SFT 数据集中的 `token/sample` 与专家激活的统计学关联，尝试为每个专家**标注其专业分工**（即“概念细胞”的语义标签）。
- **实现方法**：
  1. **数据关联**：在 SFT 数据集上进行推理时，记录每个输入样本（或其中的关键 token/短语）激活了哪些专家，以及这些专家的激活强度。
  2. **统计分析**：对每个专家，统计与其高激活相关的输入样本、关键词、语义概念或任务类型。例如，如果某个专家在处理大量包含“金融”、“股票”、“投资”等词汇的文本时频繁高激活，则可以初步推断其专业领域。
  3. **聚类与命名**：对与每个专家强关联的关键词或概念进行聚类，并尝试为其赋予一个简洁的语义标签。例如，“L0E1：Python 代码生成”、“L1E5：金融市场分析”、“L2E3：情感识别”等。
  4. **人工验证**：对自动生成的语义标签进行人工审查和验证，确保其准确性和合理性。
- **挑战**：语义标注是一个复杂且具有挑战性的任务，可能需要结合主题模型、关键词提取、注意力机制分析等多种技术。

### 6.3 技术报告撰写与模型发布

在完成评估和可解释性分析后，将撰写详细的技术报告，并准备 Tiny-ONN 模型和相关工具的发布。

#### 6.3.1 技术报告撰写

技术报告将详细记录整个“知识炼金术”工作流，包括：

- **引言**：背景、动机、Tiny-ONN 核心理念。
- **相关工作**：简要回顾稀疏专家模型、知识蒸馏、模型量化等相关领域。
- **Tiny-ONN 架构**：详细描述专家模块、路由系统（主路由、子路由）的设计。
- **知识提取方法**：
  - Qwen3 模型选择与量化加载。
  - SFT 数据集选择与预处理。
  - 精细化数据收集与PIDiff计算。
  - 构建特征向量与聚类分析。
  - 动态提取与专家组装。
- **持续预训练方法**：
  - SMK 策略（阶段 1：强制分化，阶段 2：自适应涌现）的详细描述。
  - 路由损失、知识蒸馏损失等训练目标。
  - 优化器、学习率调度等训练超参数。
  - PILF-2 经验在 LLM 上的适用性分析。
- **实验结果**：
  - 性能评估：在各项基准测试上的详细结果，与 Qwen3 的对比。
  - 效率分析：吞吐量、显存占用、FLOPs 等指标。
  - 稀疏性与性能权衡曲线。
- **可解释性分析**：
  - 专家激活可视化（散点图、热力图）。
  - 专家功能分区与语义标注结果。
- **讨论与展望**：
  - Tiny-ONN 在效率、可解释性、横向扩展能力上的突破。
  - 对未来 AI 范式和《Web://Reflect》世界观的深远影响。
  - 未来研究方向，例如 `min_k=1` 的永续训练适配问题。

报告将重点突出 Tiny-ONN 的创新性、技术细节和实验验证，力求严谨和客观。

#### 6.3.2 模型与工具发布

- **模型权重发布**：
  - 以 **PyTorch `.pth` 格式**发布 Tiny-ONN 的模型权重（包括路由和专家）。
  - 可以考虑发布到 Hugging Face Hub，以便社区轻松访问和使用。
- **推理代码与示例**：
  - 提供清晰、易于使用的推理代码和示例脚本，展示如何加载 Tiny-ONN 模型并进行推理。
  - 包括不同 `top_k` 设置下的推理示例，以展示稀疏性与性能的权衡。
- **知识提取工具**：
  - 如果可能，将知识提取阶段的关键代码封装为可复用的工具或脚本，以便其他研究者可以尝试从其他大型模型中提取专家。
- **社区挑战**：
  - **将 `min_k=1` 的永续训练适配问题作为社区挑战**，提供相关工具和文档，鼓励社区进行更深层次的优化和应用。这个问题涉及到如何在极端稀疏（每次只更新一个专家）的情况下，保持模型的持续学习能力和稳定性。

通过全面的技术报告和便捷的模型与工具发布，我们将促进 Tiny-ONN 理念的传播和应用，为稀疏模型和可解释 AI 的发展贡献力量。

## 7. 潜在挑战与解决方案

在 Tiny-ONN 的开发过程中，可能会遇到一系列技术挑战。本章将预见这些挑战并提出相应的解决方案。

### 7.1 挑战：显存管理与量化精度权衡

**挑战描述**：
在知识提取阶段，即使 Qwen3 模型经过 INT4 量化加载，但同时捕获大量激活值和梯度数据仍然可能导致 GPU 显存溢出。此外，INT4 量化带来的精度损失可能影响知识提取的准确性。

**解决方案**：

- **分批数据收集**：将 SFT 数据集分成更小的批次进行推理，每次只处理少量样本，并及时将激活值和梯度数据从 GPU 转移到 CPU 内存或磁盘。
- **数据采样**：如果 SFT 数据集过大，可以对其进行采样，只在部分代表性样本上进行知识提取。
- **统计量存储**：不存储所有原始激活值和梯度张量，而是只存储其统计量（如平均值、范数、最大值、最小值），这可以大幅减少存储需求。
- **梯度累积**：对于梯度计算，可以采用梯度累积策略，在多个小批次上累积梯度，然后进行一次反向传播，以减少显存峰值。
- **Hook 优化**：精细化 Hook 的设计，只捕获最关键的激活值和梯度，避免捕获冗余数据。例如，只捕获 FFN 输出的激活，而不是所有中间层的激活。
- **量化精度验证**：在知识提取前，对 INT4 量化后的 Qwen3 模型进行小规模的性能测试，确保其在 SFT 任务上的表现没有严重退化，从而验证量化精度对知识提取的影响在可接受范围内。

### 7.2 挑战：动态提取与专家组装的工程复杂性

**挑战描述**：
动态提取方案在数据收集、分析和工程实现上带来了巨大的复杂性。如何精确计算每个参数的PIDiff，构建有效的特征向量，以及将非连续的参数高效组装成专家模块，是核心挑战。

**解决方案**：

- **PIDiff计算优化**：
  - 确保PIDiff公式能够准确反映参数的协同贡献度，并通过小规模实验验证其有效性。
  - 优化PIDiff的计算流程，例如，利用PyTorch的`autograd.grad`进行更精细的梯度分析，或开发自定义的C++/CUDA扩展以加速计算。
- **特征工程与聚类算法**：
  - 深入探索不同特征向量的构建方法，结合参数的结构信息、激活模式和梯度行为，以更好地捕捉参数间的内在关联。
  - 尝试多种聚类算法及其参数调优，选择最能揭示“高密度因果网络”的方法。
  - 充分利用可视化工具（如t-SNE, UMAP）辅助分析聚类结果，指导特征工程和算法选择。
- **专家组装与自定义Kernel**：
  - 针对非连续参数的存储和计算，研究并实现高效的稀疏矩阵表示方法。
  - 探索使用PyTorch的`torch.compile`结合自定义后端（如Triton）来编写高性能的计算核（Kernel），以优化专家模块的前向和反向传播。这将是实现极致效率的关键。
  - 维护精确的参数映射和索引系统，确保在组装和运行时能够正确地访问和操作这些非连续的参数。

### 7.3 挑战：活跃度与关联度分析的准确性

**挑战描述**：
活跃度分数的计算公式和聚类特征向量的构建对知识提取的质量至关重要。不准确的评估可能导致提取出低质量或不相关的专家。

**解决方案**：

- **多指标融合**：综合考虑激活值、梯度范数、激活频率等多个指标，并对它们进行适当的归一化和加权，以构建鲁棒的活跃度分数。权重可以通过小规模实验或专家经验进行调整。
- **特征工程**：探索不同的特征向量构建方法，例如，除了原始激活/梯度值，还可以考虑其在不同任务子集上的差异性、稀疏性等。
- **聚类算法选择与参数调优**：尝试不同的聚类算法（如 K-Means、DBSCAN、层次聚类）和参数设置，选择最能反映参数群内在关联性的方法。
- **可视化辅助**：充分利用 t-SNE、UMAP 等可视化工具，直观地检查聚类结果的质量，并根据可视化反馈调整特征和聚类参数。
- **迭代优化**：知识提取是一个迭代优化的过程。在完成第一轮提取和持续预训练后，根据 Tiny-ONN 的性能和可解释性反馈，反过来调整知识提取阶段的策略。

### 7.4 挑战：SMK 持续预训练的收敛性与稳定性

**挑战描述**：
SMK 策略，特别是动态 `top_k` 和基于 PI 的 `min_k` 调整，可能引入训练的不稳定性。路由系统和专家模块的协同进化也可能面临收敛困难。

**解决方案**：

- **渐进式训练**：
  - **阶段 1 (强制分化)**：在训练初期，保持 `act_k=100` 和 `min_k=10` 的强制分化策略，确保所有专家都能得到充分的初始化和初步调整。
  - **阶段 2 (自适应涌现)**：在模型达到一定收敛后，再逐步引入动态 `top_k` 和基于 PI 的 `min_k` 调整，允许模型自适应地学习稀疏性。
- **路由损失正则化**：
  - 引入适当的负载均衡损失和专家特异性损失，以引导路由器学习更有效的调度策略，并促进专家分化。
  - 对路由器的权重进行 L1/L2 正则化，防止过拟合。
- **知识蒸馏**：
  - 持续使用 Qwen3 原始模型作为教师模型进行知识蒸馏，可以为 Tiny-ONN 提供稳定的学习信号，帮助其保持通用能力并稳定训练。
- **预测完整性 (PI) 的鲁棒定义**：
  - PI 的定义应足够鲁棒，能够准确反映模型对当前输入的置信度。可以尝试多种 PI 定义，并通过实验选择最佳方案。
- **超参数调优**：
  - 对学习率、批次大小、SMK 策略中的 `w1, w2, w3, w4` 权重、`top_k` 范围、PI 阈值等超参数进行系统性调优。
- **监控与诊断**：
  - 在训练过程中，密切监控损失曲线、专家激活频率、路由器的输出分布等指标，及时发现并诊断训练不稳定问题。

### 7.5 挑战：可解释性标注的准确性与自动化

**挑战描述**：
为每个专家进行语义标注是一个复杂且主观的任务，尤其是在专家功能重叠或模糊时。完全自动化标注可能面临准确性挑战。

**解决方案**：

- **多维度关联分析**：
  - 除了 `token/sample` 关联，还可以结合注意力权重、概念激活图等多种信息来推断专家功能。
  - 利用外部知识库（如 WordNet、ConceptNet）来辅助概念识别和扩展。
- **交互式标注工具**：
  - 开发一个交互式工具，允许研究人员对自动生成的语义标签进行审查、编辑和确认。这可以结合人工智慧和机器学习的优势。
- **专家功能验证**：
  - 设计小规模实验，针对特定语义标签的专家进行单独测试，验证其是否确实擅长该领域。例如，如果一个专家被标注为“Python 代码生成”，则可以输入 Python 代码相关的提示，观察其激活情况和输出质量。
- **模糊性处理**：
  - 对于功能重叠或难以清晰标注的专家，可以允许其具有多个标签，或者将其归类为“通用”专家。

### 7.6 挑战：`min_k=1` 永续训练适配问题

**挑战描述**：
将 `min_k=1`（每次只更新一个专家）的永续训练适配作为社区挑战，意味着这是一个尚未完全解决的难题。在极端稀疏的更新模式下，如何确保模型能够持续学习新知识而不遗忘旧知识，同时保持训练稳定性，是一个巨大的挑战。

**解决方案（初步设想，待社区探索）**：

- **记忆回放机制**：引入记忆回放缓冲区，存储少量旧任务的样本，在训练新任务时，定期回放旧样本，以防止灾难性遗忘。
- **弹性权重巩固 (EWC) 或其他正则化**：对重要参数施加正则化，使其在更新时不易偏离其在旧任务上的最优值。
- **动态专家分配**：路由器需要更智能地分配任务，确保每个专家都有机会被激活和更新，同时避免过度专业化。
- **知识蒸馏**：持续蒸馏自一个更大的、不断进化的教师模型，可以为 `min_k=1` 的训练提供稳定的知识锚点。
- **元学习 (Meta-Learning)**：训练一个元学习器，使其能够学习如何高效地在 `min_k=1` 的设置下进行学习和适应。

这些挑战是 Tiny-ONN 项目复杂性的体现，但也是其创新性和潜在影响力的来源。通过系统性的方法和持续的实验，我们有信心克服这些挑战，并成功构建出高效、可解释的 Tiny-ONN 模型。

## 8. 结论与展望

### 8.1 结论

Tiny-ONN 项目通过其创新的“知识炼金术”工作流，为构建高效、可解释的稀疏专家模型提供了一条新颖的路径。我们从大型预训练模型（Qwen3）中外科手术式地提取核心知识单元，将其重组为高质量的专家模块，并通过两阶段 SMK 持续预训练和自适应路由系统，实现了专家分化和高效知识调度。

本开发计划详细阐述了从模型选择、量化加载、SFT 数据集准备，到知识提取（精细化数据收集与PIDiff计算、构建特征向量与聚类分析、动态提取与专家组装），再到持续预训练（架构初始化、SMK 策略、路由系统适配），以及最终的性能评估、可解释性分析和模型发布的全过程。我们预见了潜在的挑战，并提出了相应的解决方案。

Tiny-ONN 的成功将验证以下关键假设：

- 大型预训练模型中的知识是可以被有效分解、提取和重组的。
- 通过动态选择性抽样获得的专家模块具有更高的初始质量和任务相关性。
- 稀疏专家模型可以在显著降低计算成本的同时，保持甚至比肩原始稠密模型的性能。
- 通过对专家激活模式的分析和语义标注，可以提升模型的内部可解释性。

Tiny-ONN 不仅是一个技术突破，更是对未来 AI 范式的一次探索。它预示着一个由更小、更专业、更透明的“概念细胞”组成的“数字生命”的时代，这些“细胞”可以根据需求动态组合，形成适应性更强、效率更高的智能系统。

### 8.2 展望

Tiny-ONN 的成功将为未来的研究和应用开启广阔的前景：

- **更高效的 AI 部署**：Tiny-ONN 将使 LLM 在边缘设备、移动平台和资源受限的云环境中部署成为可能，极大地拓宽了 AI 的应用边界。
- **可解释 AI 的新范式**：通过对专家功能的语义标注，Tiny-ONN 将为理解 LLM 内部工作机制提供新的视角，促进可信赖 AI 的发展。
- **横向扩展能力**：Tiny-ONN 的模块化设计使其易于横向扩展。未来可以根据需求动态添加或替换专家模块，以适应新的任务或领域知识，而无需重新训练整个大型模型。
- **持续学习与适应性**：`min_k=1` 的永续训练适配问题是未来研究的重要方向。如果能够解决，Tiny-ONN 将能够实现真正的持续学习，不断吸收新知识并适应环境变化，而无需进行昂贵的完整模型再训练。
- **多模态与多任务融合**：Tiny-ONN 的架构可以扩展到多模态领域，例如，通过提取视觉模型中的专家，并与语言专家结合，构建更高效的多模态稀疏模型。
- **《Web://Reflect》世界观的实现**：Tiny-ONN 的“概念细胞”和动态调度机制，与《Web://Reflect》中描述的智能体通过组合和反射实现复杂行为的理念高度契合。Tiny-ONN 将是构建这种自适应、自组织智能生态系统的关键基石。

Tiny-ONN 项目代表了对 AI 效率、可解释性和适应性极限的挑战。我们相信，通过本计划中概述的严谨方法和创新策略，我们将能够成功地将“知识炼金术”变为现实，并为智能的未来奠定基础。
