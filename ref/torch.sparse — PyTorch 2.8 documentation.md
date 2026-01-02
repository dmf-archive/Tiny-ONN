---
创建时间：2025-09-08T02:25:05 (UTC +00:00)
标签：[]
来源：https://docs.pytorch.org/docs/stable/sparse.html
作者：PyTorch Contributors
---

# torch.sparse — PyTorch 2.8 文档

> ## 摘录
>
> PyTorch 文档。探索 PyTorch，一个开源机器学习库，可加速从研究原型到生产部署的路径。发现教程、API 参考和指南，以帮助您高效地构建和部署深度学习模型。

---

创建于：2017 年 4 月 26 日 | 最后更新于：2025 年 6 月 18 日

警告

PyTorch 稀疏张量的 API 处于测试阶段，可能会在不久的将来发生变化。我们非常欢迎通过 GitHub issue 提出功能请求、错误报告和一般性建议。

## 为什么以及何时使用稀疏性

默认情况下，PyTorch 在物理内存中连续存储 [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor") 元素。这导致了各种需要快速访问元素的数组处理算法的高效实现。

现在，一些用户可能会决定通过张量来表示数据，例如图邻接矩阵、剪枝权重或点云，而这些张量的*元素大部分为零值*。我们认识到这些是重要的应用，并旨在通过稀疏存储格式为这些用例提供性能优化。

多年来，已经开发了各种稀疏存储格式，如 COO、CSR/CSC、半结构化、LIL 等。虽然它们在确切布局上有所不同，但它们都通过对零值元素的有效表示来压缩数据。我们将未压缩的值称为*指定的*元素，与*未指定的*、压缩的元素相对。

通过压缩重复的零，稀疏存储格式旨在节省各种 CPU 和 GPU 上的内存和计算资源。特别是对于高度稀疏性或高度结构化的稀疏性，这可能对性能产生重大影响。因此，稀疏存储格式可以被视为一种性能优化。

像许多其他性能优化一样，稀疏存储格式并非总是占优。当为您的用例尝试稀疏格式时，您可能会发现执行时间增加而不是减少。

如果您在分析上预期性能会大幅提高但测量到性能下降，请随时在 GitHub issue 中提出。这有助于我们优先实现高效内核和更广泛的性能优化。

我们使尝试不同的稀疏布局以及在它们之间进行转换变得容易，而不会对您的特定应用程序的最佳选择持固执己见。

## 功能概述

我们希望通过为每种布局提供转换例程，能够轻松地从给定的密集张量构建稀疏张量。

在下一个示例中，我们将一个具有默认密集（跨步）布局的二维张量转换为一个由 COO 内存布局支持的二维张量。在这种情况下，仅存储非零元素的*值*和*索引*。

```python
a = torch.tensor([[0, 2.], [3, 0]])
a.to_sparse()
tensor(indices=tensor([[0, 1],
                       [1, 0]]),
       values=tensor([2., 3.]),
       size=(2, 2), nnz=2, layout=torch.sparse_coo)
```

PyTorch 目前支持 [COO](https://docs.pytorch.org/docs/stable/sparse.html#sparse-coo-docs)、[CSR](https://docs.pytorch.org/docs/stable/sparse.html#sparse-csr-docs)、[CSC](https://docs.pytorch.org/docs/stable/sparse.html#sparse-csc-docs)、[BSR](https://docs.pytorch.org/docs/stable/sparse.html#sparse-bsr-docs) 和 [BSC](https://docs.pytorch.org/docs/stable/sparse.html#sparse-bsc-docs)。

我们还有一个原型实现来支持 :ref:`半结构化稀疏 <sparse-semi-structured-docs>`。请参阅参考文献以了解更多详细信息。

请注意，我们提供了这些格式的轻微泛化。

批处理：GPU 等设备需要批处理才能获得最佳性能，因此我们支持批处理维度。

我们目前提供一种非常简单的批处理形式，其中稀疏格式的每个组件本身都进行了批处理。这还需要每个批处理条目具有相同数量的指定元素。在此示例中，我们从一个 3D 密集张量构建一个 3D（批处理）CSR 张量。

```
  t = torch.tensor([[[1., 0], [2., 3.]], [[4., 0], [5., 6.]]])
  t.dim()
3
  t.to_sparse_csr()
tensor(crow_indices=tensor([[0, 1, 3],
                            [0, 1, 3]]),
       col_indices=tensor([[0, 0, 1],
                           [0, 0, 1]]),
       values=tensor([[1., 2., 3.],
                      [4., 5., 6.]]), size=(2, 2, 2), nnz=3,
       layout=torch.sparse_csr)
```

密集维度：另一方面，一些数据（如图嵌入）可能更适合视为稀疏向量集合，而不是标量。

在此示例中，我们从一个 3D 跨步张量创建一个具有 2 个稀疏维度和 1 个密集维度的 3D 混合 COO 张量。如果 3D 跨步张量中的整行都为零，则不存储该行。但是，如果行中的任何值非零，则会存储整行。这减少了索引的数量，因为我们每行需要一个索引而不是每个元素一个索引。但这也增加了值的存储量。由于只能发出*完全*为零的行，并且任何非零值的存在都会导致存储整行。

```
  t = torch.tensor([[[0., 0], [1., 2.]], [[0., 0], [3., 4.]]])
  t.to_sparse(sparse_dim=2)
tensor(indices=tensor([[0, 1],
                       [1, 1]]),
       values=tensor([[1., 2.],
                      [3., 4.]]),
       size=(2, 2, 2), nnz=2, layout=torch.sparse_coo)
```

## 运算符概述

从根本上讲，对具有稀疏存储格式的张量的操作与对具有跨步（或其他）存储格式的张量的操作行为相同。存储的特定性，即数据的物理布局，会影响操作的性能，但不应影响语义。

我们正在积极增加稀疏张量的运算符覆盖范围。用户目前不应期望获得与密集张量相同的支持级别。请参阅我们的[运算符](https://docs.pytorch.org/docs/stable/sparse.html#sparse-ops-docs)文档以获取列表。

```
  b = torch.tensor([[0, 0, 1, 2, 3, 0], [4, 5, 0, 6, 0, 0]])
  b_s = b.to_sparse_csr()
  b_s.cos()
Traceback (most recent call last):
  File "&lt;stdin", line 1, in &lt;module
RuntimeError: unsupported tensor layout: SparseCsr
  b_s.sin()
tensor(crow_indices=tensor([0, 3, 6]),
       col_indices=tensor([2, 3, 4, 0, 1, 3]),
       values=tensor([ 0.8415,  0.9093,  0.1411, -0.7568, -0.9589, -0.2794]),
       size=(2, 6), nnz=6, layout=torch.sparse_csr)
```

如上例所示，我们不支持 cos 等非零保持的一元运算符。非零保持的一元运算的输出将无法像输入那样充分利用稀疏存储格式，并可能导致内存急剧增加。我们依赖用户先显式转换为密集张量，然后运行该运算。

```
  b_s.to_dense().cos()
tensor([[ 1.0000, -0.4161],
        [-0.9900,  1.0000]])
```

我们知道有些用户希望忽略 cos 等运算的压缩零值，而不是保留运算的确切语义。为此，我们可以指向 torch.masked 及其 MaskedTensor，它又由稀疏存储格式和内核支持和驱动。

另请注意，目前，用户没有输出布局的选择。例如，将稀疏张量添加到常规跨步张量会产生一个跨步张量。一些用户可能希望将其保留为稀疏布局，因为他们知道结果仍然足够稀疏。

```
  a + b.to_sparse()
tensor([[0., 3.],
        [3., 0.]])
```

我们承认，能够高效生成不同输出布局的内核的访问非常有用。后续操作可能会从接收特定布局中受益匪浅。我们正在开发一种 API 来控制结果布局，并认识到为任何给定模型规划更优执行路径是一项重要功能。

## 半结构化稀疏张量

警告

半结构化稀疏张量目前是一项原型功能，可能会发生变化。如果您想报告错误或分享反馈，请随时提出 issue。

半结构化稀疏性是 NVIDIA Ampere 架构首次引入的稀疏数据布局。它也被称为**细粒度结构化稀疏性**或**2:4 结构化稀疏性**。

此稀疏布局存储每 2n 个元素中的 n 个元素，其中 n 由张量数据类型 (dtype) 的宽度决定。最常用的 dtype 是 float16，此时 n=2，因此称为“2:4 结构化稀疏性”。

[此 NVIDIA 博文](https://developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt)更详细地解释了半结构化稀疏性。

在 PyTorch 中，半结构化稀疏性是通过张量子类实现的。通过子类化，我们可以覆盖 `__torch_dispatch__`，这使我们能够在执行矩阵乘法时使用更快的稀疏内核。我们还可以在子类内部以压缩形式存储张量，以减少内存开销。

在此压缩形式中，稀疏张量通过仅保留*指定的*元素和一些编码掩码的元数据来存储。

注意

半结构化稀疏张量的指定元素和元数据掩码存储在单个扁平压缩张量中。它们相互追加以形成连续的内存块。

压缩张量 = \[原始张量的指定元素 | 元数据掩码 \]

对于大小为 (r, c) 的原始张量，我们期望前 m \* k // 2 个元素是保留的元素，其余张量是元数据。

为了方便用户查看指定元素和掩码，可以使用 `.indices()` 和 `.values()` 分别访问掩码和指定元素。

- `.values()` 返回指定元素，大小为 (r, c//2)，数据类型与密集矩阵相同。

- `.indices()` 返回元数据掩码，大小为 (r, c//2)，如果 dtype 是 torch.float16 或 torch.bfloat16，则元素类型为 `torch.int16`；如果 dtype 是 torch.int8，则元素类型为 `torch.int32`。

对于 2:4 稀疏张量，元数据开销很小 - 每个指定元素仅 2 位。

注意

请注意，`torch.float32` 仅支持 1:2 稀疏性。因此，它不遵循上述相同公式。

在这里，我们将分解计算 2:4 稀疏张量的压缩率（密集大小 / 稀疏大小）。

设 (r, c) = tensor.shape，e = bitwidth(tensor.dtype)，则 e = 16 用于 `torch.float16` 和 `torch.bfloat16`，e = 8 用于 `torch.int8`。

$M_{dense} = r \times c \times e \\ M_{sparse} = M_{specified} + M_{metadata} = r \times \frac{c}{2} \times e + r \times \frac{c}{2} \times 2 = \frac{rce}{2} + rc =rce(\frac{1}{2} +\frac{1}{e})$

使用这些计算，我们可以确定原始密集表示和新稀疏表示的总内存占用。

这为我们提供了一个简单的压缩率公式，该公式仅取决于张量数据类型的位宽。

$C = \frac{M_{sparse}}{M_{dense}} =  \frac{1}{2} + \frac{1}{e}$

使用此公式，我们发现 `torch.float16` 或 `torch.bfloat16` 的压缩率为 56.25%，`torch.int8` 的压缩率为 62.5%。

### 构建半结构化稀疏张量

您可以通过简单地使用 `torch.to_sparse_semi_structured` 函数将密集张量转换为半结构化稀疏张量。

另请注意，我们仅支持 CUDA 张量，因为半结构化稀疏性的硬件兼容性仅限于 NVIDIA GPU。

以下数据类型支持半结构化稀疏性。请注意，每种数据类型都有其自己的形状约束和压缩因子。

要构建半结构化稀疏张量，请首先创建一个符合 2:4（或半结构化）稀疏格式的常规密集张量。为此，我们通过平铺一个小的 1x4 条带来创建 16x16 密集 float16 张量。之后，我们可以调用 `to_sparse_semi_structured` 函数来压缩它以进行加速推理。

```
  from torch.sparse import to_sparse_semi_structured
  A = torch.Tensor([0, 0, 1, 1]).tile((128, 32)).half().cuda()
tensor([[0., 0., 1.,  ..., 0., 1., 1.],
        [0., 0., 1.,  ..., 0., 1., 1.],
        [0., 0., 1.,  ..., 0., 1., 1.],
        ...,
        [0., 0., 1.,  ..., 0., 1., 1.],
        [0., 0., 1.,  ..., 0., 1., 1.],
        [0., 0., 1.,  ..., 0., 1., 1.]], device='cuda:0', dtype=torch.float16)
  A_sparse = to_sparse_semi_structured(A)
SparseSemiStructuredTensor(shape=torch.Size([128, 128]), transposed=False, values=tensor([[1., 1., 1.,  ..., 1., 1., 1.],
        [1., 1., 1.,  ..., 1., 1., 1.],
        [1., 1., 1.,  ..., 1., 1., 1.],
        ...,
        [1., 1., 1.,  ..., 1., 1., 1.],
        [1., 1., 1.,  ..., 1., 1., 1.],
        [1., 1., 1.,  ..., 1., 1., 1.]], device='cuda:0', dtype=torch.float16), metadata=tensor([[-4370, -4370, -4370,  ..., -4370, -4370, -4370],
        [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
        [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
        ...,
        [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
        [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
        [-4370, -4370, -4370,  ..., -4370, -4370, -4370]], device='cuda:0',
dtype=torch.int16))
```

### 半结构化稀疏张量运算

目前，半结构化稀疏张量支持以下运算：

- torch.addmm(bias, dense, sparse.t())

- torch.mm(dense, sparse)

- torch.mm(sparse, dense)

- aten.linear.default(dense, sparse, bias)

- aten.t.default(sparse)

- aten.t.detach(sparse)

要使用这些运算，只需在张量处于半结构化稀疏格式（具有零值）后，将 `to_sparse_semi_structured(tensor)` 的输出传递进去，而不是使用 `tensor`，如下所示：

```
  a = torch.Tensor([0, 0, 1, 1]).tile((64, 16)).half().cuda()
  b = torch.rand(64, 64).half().cuda()
  c = torch.mm(a, b)
  a_sparse = to_sparse_semi_structured(a)
  torch.allclose(c, torch.mm(a_sparse, b))
True
```

### 使用半结构化稀疏性加速 nn.Linear

只需几行代码，即可加速模型中的线性层（如果权重已经是半结构化稀疏的）：

```
  input = torch.rand(64, 64).half().cuda()
  mask = torch.Tensor([0, 0, 1, 1]).tile((64, 16)).cuda().bool()
  linear = nn.Linear(64, 64).half().cuda()
  linear.weight = nn.Parameter(to_sparse_semi_structured(linear.weight.masked_fill(~mask, 0)))
```

## 稀疏 COO 张量

PyTorch 以坐标格式（COO 格式）作为实现稀疏张量的存储格式之一。在 COO 格式中，指定元素存储为元素索引及其对应值的元组。具体来说：

> - 指定元素的索引收集在大小为 `(ndim, nse)` 且元素类型为 `torch.int64` 的 `indices` 张量中。
>
> - 对应的*值*收集在大小为 `(nse,)` 且具有任意整数或浮点数元素类型的 `values` 张量中。

其中 `ndim` 是张量的维数，`nse` 是指定元素的数量。

注意

稀疏 COO 张量的内存消耗至少为 `(ndim * 8 + <元素类型的字节大小>) * nse` 字节（加上存储其他张量数据的常量开销）。

跨步张量的内存消耗至少为 `product(<张量形状>) * <元素类型的字节大小>`。

例如，使用 COO 张量布局时，一个具有 100,000 个非零 32 位浮点数的 10,000 x 10,000 张量的内存消耗至少为 `(2 * 8 + 4) * 100,000 = 2,000,000` 字节，而使用默认的跨步张量布局时为 `10,000 * 10,000 * 4 = 400,000,000` 字节。请注意，使用 COO 存储格式可节省 200 倍的内存。

### 构建

可以通过向 [`torch.sparse_coo_tensor()`](https://docs.pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html#torch.sparse_coo_tensor "torch.sparse_coo_tensor") 函数提供索引和值这两个张量，以及稀疏张量的大小（如果无法从索引和值张量推断）来构造稀疏 COO 张量。

假设我们要定义一个稀疏张量，其中位置 (0, 2) 的条目为 3，位置 (1, 0) 的条目为 4，位置 (1, 2) 的条目为 5。未指定的元素假定具有相同的值，即填充值，默认为零。然后，我们将编写：

```
  i = [[0, 1, 1], [2, 0, 2]]
  v =  [3, 4, 5]
  s = torch.sparse_coo_tensor(i, v, (2, 3))
  s
tensor(indices=tensor([[0, 1, 1],
                       [2, 0, 2]]),
       values=tensor([3, 4, 5]),
       size=(2, 3), nnz=3, layout=torch.sparse_coo)
  s.to_dense()
tensor([[0, 0, 3],
        [4, 0, 5]])
```

请注意，输入 `i` **不是**索引元组列表。如果您想这样编写索引，则应在传递给稀疏构造函数之前进行转置：

```
  i = [[0, 2], [1, 0], [1, 2]]
  v =  [3,      4,      5    ]
  s = torch.sparse_coo_tensor(list(zip(*i)), v, (2, 3))
  # Or another equivalent formulation to get s
  s = torch.sparse_coo_tensor(torch.tensor(i).t(), v, (2, 3))
  torch.sparse_coo_tensor(i.t(), v, torch.Size([2,3])).to_dense()
tensor([[0, 0, 3],
        [4, 0, 5]])
```

可以通过仅指定其大小来构造一个空的稀疏 COO 张量：

```
  torch.sparse_coo_tensor(size=(2, 3))
tensor(indices=tensor([], size=(2, 0)),
       values=tensor([], size=(0,)),
       size=(2, 3), nnz=0, layout=torch.sparse_coo)
```

### 稀疏混合 COO 张量

PyTorch 实现了一个稀疏张量的扩展，从标量值到稀疏张量（包含（连续）张量值）。这种张量称为混合张量。

PyTorch 混合 COO 张量扩展了稀疏 COO 张量，允许 `values` 张量成为多维张量，因此我们有：

> - 指定元素的索引收集在大小为 `(sparse_dims, nse)` 且元素类型为 `torch.int64` 的 `indices` 张量中。
>
> - 对应的（张量）值收集在大小为 `(nse, dense_dims)` 且具有任意整数或浮点数元素类型的 `values` 张量中。

注意

我们使用 (M + K) 维张量来表示 N 维稀疏混合张量，其中 M 和 K 分别是稀疏和密集维度的数量，使得 M + K == N。

假设我们要创建一个 (2 + 1) 维张量，其中位置 (0, 2) 的条目为 \[3, 4]，位置 (1, 0) 的条目为 \[5, 6]，位置 (1, 2) 的条目为 \[7, 8]。我们将编写：

```
  i = [[0, 1, 1],
         [2, 0, 2]]
  v =  [[3, 4], [5, 6], [7, 8]]
  s = torch.sparse_coo_tensor(i, v, (2, 3, 2))
  s
tensor(indices=tensor([[0, 1, 1],
                       [2, 0, 2]]),
       values=tensor([[3, 4],
                      [5, 6],
                      [7, 8]]),
       size=(2, 3, 2), nnz=3, layout=torch.sparse_coo)
```

```
  s.to_dense()
tensor([[[0, 0],
         [0, 0],
         [3, 4]],
        [[5, 6],
         [0, 0],
         [7, 8]]])
```

通常，如果 `s` 是稀疏 COO 张量，且 `M = s.sparse_dim()`，`K = s.dense_dim()`，则我们有以下不变性：

> - `M + K == len(s.shape) == s.ndim` - 张量的维度是稀疏和密集维度数量的总和，
>
> - `s.indices().shape == (M, nse)` - 稀疏索引被显式存储，
>
> - `s.values().shape == (nse,) + s.shape[M : M + K]` - 混合张量的值是 K 维张量，
>
> - `s.values().layout == torch.strided` - 值存储为跨步张量。

注意

密集维度始终跟随稀疏维度，即不支持混合密集和稀疏维度。

注意

为了确保构造的稀疏张量具有一致的索引、值和大小，可以通过 `check_invariants=True` 关键字参数按张量创建启用不变性检查，或者使用 [`torch.sparse.check_sparse_tensor_invariants`](https://docs.pytorch.org/docs/stable/generated/torch.sparse.check_sparse_tensor_invariants.html#torch.sparse.check_sparse_tensor_invariants "torch.sparse.check_sparse_tensor_invariants") 上下文管理器实例全局启用。默认情况下，稀疏张量不变性检查是禁用的。

### 未合并的稀疏 COO 张量

PyTorch 稀疏 COO 张量格式允许稀疏*未合并*张量，其中索引可能存在重复坐标；在这种情况下，解释是该索引处的值是所有重复值条目的总和。例如，可以为同一个索引 `1` 指定多个值 `3` 和 `4`，这会导致一个一维未合并张量：

```
  i = [[1, 1]]
  v =  [3, 4]
  s=torch.sparse_coo_tensor(i, v, (3,))
  s
tensor(indices=tensor([[1, 1]]),
       values=tensor(  [3, 4]),
       size=(3,), nnz=2, layout=torch.sparse_coo)
```

而合并过程将使用求和将多值元素累积为单个值：

```
  s.coalesce()
tensor(indices=tensor([[1]]),
       values=tensor([7]),
       size=(3,), nnz=1, layout=torch.sparse_coo)
```

通常，[`torch.Tensor.coalesce()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.coalesce.html#torch.Tensor.coalesce "torch.Tensor.coalesce") 方法的输出是一个稀疏张量，具有以下属性：

- 指定张量元素的索引是唯一的，

- 索引按字典顺序排序，

- [`torch.Tensor.is_coalesced()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.is_coalesced.html#torch.Tensor.is_coalesced "torch.Tensor.is_coalesced") 返回 `True`。

注意

在大多数情况下，您不必关心稀疏张量是否已合并，因为大多数操作在给定合并或未合并的稀疏张量时行为相同。

但是，某些操作可以在未合并张量上更高效地实现，而另一些操作可以在合并张量上更高效地实现。

例如，稀疏 COO 张量相加是通过简单地连接索引和值张量来实现的：

```
  a = torch.sparse_coo_tensor([[1, 1]], [5, 6], (2,))
  b = torch.sparse_coo_tensor([[0, 0]], [7, 8], (2,))
  a + b
tensor(indices=tensor([[0, 0, 1, 1]]),
       values=tensor([7, 8, 5, 6]),
       size=(2,), nnz=4, layout=torch.sparse_coo)
```

如果您反复执行可能产生重复条目的操作（例如 [`torch.Tensor.add()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.add.html#torch.Tensor.add "torch.Tensor.add")），您应该偶尔合并稀疏张量，以防止它们变得太大。

另一方面，索引的字典序对于实现涉及许多元素选择操作（如切片或矩阵乘积）的算法可能是有利的。

### 使用稀疏 COO 张量

让我们考虑以下示例：

```
  i = [[0, 1, 1],
         [2, 0, 2]]
  v =  [[3, 4], [5, 6], [7, 8]]
  s = torch.sparse_coo_tensor(i, v, (2, 3, 2))
```

如上所述，稀疏 COO 张量是 [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor") 实例，为了将其与使用其他布局的张量实例区分开来，可以使用 [`torch.Tensor.is_sparse`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.is_sparse.html#torch.Tensor.is_sparse "torch.Tensor.is_sparse") 或 `torch.Tensor.layout` 属性：

```
  isinstance(s, torch.Tensor)
True
  s.is_sparse
True
  s.layout == torch.sparse_coo
True
```

稀疏和密集维度的数量可以使用方法 [`torch.Tensor.sparse_dim()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.sparse_dim.html#torch.Tensor.sparse_dim "torch.Tensor.sparse_dim") 和 [`torch.Tensor.dense_dim()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.dense_dim.html#torch.Tensor.dense_dim "torch.Tensor.dense_dim") 分别获取。例如：

```
  s.sparse_dim(), s.dense_dim()
(2, 1)
```

如果 `s` 是稀疏 COO 张量，则可以使用方法 [`torch.Tensor.indices()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.indices.html#torch.Tensor.indices "torch.Tensor.indices") 和 [`torch.Tensor.values()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.values.html#torch.Tensor.values "torch.Tensor.values") 获取其 COO 格式数据。

注意

当前，只有当张量实例已合并时，才能获取 COO 格式数据：

```
  s.indices()
RuntimeError: Cannot get indices on an uncoalesced tensor, please call .coalesce() first
```

要获取未合并张量的 COO 格式数据，请使用 `torch.Tensor._values()` 和 `torch.Tensor._indices()`：

```
  s._indices()
tensor([[0, 1, 1],
        [2, 0, 2]])
```

警告

调用 `torch.Tensor._values()` 将返回一个*已分离*的张量。要跟踪梯度，必须改用 `torch.Tensor.coalesce().values()`。

构造一个新的稀疏 COO 张量会得到一个未合并的张量：

```
  s.is_coalesced()
False
```

但是，可以使用 [`torch.Tensor.coalesce()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.coalesce.html#torch.Tensor.coalesce "torch.Tensor.coalesce") 方法构造稀疏 COO 张量的合并副本：

```
  s2 = s.coalesce()
  s2.indices()
tensor([[0, 1, 1],
       [2, 0, 2]])
```

在使用未合并稀疏 COO 张量时，必须考虑到未合并数据的加法性质：相同索引的值是求和的项，求和结果即为相应张量元素的值。例如，稀疏未合并张量上的标量乘法可以通过将所有未合并值乘以标量来实现，因为 `c * (a + b) == c * a + c * b` 成立。但是，任何非线性运算，例如平方根，都不能通过对未合并数据应用运算来实现，因为 `sqrt(a + b) == sqrt(a) + sqrt(b)` 通常不成立。

稀疏 COO 张量的切片（步长为正）仅支持密集维度。索引同时支持稀疏和密集维度：

```
  s[1]
tensor(indices=tensor([[0, 2]]),
       values=tensor([[5, 6],
                      [7, 8]]),
       size=(3, 2), nnz=2, layout=torch.sparse_coo)
  s[1, 0, 1]
tensor(6)
  s[1, 0, 1:]
tensor([6])
```

在 PyTorch 中，稀疏张量的填充值不能显式指定，通常假定为零。但是，存在一些运算可能以不同的方式解释填充值。例如，[`torch.sparse.softmax()`](https://docs.pytorch.org/docs/stable/generated/torch.sparse.softmax.html#torch.sparse.softmax "torch.sparse.softmax") 在填充值为负无穷大的假设下计算 softmax。

## 稀疏压缩张量

稀疏压缩张量是一类稀疏张量，它们具有一个共同的特征，即使用一种编码来压缩某个维度的索引，这种编码能够对稀疏压缩张量的线性代数内核进行某些优化。这种编码基于 [行压缩存储（CSR）](<https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)>) 格式，PyTorch 稀疏压缩张量通过支持稀疏张量批次、允许多维张量值以及以密集块形式存储稀疏张量值来对其进行扩展。

注意

我们使用 (B + M + K) 维张量来表示 N 维稀疏压缩混合张量，其中 B、M 和 K 分别是批处理、稀疏和密集维度的数量，使得 B + M + K == N。稀疏压缩张量的稀疏维度数量始终为两个，M == 2。

注意

我们说索引张量 `compressed_indices` 使用 CSR 编码，如果满足以下不变量：

- `compressed_indices` 是一个连续的跨步 32 位或 64 位整数张量。

- `compressed_indices` 的形状为 `(*batchsize, compressed_dim_size + 1)`，其中 `compressed_dim_size` 是压缩维度（例如行或列）的数量。

- `compressed_indices[..., 0] == 0`，其中 `...` 表示批处理索引。

- `compressed_indices[..., compressed_dim_size] == nse`，其中 `nse` 是指定元素的数量。

- 对于 `i=1, ..., compressed_dim_size`，`0 <= compressed_indices[..., i] - compressed_indices[..., i - 1] <= plain_dim_size`，其中 `plain_dim_size` 是普通维度（与压缩维度正交，例如列或行）的数量。

为了确保构造的稀疏张量具有一致的索引、值和大小，可以通过 `check_invariants=True` 关键字参数按张量创建启用不变性检查，或者使用 [`torch.sparse.check_sparse_tensor_invariants`](https://docs.pytorch.org/docs/stable/generated/torch.sparse.check_sparse_tensor_invariants.html#torch.sparse.check_sparse_tensor_invariants "torch.sparse.check_sparse_tensor_invariants") 上下文管理器实例全局启用。默认情况下，稀疏张量不变性检查是禁用的。

注意

将稀疏压缩布局泛化到 N 维张量可能会导致对指定元素数量的一些混淆。当稀疏压缩张量包含批处理维度时，指定元素的数量将对应于每个批处理的此类元素的数量。当稀疏压缩张量具有密集维度时，所考虑的元素现在是具有自身维度的 K 维数组。对于块稀疏压缩布局，2D 块被视为指定的元素。以一个三维块稀疏张量为例，它有一个长度为 `b` 的批处理维度，以及一个 `p, q` 的块形状。如果此张量有 `n` 个指定元素，则实际上我们每个批处理有 `n` 个指定块。此指定元素数量的解释源于所有稀疏压缩布局都源自二维矩阵的压缩。批处理维度被视为稀疏矩阵的堆叠，密集维度将元素的意思从简单的标量值更改为具有自身维度的数组。

### 稀疏 CSR 张量

CSR 格式相对于 COO 格式的主要优点是更好地利用存储空间，并且使用 MKL 和 MAGMA 后端进行稀疏矩阵向量乘法等计算操作速度更快。

在最简单的情况下，(0 + 2 + 0) 维稀疏 CSR 张量由三个一维张量组成：`crow_indices`、`col_indices` 和 `values`：

> - `crow_indices` 张量由压缩的行索引组成。这是一个大小为 `nrows + 1`（行数加 1）的一维张量。`crow_indices` 的最后一个元素是指定元素的数量，即 `nse`。此张量根据给定行开始的位置编码 `values` 和 `col_indices` 中的索引。张量中的每个连续数字减去其前面的数字表示给定行中的元素数量。
>
> - `col_indices` 张量包含每个元素的列索引。这是一个大小为 `nse` 的一维张量。
>
> - `values` 张量包含 CSR 张量元素的*值*。这是一个大小为 `nse` 的一维张量。

注意

索引张量 `crow_indices` 和 `col_indices` 的元素类型应为 `torch.int64`（默认）或 `torch.int32`。如果要使用启用 MKL 的矩阵运算，请使用 `torch.int32`。这是因为 PyTorch 的默认链接是 MKL LP64，它使用 32 位整数索引。

在一般情况下，(B + 2 + K) 维稀疏 CSR 张量由两个 (B + 1) 维索引张量 `crow_indices` 和 `col_indices`，以及 (1 + K) 维 `values` 张量组成，使得

> - `crow_indices.shape == (*batchsize, nrows + 1)`
>
> - `col_indices.shape == (*batchsize, nse)`
>
> - `values.shape == (nse, *densesize)`

而稀疏 CSR 张量的形状为 `(*batchsize, nrows, ncols, *densesize)`，其中 `len(batchsize) == B` 且 `len(densesize) == K`。

注意

稀疏 CSR 张量的批次是依赖的：所有批次中的指定元素数量必须相同。这个有些人为的约束允许高效存储不同 CSR 批次的索引。

注意

稀疏和密集维度的数量可以使用 [`torch.Tensor.sparse_dim()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.sparse_dim.html#torch.Tensor.sparse_dim "torch.Tensor.sparse_dim") 和 [`torch.Tensor.dense_dim()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.dense_dim.html#torch.Tensor.dense_dim "torch.Tensor.dense_dim") 方法获取。批处理维度可以从张量形状计算得出：`batchsize = tensor.shape[:-tensor.sparse_dim() - tensor.dense_dim()]`。

注意

稀疏 CSR 张量的内存消耗至少为 `(nrows * 8 + (8 + <元素类型的字节大小> * prod(densesize)) * nse) * prod(batchsize)` 字节（加上存储其他张量数据的常量开销）。

使用[稀疏 COO 格式介绍中的注释](https://docs.pytorch.org/docs/stable/sparse.html#sparse-coo-docs)中的相同示例数据，使用 CSR 张量布局时，一个具有 100,000 个非零 32 位浮点数的 10,000 x 10,000 张量的内存消耗至少为 `(10000 * 8 + (8 + 4 * 1) * 100 000) * 1 = 1,280,000` 字节。与使用 COO 和跨步格式相比，使用 CSR 存储格式分别节省了 1.6 倍和 310 倍的内存。

#### CSR 张量构造

可以使用 [`torch.sparse_csr_tensor()`](https://docs.pytorch.org/docs/stable/generated/torch.sparse_csr_tensor.html#torch.sparse_csr_tensor "torch.sparse_csr_tensor") 函数直接构造稀疏 CSR 张量。用户必须分别提供行和列索引以及值张量，其中行索引必须使用 CSR 压缩编码指定。`size` 参数是可选的，如果不存在，则将从 `crow_indices` 和 `col_indices` 中推断出来。

```
  crow_indices = torch.tensor([0, 2, 4])
  col_indices = torch.tensor([0, 1, 0, 1])
  values = torch.tensor([1, 2, 3, 4])
  csr = torch.sparse_csr_tensor(crow_indices, col_indices, values, dtype=torch.float64)
  csr
tensor(crow_indices=tensor([0, 2, 4]),
       col_indices=tensor([0, 1, 0, 1]),
       values=tensor([1., 2., 3., 4.]), size=(2, 2), nnz=4,
       dtype=torch.float64)
  csr.to_dense()
tensor([[1., 2.],
        [3., 4.]], dtype=torch.float64)
```

注意

从推断出的 `size` 中计算稀疏维度的值，该值由 `crow_indices` 的大小和 `col_indices` 中的最大索引值计算得出。如果列数需要大于推断出的 `size`，则必须显式指定 `size` 参数。

从跨步张量或稀疏 COO 张量构造 2D 稀疏 CSR 张量的最简单方法是使用 [`torch.Tensor.to_sparse_csr()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to_sparse_csr.html#torch.Tensor.to_sparse_csr "torch.Tensor.to_sparse_csr") 方法。（跨步）张量中的任何零值都将被解释为稀疏张量中的缺失值：

```
  a = torch.tensor([[0, 0, 1, 0], [1, 2, 0, 0], [0, 0, 0, 0]], dtype=torch.float64)
  sp = a.to_sparse_csr()
  sp
tensor(crow_indices=tensor([0, 1, 3, 3]),
      col_indices=tensor([2, 0, 1]),
      values=tensor([1., 1., 2.]), size=(3, 4), nnz=3, dtype=torch.float64)
```

#### CSR 张量运算

可以使用 `tensor.matmul()` 方法执行稀疏矩阵向量乘法。这是目前 CSR 张量上唯一支持的数学运算。

```
  vec = torch.randn(4, 1, dtype=torch.float64)
  sp.matmul(vec)
tensor([[0.9078],
        [1.3180],
        [0.0000]], dtype=torch.float64)
```

### 稀疏 CSC 张量

稀疏 CSC（列压缩存储）张量格式实现了 CSC 格式来存储二维张量，并扩展支持稀疏 CSC 张量批次以及多维张量值。

注意

稀疏 CSC 张量本质上是稀疏 CSR 张量的转置，当转置是交换稀疏维度时。

与[稀疏 CSR 张量](https://docs.pytorch.org/docs/stable/sparse.html#sparse-csr-docs)类似，稀疏 CSC 张量由三个张量组成：`ccol_indices`、`row_indices` 和 `values`：

> - `ccol_indices` 张量由压缩的列索引组成。这是一个形状为 `(*batchsize, ncols + 1)` 的 (B + 1) 维张量。最后一个元素是指定元素的数量，即 `nse`。此张量根据给定列开始的位置编码 `values` 和 `row_indices` 中的索引。张量中的每个连续数字减去其前面的数字表示给定列中的元素数量。
>
> - `row_indices` 张量包含每个元素的行索引。这是一个形状为 `(*batchsize, nse)` 的 (B + 1) 维张量。
>
> - `values` 张量包含 CSC 张量元素的*值*。这是一个形状为 `(nse, *densesize)` 的 (1 + K) 维张量。

#### CSC 张量构造

可以使用 [`torch.sparse_csc_tensor()`](https://docs.pytorch.org/docs/stable/generated/torch.sparse_csc_tensor.html#torch.sparse_csc_tensor "torch.sparse_csc_tensor") 函数直接构造稀疏 CSC 张量。用户必须分别提供行和列索引以及值张量，其中列索引必须使用 CSR 压缩编码指定。`size` 参数是可选的，如果不存在，则将从 `row_indices` 和 `ccol_indices` 张量中推断出来。

```
  ccol_indices = torch.tensor([0, 2, 4])
  row_indices = torch.tensor([0, 1, 0, 1])
  values = torch.tensor([1, 2, 3, 4])
  csc = torch.sparse_csc_tensor(ccol_indices, row_indices, values, dtype=torch.float64)
  csc
tensor(ccol_indices=tensor([0, 2, 4]),
       row_indices=tensor([0, 1, 0, 1]),
       values=tensor([1., 2., 3., 4.]), size=(2, 2), nnz=4,
       dtype=torch.float64, layout=torch.sparse_csc)
  csc.to_dense()
tensor([[1., 3.],
        [2., 4.]], dtype=torch.float64)
```

注意

稀疏 CSC 张量构造函数在行索引参数之前有压缩列索引参数。

可以使用 [`torch.Tensor.to_sparse_csc()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to_sparse_csc.html#torch.Tensor.to_sparse_csc "torch.Tensor.to_sparse_csc") 方法从任何二维张量构造 (0 + 2 + 0) 维稀疏 CSC 张量。二维张量中的任何零值都将被解释为稀疏张量中的缺失值：

```
  a = torch.tensor([[0, 0, 1, 0], [1, 2, 0, 0], [0, 0, 0, 0]], dtype=torch.float64)
  sp = a.to_sparse_csc()
  sp
tensor(ccol_indices=tensor([0, 1, 2, 3, 3]),
       row_indices=tensor([1, 1, 0]),
       values=tensor([1., 2., 1.]), size=(3, 4), nnz=3, dtype=torch.float64,
       layout=torch.sparse_csc)
```

### 稀疏 BSR 张量

稀疏 BSR（块压缩行存储）张量格式实现了 BSR 格式来存储二维张量，并扩展支持稀疏 BSR 张量批次以及多维张量值块。

稀疏 BSR 张量由三个张量组成：`crow_indices`、`col_indices` 和 `values`：

> - `crow_indices` 张量由压缩的行索引组成。这是一个形状为 `(*batchsize, nrowblocks + 1)` 的 (B + 1) 维张量。最后一个元素是指定块的数量，即 `nse`。此张量编码 `values` 和 `col_indices` 中取决于给定列块开始位置的索引。张量中的每个连续数字减去其前面的数字表示给定行中的块数量。
>
> - `col_indices` 张量包含每个元素的列块索引。这是一个形状为 `(*batchsize, nse)` 的 (B + 1) 维张量。
>
> - `values` 张量包含收集到二维块中的稀疏 BSR 张量元素的*值*。这是一个形状为 `(nse, nrowblocks, ncolblocks, *densesize)` 的 (1 + 2 + K) 维张量。

#### BSR 张量构造

可以使用 [`torch.sparse_bsr_tensor()`](https://docs.pytorch.org/docs/stable/generated/torch.sparse_bsr_tensor.html#torch.sparse_bsr_tensor "torch.sparse_bsr_tensor") 函数直接构造稀疏 BSR 张量。用户必须分别提供行和列块索引以及值张量，其中行块索引必须使用 CSR 压缩编码指定。`size` 参数是可选的，如果不存在，则将从 `crow_indices` 和 `col_indices` 张量中推断出来。

```
  crow_indices = torch.tensor([0, 2, 4])
  col_indices = torch.tensor([0, 1, 0, 1])
  values = torch.tensor([[[0, 1, 2], [6, 7, 8]],
...                        [[3, 4, 5], [9, 10, 11]],
...                        [[12, 13, 14], [18, 19, 20]],
...                        [[15, 16, 17], [21, 22, 23]]])
  bsr = torch.sparse_bsr_tensor(crow_indices, col_indices, values, dtype=torch.float64)
  bsr
tensor(crow_indices=tensor([0, 2, 4]),
       col_indices=tensor([0, 1, 0, 1]),
       values=tensor([[[ 0.,  1.,  2.],
                       [ 6.,  7.,  8.]],
                      [[ 3.,  4.,  5.],
                       [ 9., 10., 11.]],
                      [[12., 13., 14.],
                       [18., 19., 20.]],
                      [[15., 16., 17.],
                       [21., 22., 23.]]]),
       size=(4, 6), nnz=4, dtype=torch.float64, layout=torch.sparse_bsr)
  bsr.to_dense()
tensor([[ 0.,  1.,  2.,  3.,  4.,  5.],
        [ 6.,  7.,  8.,  9., 10., 11.],
        [12., 13., 14., 15., 16., 17.],
        [18., 19., 20., 21., 22., 23.]], dtype=torch.float64)
```

可以使用 [`torch.Tensor.to_sparse_bsr()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to_sparse_bsr.html#torch.Tensor.to_sparse_bsr "torch.Tensor.to_sparse_bsr") 方法从任何二维张量构造 (0 + 2 + 0) 维稀疏 BSR 张量，该方法还需要指定值块大小：

```
  dense = torch.tensor([[0, 1, 2, 3, 4, 5],
...                       [6, 7, 8, 9, 10, 11],
...                       [12, 13, 14, 15, 16, 17],
...                       [18, 19, 20, 21, 22, 23]])
  bsr = dense.to_sparse_bsr(blocksize=(2, 3))
  bsr
tensor(crow_indices=tensor([0, 2, 4]),
       col_indices=tensor([0, 1, 0, 1]),
       values=tensor([[[ 0,  1,  2],
                       [ 6,  7,  8]],
                      [[ 3,  4,  5],
                       [ 9, 10, 11]],
                      [[12, 13, 14],
                       [18, 19, 20]],
                      [[15, 16, 17],
                       [21, 22, 23]]]), size=(4, 6), nnz=4,
       layout=torch.sparse_bsr)
```

### 稀疏 BSC 张量

稀疏 BSC（块压缩列存储）张量格式实现了 BSC 格式来存储二维张量，并扩展支持稀疏 BSC 张量批次以及多维张量块。

稀疏 BSC 张量由三个张量组成：`ccol_indices`、`row_indices` 和 `values`：

> - `ccol_indices` 张量由压缩的列索引组成。这是一个形状为 `(*batchsize, ncolblocks + 1)` 的 (B + 1) 维张量。最后一个元素是指定块的数量，即 `nse`。此张量编码 `values` 和 `row_indices` 中取决于给定行块开始位置的索引。张量中的每个连续数字减去其前面的数字表示给定列中的块数量。
>
> - `row_indices` 张量包含每个元素的行块索引。这是一个形状为 `(*batchsize, nse)` 的 (B + 1) 维张量。
>
> - `values` 张量包含收集到二维块中的稀疏 BSC 张量元素的*值*。这是一个形状为 `(nse, nrowblocks, ncolblocks, *densesize)` 的 (1 + 2 + K) 维张量。

#### BSC 张量构造

可以使用 [`torch.sparse_bsc_tensor()`](https://docs.pytorch.org/docs/stable/generated/torch.sparse_bsc_tensor.html#torch.sparse_bsc_tensor "torch.sparse_bsc_tensor") 函数直接构造稀疏 BSC 张量。用户必须分别提供行和列块索引以及值张量，其中列块索引必须使用 CSR 压缩编码指定。`size` 参数是可选的，如果不存在，则将从 `ccol_indices` 和 `row_indices` 张量中推断出来。

```
  ccol_indices = torch.tensor([0, 2, 4])
  row_indices = torch.tensor([0, 1, 0, 1])
  values = torch.tensor([[[0, 1, 2], [6, 7, 8]],
...                        [[3, 4, 5], [9, 10, 11]],
...                        [[12, 13, 14], [18, 19, 20]],
...                        [[15, 16, 17], [21, 22, 23]]])
  bsc = torch.sparse_bsc_tensor(ccol_indices, row_indices, values, dtype=torch.float64)
  bsc
tensor(ccol_indices=tensor([0, 2, 4]),
       row_indices=tensor([0, 1, 0, 1]),
       values=tensor([[[ 0.,  1.,  2.],
                       [ 6.,  7.,  8.]],
                      [[ 3.,  4.,  5.],
                       [ 9., 10., 11.]],
                      [[12., 13., 14.],
                       [18., 19., 20.]],
                      [[15., 16., 17.],
                       [21., 22., 23.]]]), size=(4, 6), nnz=4,
       dtype=torch.float64, layout=torch.sparse_bsc)
```

## 支持的运算

### 线性代数运算

下表总结了稀疏矩阵支持的线性代数运算，其中操作数布局可能不同。此处 `T[layout]` 表示具有给定布局的张量。类似地，`M[layout]` 表示矩阵（2D PyTorch 张量），`V[layout]` 表示向量（1D PyTorch 张量）。此外，`f` 表示标量（浮点数或 0D PyTorch 张量），`*` 表示逐元素乘法，`@` 表示矩阵乘法。

其中“稀疏梯度？”列指示 PyTorch 操作是否支持相对于稀疏矩阵参数的反向传播。除 [`torch.smm()`](https://docs.pytorch.org/docs/stable/generated/torch.smm.html#torch.smm "torch.smm") 外，所有 PyTorch 操作都支持相对于跨步矩阵参数的反向传播。

注意

当前，PyTorch 不支持布局签名 `M[strided] @ M[sparse_coo]` 的矩阵乘法。但是，应用程序仍可以使用矩阵关系 `D @ S == (S.t() @ D.t()).t()` 来计算此值。

### 仅适用于稀疏张量的 Torch 函数
