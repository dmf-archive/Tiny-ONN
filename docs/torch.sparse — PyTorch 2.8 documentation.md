---
created: 2025-09-08T02:25:05 (UTC +00:00)
tags: []
source: https://docs.pytorch.org/docs/stable/sparse.html
author: PyTorch Contributors
---

# torch.sparse — PyTorch 2.8 documentation

> ## Excerpt
> PyTorch Documentation. Explore PyTorch, an open-source machine learning library that accelerates the path from research prototyping to production deployment. Discover tutorials, API references, and guides to help you build and deploy deep learning models efficiently.

---
Created On: Apr 26, 2017 | Last Updated On: Jun 18, 2025

Warning

The PyTorch API of sparse tensors is in beta and may change in the near future. We highly welcome feature requests, bug reports and general suggestions as GitHub issues.

## Why and when to use sparsity

By default, PyTorch stores [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor") elements contiguously in physical memory. This leads to efficient implementations of various array processing algorithms that require fast access to elements.

Now, some users might decide to represent data such as graph adjacency matrices, pruned weights or points clouds by Tensors whose _elements are mostly zero valued_. We recognize these are important applications and aim to provide performance optimizations for these use cases via sparse storage formats.

Various sparse storage formats such as COO, CSR/CSC, semi-structured, LIL, etc. have been developed over the years. While they differ in exact layouts, they all compress data through efficient representation of zero valued elements. We call the uncompressed values _specified_ in contrast to _unspecified_, compressed elements.

By compressing repeat zeros sparse storage formats aim to save memory and computational resources on various CPUs and GPUs. Especially for high degrees of sparsity or highly structured sparsity this can have significant performance implications. As such sparse storage formats can be seen as a performance optimization.

Like many other performance optimization sparse storage formats are not always advantageous. When trying sparse formats for your use case you might find your execution time to increase rather than decrease.

Please feel encouraged to open a GitHub issue if you analytically expected to see a stark increase in performance but measured a degradation instead. This helps us prioritize the implementation of efficient kernels and wider performance optimizations.

We make it easy to try different sparsity layouts, and convert between them, without being opinionated on what’s best for your particular application.

## Functionality overview

We want it to be straightforward to construct a sparse Tensor from a given dense Tensor by providing conversion routines for each layout.

In the next example we convert a 2D Tensor with default dense (strided) layout to a 2D Tensor backed by the COO memory layout. Only values and indices of non-zero elements are stored in this case.

```
<span></span><span>&gt;&gt;&gt; </span><span>a</span> <span>=</span> <span>torch</span><span>.</span><span>tensor</span><span>([[</span><span>0</span><span>,</span> <span>2.</span><span>],</span> <span>[</span><span>3</span><span>,</span> <span>0</span><span>]])</span>
<span>&gt;&gt;&gt; </span><span>a</span><span>.</span><span>to_sparse</span><span>()</span>
<span>tensor(indices=tensor([[0, 1],</span>
<span>                       [1, 0]]),</span>
<span>       values=tensor([2., 3.]),</span>
<span>       size=(2, 2), nnz=2, layout=torch.sparse_coo)</span>
```

PyTorch currently supports [COO](https://docs.pytorch.org/docs/stable/sparse.html#sparse-coo-docs), [CSR](https://docs.pytorch.org/docs/stable/sparse.html#sparse-csr-docs), [CSC](https://docs.pytorch.org/docs/stable/sparse.html#sparse-csc-docs), [BSR](https://docs.pytorch.org/docs/stable/sparse.html#sparse-bsr-docs), and [BSC](https://docs.pytorch.org/docs/stable/sparse.html#sparse-bsc-docs).

We also have a prototype implementation to support :ref: semi-structured sparsity<sparse-semi-structured-docs>. Please see the references for more details.

Note that we provide slight generalizations of these formats.

Batching: Devices such as GPUs require batching for optimal performance and thus we support batch dimensions.

We currently offer a very simple version of batching where each component of a sparse format itself is batched. This also requires the same number of specified elements per batch entry. In this example we construct a 3D (batched) CSR Tensor from a 3D dense Tensor.

```
<span></span><span>&gt;&gt;&gt; </span><span>t</span> <span>=</span> <span>torch</span><span>.</span><span>tensor</span><span>([[[</span><span>1.</span><span>,</span> <span>0</span><span>],</span> <span>[</span><span>2.</span><span>,</span> <span>3.</span><span>]],</span> <span>[[</span><span>4.</span><span>,</span> <span>0</span><span>],</span> <span>[</span><span>5.</span><span>,</span> <span>6.</span><span>]]])</span>
<span>&gt;&gt;&gt; </span><span>t</span><span>.</span><span>dim</span><span>()</span>
<span>3</span>
<span>&gt;&gt;&gt; </span><span>t</span><span>.</span><span>to_sparse_csr</span><span>()</span>
<span>tensor(crow_indices=tensor([[0, 1, 3],</span>
<span>                            [0, 1, 3]]),</span>
<span>       col_indices=tensor([[0, 0, 1],</span>
<span>                           [0, 0, 1]]),</span>
<span>       values=tensor([[1., 2., 3.],</span>
<span>                      [4., 5., 6.]]), size=(2, 2, 2), nnz=3,</span>
<span>       layout=torch.sparse_csr)</span>
```

Dense dimensions: On the other hand, some data such as Graph embeddings might be better viewed as sparse collections of vectors instead of scalars.

In this example we create a 3D Hybrid COO Tensor with 2 sparse and 1 dense dimension from a 3D strided Tensor. If an entire row in the 3D strided Tensor is zero, it is not stored. If however any of the values in the row are non-zero, they are stored entirely. This reduces the number of indices since we need one index one per row instead of one per element. But it also increases the amount of storage for the values. Since only rows that are _entirely_ zero can be emitted and the presence of any non-zero valued elements cause the entire row to be stored.

```
<span></span><span>&gt;&gt;&gt; </span><span>t</span> <span>=</span> <span>torch</span><span>.</span><span>tensor</span><span>([[[</span><span>0.</span><span>,</span> <span>0</span><span>],</span> <span>[</span><span>1.</span><span>,</span> <span>2.</span><span>]],</span> <span>[[</span><span>0.</span><span>,</span> <span>0</span><span>],</span> <span>[</span><span>3.</span><span>,</span> <span>4.</span><span>]]])</span>
<span>&gt;&gt;&gt; </span><span>t</span><span>.</span><span>to_sparse</span><span>(</span><span>sparse_dim</span><span>=</span><span>2</span><span>)</span>
<span>tensor(indices=tensor([[0, 1],</span>
<span>                       [1, 1]]),</span>
<span>       values=tensor([[1., 2.],</span>
<span>                      [3., 4.]]),</span>
<span>       size=(2, 2, 2), nnz=2, layout=torch.sparse_coo)</span>
```

## Operator overview

Fundamentally, operations on Tensor with sparse storage formats behave the same as operations on Tensor with strided (or other) storage formats. The particularities of storage, that is the physical layout of the data, influences the performance of an operation but should not influence the semantics.

We are actively increasing operator coverage for sparse tensors. Users should not expect support same level of support as for dense Tensors yet. See our [operator](https://docs.pytorch.org/docs/stable/sparse.html#sparse-ops-docs) documentation for a list.

```
<span></span><span>&gt;&gt;&gt; </span><span>b</span> <span>=</span> <span>torch</span><span>.</span><span>tensor</span><span>([[</span><span>0</span><span>,</span> <span>0</span><span>,</span> <span>1</span><span>,</span> <span>2</span><span>,</span> <span>3</span><span>,</span> <span>0</span><span>],</span> <span>[</span><span>4</span><span>,</span> <span>5</span><span>,</span> <span>0</span><span>,</span> <span>6</span><span>,</span> <span>0</span><span>,</span> <span>0</span><span>]])</span>
<span>&gt;&gt;&gt; </span><span>b_s</span> <span>=</span> <span>b</span><span>.</span><span>to_sparse_csr</span><span>()</span>
<span>&gt;&gt;&gt; </span><span>b_s</span><span>.</span><span>cos</span><span>()</span>
<span>Traceback (most recent call last):</span>
  File <span>"&lt;stdin&gt;"</span>, line <span>1</span>, in <span>&lt;module&gt;</span>
<span>RuntimeError</span>: <span>unsupported tensor layout: SparseCsr</span>
<span>&gt;&gt;&gt; </span><span>b_s</span><span>.</span><span>sin</span><span>()</span>
<span>tensor(crow_indices=tensor([0, 3, 6]),</span>
<span>       col_indices=tensor([2, 3, 4, 0, 1, 3]),</span>
<span>       values=tensor([ 0.8415,  0.9093,  0.1411, -0.7568, -0.9589, -0.2794]),</span>
<span>       size=(2, 6), nnz=6, layout=torch.sparse_csr)</span>
```

As shown in the example above, we don’t support non-zero preserving unary operators such as cos. The output of a non-zero preserving unary operation will not be able to take advantage of sparse storage formats to the same extent as the input and potentially result in a catastrophic increase in memory. We instead rely on the user to explicitly convert to a dense Tensor first and then run the operation.

```
<span></span><span>&gt;&gt;&gt; </span><span>b_s</span><span>.</span><span>to_dense</span><span>()</span><span>.</span><span>cos</span><span>()</span>
<span>tensor([[ 1.0000, -0.4161],</span>
<span>        [-0.9900,  1.0000]])</span>
```

We are aware that some users want to ignore compressed zeros for operations such as cos instead of preserving the exact semantics of the operation. For this we can point to torch.masked and its MaskedTensor, which is in turn also backed and powered by sparse storage formats and kernels.

Also note that, for now, the user doesn’t have a choice of the output layout. For example, adding a sparse Tensor to a regular strided Tensor results in a strided Tensor. Some users might prefer for this to stay a sparse layout, because they know the result will still be sufficiently sparse.

```
<span></span><span>&gt;&gt;&gt; </span><span>a</span> <span>+</span> <span>b</span><span>.</span><span>to_sparse</span><span>()</span>
<span>tensor([[0., 3.],</span>
<span>        [3., 0.]])</span>
```

We acknowledge that access to kernels that can efficiently produce different output layouts can be very useful. A subsequent operation might significantly benefit from receiving a particular layout. We are working on an API to control the result layout and recognize it is an important feature to plan a more optimal path of execution for any given model.

## Sparse Semi-Structured Tensors

Warning

Sparse semi-structured tensors are currently a prototype feature and subject to change. Please feel free to open an issue to report a bug or if you have feedback to share.

Semi-Structured sparsity is a sparse data layout that was first introduced in NVIDIA’s Ampere architecture. It is also referred to as **fine-grained structured sparsity** or **2:4 structured sparsity**.

This sparse layout stores n elements out of every 2n elements, with n being determined by the width of the Tensor’s data type (dtype). The most frequently used dtype is float16, where n=2, thus the term “2:4 structured sparsity.”

Semi-structured sparsity is explained in greater detail in [this NVIDIA blog post](https://developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt).

In PyTorch, semi-structured sparsity is implemented via a Tensor subclass. By subclassing, we can override `__torch_dispatch__` , allowing us to use faster sparse kernels when performing matrix multiplication. We can also store the tensor in it’s compressed form inside the subclass to reduce memory overhead.

In this compressed form, the sparse tensor is stored by retaining only the _specified_ elements and some metadata, which encodes the mask.

Note

The specified elements and metadata mask of a semi-structured sparse tensor are stored together in a single flat compressed tensor. They are appended to each other to form a contiguous chunk of memory.

compressed tensor = \[ specified elements of original tensor | metadata\_mask \]

For an original tensor of size (r, c) we expect the first m \* k // 2 elements to be the kept elements and the rest of the tensor is metadata.

In order to make it easier for the user to view the specified elements and mask, one can use `.indices()` and `.values()` to access the mask and specified elements respectively.

-   `.values()` returns the specified elements in a tensor of size (r, c//2) and with the same dtype as the dense matrix.
    
-   `.indices()` returns the metadata\_mask in a tensor of size (r, c//2 ) and with element type `torch.int16` if dtype is torch.float16 or torch.bfloat16, and element type `torch.int32` if dtype is torch.int8.
    

For 2:4 sparse tensors, the metadata overhead is minor - just 2 bits per specified element.

Note

It’s important to note that `torch.float32` is only supported for 1:2 sparsity. Therefore, it does not follow the same formula as above.

Here, we break down how to calculate the compression ratio ( size dense / size sparse) of a 2:4 sparse tensor.

Let (r, c) = tensor.shape and e = bitwidth(tensor.dtype), so e = 16 for `torch.float16` and `torch.bfloat16` and e = 8 for `torch.int8`.

$M_{dense} = r \times c \times e \\ M_{sparse} = M_{specified} + M_{metadata} = r \times \frac{c}{2} \times e + r \times \frac{c}{2} \times 2 = \frac{rce}{2} + rc =rce(\frac{1}{2} +\frac{1}{e})$

Using these calculations, we can determine the total memory footprint for both the original dense and the new sparse representation.

This gives us a simple formula for the compression ratio, which is dependent only on the bitwidth of the tensor datatype.

$C = \frac{M_{sparse}}{M_{dense}} =  \frac{1}{2} + \frac{1}{e}$

By using this formula, we find that the compression ratio is 56.25% for `torch.float16` or `torch.bfloat16`, and 62.5% for `torch.int8`.

### Constructing Sparse Semi-Structured Tensors

You can transform a dense tensor into a sparse semi-structured tensor by simply using the `torch.to_sparse_semi_structured` function.

Please also note that we only support CUDA tensors since hardware compatibility for semi-structured sparsity is limited to NVIDIA GPUs.

The following datatypes are supported for semi-structured sparsity. Note that each datatype has its own shape constraints and compression factor.

To construct a semi-structured sparse tensor, start by creating a regular dense tensor that adheres to a 2:4 (or semi-structured) sparse format. To do this we tile a small 1x4 strip to create a 16x16 dense float16 tensor. Afterwards, we can call `to_sparse_semi_structured` function to compress it for accelerated inference.

```
<span></span><span>&gt;&gt;&gt; </span><span>from</span> <span>torch.sparse</span> <span>import</span> <span>to_sparse_semi_structured</span>
<span>&gt;&gt;&gt; </span><span>A</span> <span>=</span> <span>torch</span><span>.</span><span>Tensor</span><span>([</span><span>0</span><span>,</span> <span>0</span><span>,</span> <span>1</span><span>,</span> <span>1</span><span>])</span><span>.</span><span>tile</span><span>((</span><span>128</span><span>,</span> <span>32</span><span>))</span><span>.</span><span>half</span><span>()</span><span>.</span><span>cuda</span><span>()</span>
<span>tensor([[0., 0., 1.,  ..., 0., 1., 1.],</span>
<span>        [0., 0., 1.,  ..., 0., 1., 1.],</span>
<span>        [0., 0., 1.,  ..., 0., 1., 1.],</span>
<span>        ...,</span>
<span>        [0., 0., 1.,  ..., 0., 1., 1.],</span>
<span>        [0., 0., 1.,  ..., 0., 1., 1.],</span>
<span>        [0., 0., 1.,  ..., 0., 1., 1.]], device='cuda:0', dtype=torch.float16)</span>
<span>&gt;&gt;&gt; </span><span>A_sparse</span> <span>=</span> <span>to_sparse_semi_structured</span><span>(</span><span>A</span><span>)</span>
<span>SparseSemiStructuredTensor(shape=torch.Size([128, 128]), transposed=False, values=tensor([[1., 1., 1.,  ..., 1., 1., 1.],</span>
<span>        [1., 1., 1.,  ..., 1., 1., 1.],</span>
<span>        [1., 1., 1.,  ..., 1., 1., 1.],</span>
<span>        ...,</span>
<span>        [1., 1., 1.,  ..., 1., 1., 1.],</span>
<span>        [1., 1., 1.,  ..., 1., 1., 1.],</span>
<span>        [1., 1., 1.,  ..., 1., 1., 1.]], device='cuda:0', dtype=torch.float16), metadata=tensor([[-4370, -4370, -4370,  ..., -4370, -4370, -4370],</span>
<span>        [-4370, -4370, -4370,  ..., -4370, -4370, -4370],</span>
<span>        [-4370, -4370, -4370,  ..., -4370, -4370, -4370],</span>
<span>        ...,</span>
<span>        [-4370, -4370, -4370,  ..., -4370, -4370, -4370],</span>
<span>        [-4370, -4370, -4370,  ..., -4370, -4370, -4370],</span>
<span>        [-4370, -4370, -4370,  ..., -4370, -4370, -4370]], device='cuda:0',</span>
<span>dtype=torch.int16))</span>
```

### Sparse Semi-Structured Tensor Operations

Currently, the following operations are supported for semi-structured sparse tensors:

-   torch.addmm(bias, dense, sparse.t())
    
-   torch.mm(dense, sparse)
    
-   torch.mm(sparse, dense)
    
-   aten.linear.default(dense, sparse, bias)
    
-   aten.t.default(sparse)
    
-   aten.t.detach(sparse)
    

To use these ops, simply pass the output of `to_sparse_semi_structured(tensor)` instead of using `tensor` once your tensor has 0s in a semi-structured sparse format, like this:

```
<span></span><span>&gt;&gt;&gt; </span><span>a</span> <span>=</span> <span>torch</span><span>.</span><span>Tensor</span><span>([</span><span>0</span><span>,</span> <span>0</span><span>,</span> <span>1</span><span>,</span> <span>1</span><span>])</span><span>.</span><span>tile</span><span>((</span><span>64</span><span>,</span> <span>16</span><span>))</span><span>.</span><span>half</span><span>()</span><span>.</span><span>cuda</span><span>()</span>
<span>&gt;&gt;&gt; </span><span>b</span> <span>=</span> <span>torch</span><span>.</span><span>rand</span><span>(</span><span>64</span><span>,</span> <span>64</span><span>)</span><span>.</span><span>half</span><span>()</span><span>.</span><span>cuda</span><span>()</span>
<span>&gt;&gt;&gt; </span><span>c</span> <span>=</span> <span>torch</span><span>.</span><span>mm</span><span>(</span><span>a</span><span>,</span> <span>b</span><span>)</span>
<span>&gt;&gt;&gt; </span><span>a_sparse</span> <span>=</span> <span>to_sparse_semi_structured</span><span>(</span><span>a</span><span>)</span>
<span>&gt;&gt;&gt; </span><span>torch</span><span>.</span><span>allclose</span><span>(</span><span>c</span><span>,</span> <span>torch</span><span>.</span><span>mm</span><span>(</span><span>a_sparse</span><span>,</span> <span>b</span><span>))</span>
<span>True</span>
```

### Accelerating nn.Linear with semi-structured sparsity

You can accelerate the linear layers in your model if the weights are already semi-structured sparse with just a few lines of code:

```
<span></span><span>&gt;&gt;&gt; </span><span>input</span> <span>=</span> <span>torch</span><span>.</span><span>rand</span><span>(</span><span>64</span><span>,</span> <span>64</span><span>)</span><span>.</span><span>half</span><span>()</span><span>.</span><span>cuda</span><span>()</span>
<span>&gt;&gt;&gt; </span><span>mask</span> <span>=</span> <span>torch</span><span>.</span><span>Tensor</span><span>([</span><span>0</span><span>,</span> <span>0</span><span>,</span> <span>1</span><span>,</span> <span>1</span><span>])</span><span>.</span><span>tile</span><span>((</span><span>64</span><span>,</span> <span>16</span><span>))</span><span>.</span><span>cuda</span><span>()</span><span>.</span><span>bool</span><span>()</span>
<span>&gt;&gt;&gt; </span><span>linear</span> <span>=</span> <span>nn</span><span>.</span><span>Linear</span><span>(</span><span>64</span><span>,</span> <span>64</span><span>)</span><span>.</span><span>half</span><span>()</span><span>.</span><span>cuda</span><span>()</span>
<span>&gt;&gt;&gt; </span><span>linear</span><span>.</span><span>weight</span> <span>=</span> <span>nn</span><span>.</span><span>Parameter</span><span>(</span><span>to_sparse_semi_structured</span><span>(</span><span>linear</span><span>.</span><span>weight</span><span>.</span><span>masked_fill</span><span>(</span><span>~</span><span>mask</span><span>,</span> <span>0</span><span>)))</span>
```

## Sparse COO tensors

PyTorch implements the so-called Coordinate format, or COO format, as one of the storage formats for implementing sparse tensors. In COO format, the specified elements are stored as tuples of element indices and the corresponding values. In particular,

> -   the indices of specified elements are collected in `indices` tensor of size `(ndim, nse)` and with element type `torch.int64`,
>     
> -   the corresponding values are collected in `values` tensor of size `(nse,)` and with an arbitrary integer or floating point number element type,
>     

where `ndim` is the dimensionality of the tensor and `nse` is the number of specified elements.

Note

The memory consumption of a sparse COO tensor is at least `(ndim * 8 + <size of element type in bytes>) * nse` bytes (plus a constant overhead from storing other tensor data).

The memory consumption of a strided tensor is at least `product(<tensor shape>) * <size of element type in bytes>`.

For example, the memory consumption of a 10 000 x 10 000 tensor with 100 000 non-zero 32-bit floating point numbers is at least `(2 * 8 + 4) * 100 000 = 2 000 000` bytes when using COO tensor layout and `10 000 * 10 000 * 4 = 400 000 000` bytes when using the default strided tensor layout. Notice the 200 fold memory saving from using the COO storage format.

### Construction

A sparse COO tensor can be constructed by providing the two tensors of indices and values, as well as the size of the sparse tensor (when it cannot be inferred from the indices and values tensors) to a function [`torch.sparse_coo_tensor()`](https://docs.pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html#torch.sparse_coo_tensor "torch.sparse_coo_tensor").

Suppose we want to define a sparse tensor with the entry 3 at location (0, 2), entry 4 at location (1, 0), and entry 5 at location (1, 2). Unspecified elements are assumed to have the same value, fill value, which is zero by default. We would then write:

```
<span></span><span>&gt;&gt;&gt; </span><span>i</span> <span>=</span> <span>[[</span><span>0</span><span>,</span> <span>1</span><span>,</span> <span>1</span><span>],</span> <span>[</span><span>2</span><span>,</span> <span>0</span><span>,</span> <span>2</span><span>]]</span>
<span>&gt;&gt;&gt; </span><span>v</span> <span>=</span>  <span>[</span><span>3</span><span>,</span> <span>4</span><span>,</span> <span>5</span><span>]</span>
<span>&gt;&gt;&gt; </span><span>s</span> <span>=</span> <span>torch</span><span>.</span><span>sparse_coo_tensor</span><span>(</span><span>i</span><span>,</span> <span>v</span><span>,</span> <span>(</span><span>2</span><span>,</span> <span>3</span><span>))</span>
<span>&gt;&gt;&gt; </span><span>s</span>
<span>tensor(indices=tensor([[0, 1, 1],</span>
<span>                       [2, 0, 2]]),</span>
<span>       values=tensor([3, 4, 5]),</span>
<span>       size=(2, 3), nnz=3, layout=torch.sparse_coo)</span>
<span>&gt;&gt;&gt; </span><span>s</span><span>.</span><span>to_dense</span><span>()</span>
<span>tensor([[0, 0, 3],</span>
<span>        [4, 0, 5]])</span>
```

Note that the input `i` is NOT a list of index tuples. If you want to write your indices this way, you should transpose before passing them to the sparse constructor:

```
<span></span><span>&gt;&gt;&gt; </span><span>i</span> <span>=</span> <span>[[</span><span>0</span><span>,</span> <span>2</span><span>],</span> <span>[</span><span>1</span><span>,</span> <span>0</span><span>],</span> <span>[</span><span>1</span><span>,</span> <span>2</span><span>]]</span>
<span>&gt;&gt;&gt; </span><span>v</span> <span>=</span>  <span>[</span><span>3</span><span>,</span>      <span>4</span><span>,</span>      <span>5</span>    <span>]</span>
<span>&gt;&gt;&gt; </span><span>s</span> <span>=</span> <span>torch</span><span>.</span><span>sparse_coo_tensor</span><span>(</span><span>list</span><span>(</span><span>zip</span><span>(</span><span>*</span><span>i</span><span>)),</span> <span>v</span><span>,</span> <span>(</span><span>2</span><span>,</span> <span>3</span><span>))</span>
<span>&gt;&gt;&gt; </span><span># Or another equivalent formulation to get s</span>
<span>&gt;&gt;&gt; </span><span>s</span> <span>=</span> <span>torch</span><span>.</span><span>sparse_coo_tensor</span><span>(</span><span>torch</span><span>.</span><span>tensor</span><span>(</span><span>i</span><span>)</span><span>.</span><span>t</span><span>(),</span> <span>v</span><span>,</span> <span>(</span><span>2</span><span>,</span> <span>3</span><span>))</span>
<span>&gt;&gt;&gt; </span><span>torch</span><span>.</span><span>sparse_coo_tensor</span><span>(</span><span>i</span><span>.</span><span>t</span><span>(),</span> <span>v</span><span>,</span> <span>torch</span><span>.</span><span>Size</span><span>([</span><span>2</span><span>,</span><span>3</span><span>]))</span><span>.</span><span>to_dense</span><span>()</span>
<span>tensor([[0, 0, 3],</span>
<span>        [4, 0, 5]])</span>
```

An empty sparse COO tensor can be constructed by specifying its size only:

```
<span></span><span>&gt;&gt;&gt; </span><span>torch</span><span>.</span><span>sparse_coo_tensor</span><span>(</span><span>size</span><span>=</span><span>(</span><span>2</span><span>,</span> <span>3</span><span>))</span>
<span>tensor(indices=tensor([], size=(2, 0)),</span>
<span>       values=tensor([], size=(0,)),</span>
<span>       size=(2, 3), nnz=0, layout=torch.sparse_coo)</span>
```

### Sparse hybrid COO tensors

PyTorch implements an extension of sparse tensors with scalar values to sparse tensors with (contiguous) tensor values. Such tensors are called hybrid tensors.

PyTorch hybrid COO tensor extends the sparse COO tensor by allowing the `values` tensor to be a multi-dimensional tensor so that we have:

> -   the indices of specified elements are collected in `indices` tensor of size `(sparse_dims, nse)` and with element type `torch.int64`,
>     
> -   the corresponding (tensor) values are collected in `values` tensor of size `(nse, dense_dims)` and with an arbitrary integer or floating point number element type.
>     

Note

We use (M + K)-dimensional tensor to denote a N-dimensional sparse hybrid tensor, where M and K are the numbers of sparse and dense dimensions, respectively, such that M + K == N holds.

Suppose we want to create a (2 + 1)-dimensional tensor with the entry \[3, 4\] at location (0, 2), entry \[5, 6\] at location (1, 0), and entry \[7, 8\] at location (1, 2). We would write

```
<span></span><span>&gt;&gt;&gt; </span><span>i</span> <span>=</span> <span>[[</span><span>0</span><span>,</span> <span>1</span><span>,</span> <span>1</span><span>],</span>
<span>         [2, 0, 2]]</span>
<span>&gt;&gt;&gt; </span><span>v</span> <span>=</span>  <span>[[</span><span>3</span><span>,</span> <span>4</span><span>],</span> <span>[</span><span>5</span><span>,</span> <span>6</span><span>],</span> <span>[</span><span>7</span><span>,</span> <span>8</span><span>]]</span>
<span>&gt;&gt;&gt; </span><span>s</span> <span>=</span> <span>torch</span><span>.</span><span>sparse_coo_tensor</span><span>(</span><span>i</span><span>,</span> <span>v</span><span>,</span> <span>(</span><span>2</span><span>,</span> <span>3</span><span>,</span> <span>2</span><span>))</span>
<span>&gt;&gt;&gt; </span><span>s</span>
<span>tensor(indices=tensor([[0, 1, 1],</span>
<span>                       [2, 0, 2]]),</span>
<span>       values=tensor([[3, 4],</span>
<span>                      [5, 6],</span>
<span>                      [7, 8]]),</span>
<span>       size=(2, 3, 2), nnz=3, layout=torch.sparse_coo)</span>
```

```
<span></span><span>&gt;&gt;&gt; </span><span>s</span><span>.</span><span>to_dense</span><span>()</span>
<span>tensor([[[0, 0],</span>
<span>         [0, 0],</span>
<span>         [3, 4]],</span>
<span>        [[5, 6],</span>
<span>         [0, 0],</span>
<span>         [7, 8]]])</span>
```

In general, if `s` is a sparse COO tensor and `M = s.sparse_dim()`, `K = s.dense_dim()`, then we have the following invariants:

> -   `M + K == len(s.shape) == s.ndim` - dimensionality of a tensor is the sum of the number of sparse and dense dimensions,
>     
> -   `s.indices().shape == (M, nse)` - sparse indices are stored explicitly,
>     
> -   `s.values().shape == (nse,) + s.shape[M : M + K]` - the values of a hybrid tensor are K-dimensional tensors,
>     
> -   `s.values().layout == torch.strided` - values are stored as strided tensors.
>     

Note

Dense dimensions always follow sparse dimensions, that is, mixing of dense and sparse dimensions is not supported.

Note

To be sure that a constructed sparse tensor has consistent indices, values, and size, the invariant checks can be enabled per tensor creation via `check_invariants=True` keyword argument, or globally using [`torch.sparse.check_sparse_tensor_invariants`](https://docs.pytorch.org/docs/stable/generated/torch.sparse.check_sparse_tensor_invariants.html#torch.sparse.check_sparse_tensor_invariants "torch.sparse.check_sparse_tensor_invariants") context manager instance. By default, the sparse tensor invariants checks are disabled.

### Uncoalesced sparse COO tensors

PyTorch sparse COO tensor format permits sparse _uncoalesced_ tensors, where there may be duplicate coordinates in the indices; in this case, the interpretation is that the value at that index is the sum of all duplicate value entries. For example, one can specify multiple values, `3` and `4`, for the same index `1`, that leads to an 1-D uncoalesced tensor:

```
<span></span><span>&gt;&gt;&gt; </span><span>i</span> <span>=</span> <span>[[</span><span>1</span><span>,</span> <span>1</span><span>]]</span>
<span>&gt;&gt;&gt; </span><span>v</span> <span>=</span>  <span>[</span><span>3</span><span>,</span> <span>4</span><span>]</span>
<span>&gt;&gt;&gt; </span><span>s</span><span>=</span><span>torch</span><span>.</span><span>sparse_coo_tensor</span><span>(</span><span>i</span><span>,</span> <span>v</span><span>,</span> <span>(</span><span>3</span><span>,))</span>
<span>&gt;&gt;&gt; </span><span>s</span>
<span>tensor(indices=tensor([[1, 1]]),</span>
<span>       values=tensor(  [3, 4]),</span>
<span>       size=(3,), nnz=2, layout=torch.sparse_coo)</span>
```

while the coalescing process will accumulate the multi-valued elements into a single value using summation:

```
<span></span><span>&gt;&gt;&gt; </span><span>s</span><span>.</span><span>coalesce</span><span>()</span>
<span>tensor(indices=tensor([[1]]),</span>
<span>       values=tensor([7]),</span>
<span>       size=(3,), nnz=1, layout=torch.sparse_coo)</span>
```

In general, the output of [`torch.Tensor.coalesce()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.coalesce.html#torch.Tensor.coalesce "torch.Tensor.coalesce") method is a sparse tensor with the following properties:

-   the indices of specified tensor elements are unique,
    
-   the indices are sorted in lexicographical order,
    
-   [`torch.Tensor.is_coalesced()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.is_coalesced.html#torch.Tensor.is_coalesced "torch.Tensor.is_coalesced") returns `True`.
    

Note

For the most part, you shouldn’t have to care whether or not a sparse tensor is coalesced or not, as most operations will work identically given a sparse coalesced or uncoalesced tensor.

However, some operations can be implemented more efficiently on uncoalesced tensors, and some on coalesced tensors.

For instance, addition of sparse COO tensors is implemented by simply concatenating the indices and values tensors:

```
<span></span><span>&gt;&gt;&gt; </span><span>a</span> <span>=</span> <span>torch</span><span>.</span><span>sparse_coo_tensor</span><span>([[</span><span>1</span><span>,</span> <span>1</span><span>]],</span> <span>[</span><span>5</span><span>,</span> <span>6</span><span>],</span> <span>(</span><span>2</span><span>,))</span>
<span>&gt;&gt;&gt; </span><span>b</span> <span>=</span> <span>torch</span><span>.</span><span>sparse_coo_tensor</span><span>([[</span><span>0</span><span>,</span> <span>0</span><span>]],</span> <span>[</span><span>7</span><span>,</span> <span>8</span><span>],</span> <span>(</span><span>2</span><span>,))</span>
<span>&gt;&gt;&gt; </span><span>a</span> <span>+</span> <span>b</span>
<span>tensor(indices=tensor([[0, 0, 1, 1]]),</span>
<span>       values=tensor([7, 8, 5, 6]),</span>
<span>       size=(2,), nnz=4, layout=torch.sparse_coo)</span>
```

If you repeatedly perform an operation that can produce duplicate entries (e.g., [`torch.Tensor.add()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.add.html#torch.Tensor.add "torch.Tensor.add")), you should occasionally coalesce your sparse tensors to prevent them from growing too large.

On the other hand, the lexicographical ordering of indices can be advantageous for implementing algorithms that involve many element selection operations, such as slicing or matrix products.

### Working with sparse COO tensors

Let’s consider the following example:

```
<span></span><span>&gt;&gt;&gt; </span><span>i</span> <span>=</span> <span>[[</span><span>0</span><span>,</span> <span>1</span><span>,</span> <span>1</span><span>],</span>
<span>         [2, 0, 2]]</span>
<span>&gt;&gt;&gt; </span><span>v</span> <span>=</span>  <span>[[</span><span>3</span><span>,</span> <span>4</span><span>],</span> <span>[</span><span>5</span><span>,</span> <span>6</span><span>],</span> <span>[</span><span>7</span><span>,</span> <span>8</span><span>]]</span>
<span>&gt;&gt;&gt; </span><span>s</span> <span>=</span> <span>torch</span><span>.</span><span>sparse_coo_tensor</span><span>(</span><span>i</span><span>,</span> <span>v</span><span>,</span> <span>(</span><span>2</span><span>,</span> <span>3</span><span>,</span> <span>2</span><span>))</span>
```

As mentioned above, a sparse COO tensor is a [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor") instance and to distinguish it from the Tensor instances that use some other layout, one can use [`torch.Tensor.is_sparse`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.is_sparse.html#torch.Tensor.is_sparse "torch.Tensor.is_sparse") or `torch.Tensor.layout` properties:

```
<span></span><span>&gt;&gt;&gt; </span><span>isinstance</span><span>(</span><span>s</span><span>,</span> <span>torch</span><span>.</span><span>Tensor</span><span>)</span>
<span>True</span>
<span>&gt;&gt;&gt; </span><span>s</span><span>.</span><span>is_sparse</span>
<span>True</span>
<span>&gt;&gt;&gt; </span><span>s</span><span>.</span><span>layout</span> <span>==</span> <span>torch</span><span>.</span><span>sparse_coo</span>
<span>True</span>
```

The number of sparse and dense dimensions can be acquired using methods [`torch.Tensor.sparse_dim()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.sparse_dim.html#torch.Tensor.sparse_dim "torch.Tensor.sparse_dim") and [`torch.Tensor.dense_dim()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.dense_dim.html#torch.Tensor.dense_dim "torch.Tensor.dense_dim"), respectively. For instance:

```
<span></span><span>&gt;&gt;&gt; </span><span>s</span><span>.</span><span>sparse_dim</span><span>(),</span> <span>s</span><span>.</span><span>dense_dim</span><span>()</span>
<span>(2, 1)</span>
```

If `s` is a sparse COO tensor then its COO format data can be acquired using methods [`torch.Tensor.indices()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.indices.html#torch.Tensor.indices "torch.Tensor.indices") and [`torch.Tensor.values()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.values.html#torch.Tensor.values "torch.Tensor.values").

Note

Currently, one can acquire the COO format data only when the tensor instance is coalesced:

```
<span></span><span>&gt;&gt;&gt; </span><span>s</span><span>.</span><span>indices</span><span>()</span>
<span>RuntimeError: Cannot get indices on an uncoalesced tensor, please call .coalesce() first</span>
```

For acquiring the COO format data of an uncoalesced tensor, use `torch.Tensor._values()` and `torch.Tensor._indices()`:

```
<span></span><span>&gt;&gt;&gt; </span><span>s</span><span>.</span><span>_indices</span><span>()</span>
<span>tensor([[0, 1, 1],</span>
<span>        [2, 0, 2]])</span>
```

Warning

Calling `torch.Tensor._values()` will return a _detached_ tensor. To track gradients, `torch.Tensor.coalesce().values()` must be used instead.

Constructing a new sparse COO tensor results a tensor that is not coalesced:

```
<span></span><span>&gt;&gt;&gt; </span><span>s</span><span>.</span><span>is_coalesced</span><span>()</span>
<span>False</span>
```

but one can construct a coalesced copy of a sparse COO tensor using the [`torch.Tensor.coalesce()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.coalesce.html#torch.Tensor.coalesce "torch.Tensor.coalesce") method:

```
<span></span><span>&gt;&gt;&gt; </span><span>s2</span> <span>=</span> <span>s</span><span>.</span><span>coalesce</span><span>()</span>
<span>&gt;&gt;&gt; </span><span>s2</span><span>.</span><span>indices</span><span>()</span>
<span>tensor([[0, 1, 1],</span>
<span>       [2, 0, 2]])</span>
```

When working with uncoalesced sparse COO tensors, one must take into an account the additive nature of uncoalesced data: the values of the same indices are the terms of a sum that evaluation gives the value of the corresponding tensor element. For example, the scalar multiplication on a sparse uncoalesced tensor could be implemented by multiplying all the uncoalesced values with the scalar because `c * (a + b) == c * a + c * b` holds. However, any nonlinear operation, say, a square root, cannot be implemented by applying the operation to uncoalesced data because `sqrt(a + b) == sqrt(a) + sqrt(b)` does not hold in general.

Slicing (with positive step) of a sparse COO tensor is supported only for dense dimensions. Indexing is supported for both sparse and dense dimensions:

```
<span></span><span>&gt;&gt;&gt; </span><span>s</span><span>[</span><span>1</span><span>]</span>
<span>tensor(indices=tensor([[0, 2]]),</span>
<span>       values=tensor([[5, 6],</span>
<span>                      [7, 8]]),</span>
<span>       size=(3, 2), nnz=2, layout=torch.sparse_coo)</span>
<span>&gt;&gt;&gt; </span><span>s</span><span>[</span><span>1</span><span>,</span> <span>0</span><span>,</span> <span>1</span><span>]</span>
<span>tensor(6)</span>
<span>&gt;&gt;&gt; </span><span>s</span><span>[</span><span>1</span><span>,</span> <span>0</span><span>,</span> <span>1</span><span>:]</span>
<span>tensor([6])</span>
```

In PyTorch, the fill value of a sparse tensor cannot be specified explicitly and is assumed to be zero in general. However, there exists operations that may interpret the fill value differently. For instance, [`torch.sparse.softmax()`](https://docs.pytorch.org/docs/stable/generated/torch.sparse.softmax.html#torch.sparse.softmax "torch.sparse.softmax") computes the softmax with the assumption that the fill value is negative infinity.

## Sparse Compressed Tensors

Sparse Compressed Tensors represents a class of sparse tensors that have a common feature of compressing the indices of a certain dimension using an encoding that enables certain optimizations on linear algebra kernels of sparse compressed tensors. This encoding is based on the [Compressed Sparse Row (CSR)](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) format that PyTorch sparse compressed tensors extend with the support of sparse tensor batches, allowing multi-dimensional tensor values, and storing sparse tensor values in dense blocks.

Note

We use (B + M + K)-dimensional tensor to denote a N-dimensional sparse compressed hybrid tensor, where B, M, and K are the numbers of batch, sparse, and dense dimensions, respectively, such that `B + M + K == N` holds. The number of sparse dimensions for sparse compressed tensors is always two, `M == 2`.

Note

We say that an indices tensor `compressed_indices` uses CSR encoding if the following invariants are satisfied:

-   `compressed_indices` is a contiguous strided 32 or 64 bit integer tensor
    
-   `compressed_indices` shape is `(*batchsize, compressed_dim_size + 1)` where `compressed_dim_size` is the number of compressed dimensions (e.g. rows or columns)
    
-   `compressed_indices[..., 0] == 0` where `...` denotes batch indices
    
-   `compressed_indices[..., compressed_dim_size] == nse` where `nse` is the number of specified elements
    
-   `0 <= compressed_indices[..., i] - compressed_indices[..., i - 1] <= plain_dim_size` for `i=1, ..., compressed_dim_size`, where `plain_dim_size` is the number of plain dimensions (orthogonal to compressed dimensions, e.g. columns or rows).
    

To be sure that a constructed sparse tensor has consistent indices, values, and size, the invariant checks can be enabled per tensor creation via `check_invariants=True` keyword argument, or globally using [`torch.sparse.check_sparse_tensor_invariants`](https://docs.pytorch.org/docs/stable/generated/torch.sparse.check_sparse_tensor_invariants.html#torch.sparse.check_sparse_tensor_invariants "torch.sparse.check_sparse_tensor_invariants") context manager instance. By default, the sparse tensor invariants checks are disabled.

Note

The generalization of sparse compressed layouts to N-dimensional tensors can lead to some confusion regarding the count of specified elements. When a sparse compressed tensor contains batch dimensions the number of specified elements will correspond to the number of such elements per-batch. When a sparse compressed tensor has dense dimensions the element considered is now the K-dimensional array. Also for block sparse compressed layouts the 2-D block is considered as the element being specified. Take as an example a 3-dimensional block sparse tensor, with one batch dimension of length `b`, and a block shape of `p, q`. If this tensor has `n` specified elements, then in fact we have `n` blocks specified per batch. This tensor would have `values` with shape `(b, n, p, q)`. This interpretation of the number of specified elements comes from all sparse compressed layouts being derived from the compression of a 2-dimensional matrix. Batch dimensions are treated as stacking of sparse matrices, dense dimensions change the meaning of the element from a simple scalar value to an array with its own dimensions.

### Sparse CSR Tensor

The primary advantage of the CSR format over the COO format is better use of storage and much faster computation operations such as sparse matrix-vector multiplication using MKL and MAGMA backends.

In the simplest case, a (0 + 2 + 0)-dimensional sparse CSR tensor consists of three 1-D tensors: `crow_indices`, `col_indices` and `values`:

> -   The `crow_indices` tensor consists of compressed row indices. This is a 1-D tensor of size `nrows + 1` (the number of rows plus 1). The last element of `crow_indices` is the number of specified elements, `nse`. This tensor encodes the index in `values` and `col_indices` depending on where the given row starts. Each successive number in the tensor subtracted by the number before it denotes the number of elements in a given row.
>     
> -   The `col_indices` tensor contains the column indices of each element. This is a 1-D tensor of size `nse`.
>     
> -   The `values` tensor contains the values of the CSR tensor elements. This is a 1-D tensor of size `nse`.
>     

Note

The index tensors `crow_indices` and `col_indices` should have element type either `torch.int64` (default) or `torch.int32`. If you want to use MKL-enabled matrix operations, use `torch.int32`. This is as a result of the default linking of pytorch being with MKL LP64, which uses 32 bit integer indexing.

In the general case, the (B + 2 + K)-dimensional sparse CSR tensor consists of two (B + 1)-dimensional index tensors `crow_indices` and `col_indices`, and of (1 + K)-dimensional `values` tensor such that

> -   `crow_indices.shape == (*batchsize, nrows + 1)`
>     
> -   `col_indices.shape == (*batchsize, nse)`
>     
> -   `values.shape == (nse, *densesize)`
>     

while the shape of the sparse CSR tensor is `(*batchsize, nrows, ncols, *densesize)` where `len(batchsize) == B` and `len(densesize) == K`.

Note

The batches of sparse CSR tensors are dependent: the number of specified elements in all batches must be the same. This somewhat artificial constraint allows efficient storage of the indices of different CSR batches.

Note

The number of sparse and dense dimensions can be acquired using [`torch.Tensor.sparse_dim()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.sparse_dim.html#torch.Tensor.sparse_dim "torch.Tensor.sparse_dim") and [`torch.Tensor.dense_dim()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.dense_dim.html#torch.Tensor.dense_dim "torch.Tensor.dense_dim") methods. The batch dimensions can be computed from the tensor shape: `batchsize = tensor.shape[:-tensor.sparse_dim() - tensor.dense_dim()]`.

Note

The memory consumption of a sparse CSR tensor is at least `(nrows * 8 + (8 + <size of element type in bytes> * prod(densesize)) * nse) * prod(batchsize)` bytes (plus a constant overhead from storing other tensor data).

With the same example data of [the note in sparse COO format introduction](https://docs.pytorch.org/docs/stable/sparse.html#sparse-coo-docs), the memory consumption of a 10 000 x 10 000 tensor with 100 000 non-zero 32-bit floating point numbers is at least `(10000 * 8 + (8 + 4 * 1) * 100 000) * 1 = 1 280 000` bytes when using CSR tensor layout. Notice the 1.6 and 310 fold savings from using CSR storage format compared to using the COO and strided formats, respectively.

#### Construction of CSR tensors

Sparse CSR tensors can be directly constructed by using the [`torch.sparse_csr_tensor()`](https://docs.pytorch.org/docs/stable/generated/torch.sparse_csr_tensor.html#torch.sparse_csr_tensor "torch.sparse_csr_tensor") function. The user must supply the row and column indices and values tensors separately where the row indices must be specified using the CSR compression encoding. The `size` argument is optional and will be deduced from the `crow_indices` and `col_indices` if it is not present.

```
<span></span><span>&gt;&gt;&gt; </span><span>crow_indices</span> <span>=</span> <span>torch</span><span>.</span><span>tensor</span><span>([</span><span>0</span><span>,</span> <span>2</span><span>,</span> <span>4</span><span>])</span>
<span>&gt;&gt;&gt; </span><span>col_indices</span> <span>=</span> <span>torch</span><span>.</span><span>tensor</span><span>([</span><span>0</span><span>,</span> <span>1</span><span>,</span> <span>0</span><span>,</span> <span>1</span><span>])</span>
<span>&gt;&gt;&gt; </span><span>values</span> <span>=</span> <span>torch</span><span>.</span><span>tensor</span><span>([</span><span>1</span><span>,</span> <span>2</span><span>,</span> <span>3</span><span>,</span> <span>4</span><span>])</span>
<span>&gt;&gt;&gt; </span><span>csr</span> <span>=</span> <span>torch</span><span>.</span><span>sparse_csr_tensor</span><span>(</span><span>crow_indices</span><span>,</span> <span>col_indices</span><span>,</span> <span>values</span><span>,</span> <span>dtype</span><span>=</span><span>torch</span><span>.</span><span>float64</span><span>)</span>
<span>&gt;&gt;&gt; </span><span>csr</span>
<span>tensor(crow_indices=tensor([0, 2, 4]),</span>
<span>       col_indices=tensor([0, 1, 0, 1]),</span>
<span>       values=tensor([1., 2., 3., 4.]), size=(2, 2), nnz=4,</span>
<span>       dtype=torch.float64)</span>
<span>&gt;&gt;&gt; </span><span>csr</span><span>.</span><span>to_dense</span><span>()</span>
<span>tensor([[1., 2.],</span>
<span>        [3., 4.]], dtype=torch.float64)</span>
```

Note

The values of sparse dimensions in deduced `size` is computed from the size of `crow_indices` and the maximal index value in `col_indices`. If the number of columns needs to be larger than in the deduced `size` then the `size` argument must be specified explicitly.

The simplest way of constructing a 2-D sparse CSR tensor from a strided or sparse COO tensor is to use [`torch.Tensor.to_sparse_csr()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to_sparse_csr.html#torch.Tensor.to_sparse_csr "torch.Tensor.to_sparse_csr") method. Any zeros in the (strided) tensor will be interpreted as missing values in the sparse tensor:

```
<span></span><span>&gt;&gt;&gt; </span><span>a</span> <span>=</span> <span>torch</span><span>.</span><span>tensor</span><span>([[</span><span>0</span><span>,</span> <span>0</span><span>,</span> <span>1</span><span>,</span> <span>0</span><span>],</span> <span>[</span><span>1</span><span>,</span> <span>2</span><span>,</span> <span>0</span><span>,</span> <span>0</span><span>],</span> <span>[</span><span>0</span><span>,</span> <span>0</span><span>,</span> <span>0</span><span>,</span> <span>0</span><span>]],</span> <span>dtype</span><span>=</span><span>torch</span><span>.</span><span>float64</span><span>)</span>
<span>&gt;&gt;&gt; </span><span>sp</span> <span>=</span> <span>a</span><span>.</span><span>to_sparse_csr</span><span>()</span>
<span>&gt;&gt;&gt; </span><span>sp</span>
<span>tensor(crow_indices=tensor([0, 1, 3, 3]),</span>
<span>      col_indices=tensor([2, 0, 1]),</span>
<span>      values=tensor([1., 1., 2.]), size=(3, 4), nnz=3, dtype=torch.float64)</span>
```

#### CSR Tensor Operations

The sparse matrix-vector multiplication can be performed with the `tensor.matmul()` method. This is currently the only math operation supported on CSR tensors.

```
<span></span><span>&gt;&gt;&gt; </span><span>vec</span> <span>=</span> <span>torch</span><span>.</span><span>randn</span><span>(</span><span>4</span><span>,</span> <span>1</span><span>,</span> <span>dtype</span><span>=</span><span>torch</span><span>.</span><span>float64</span><span>)</span>
<span>&gt;&gt;&gt; </span><span>sp</span><span>.</span><span>matmul</span><span>(</span><span>vec</span><span>)</span>
<span>tensor([[0.9078],</span>
<span>        [1.3180],</span>
<span>        [0.0000]], dtype=torch.float64)</span>
```

### Sparse CSC Tensor

The sparse CSC (Compressed Sparse Column) tensor format implements the CSC format for storage of 2 dimensional tensors with an extension to supporting batches of sparse CSC tensors and values being multi-dimensional tensors.

Note

Sparse CSC tensor is essentially a transpose of the sparse CSR tensor when the transposition is about swapping the sparse dimensions.

Similarly to [sparse CSR tensors](https://docs.pytorch.org/docs/stable/sparse.html#sparse-csr-docs), a sparse CSC tensor consists of three tensors: `ccol_indices`, `row_indices` and `values`:

> -   The `ccol_indices` tensor consists of compressed column indices. This is a (B + 1)-D tensor of shape `(*batchsize, ncols + 1)`. The last element is the number of specified elements, `nse`. This tensor encodes the index in `values` and `row_indices` depending on where the given column starts. Each successive number in the tensor subtracted by the number before it denotes the number of elements in a given column.
>     
> -   The `row_indices` tensor contains the row indices of each element. This is a (B + 1)-D tensor of shape `(*batchsize, nse)`.
>     
> -   The `values` tensor contains the values of the CSC tensor elements. This is a (1 + K)-D tensor of shape `(nse, *densesize)`.
>     

#### Construction of CSC tensors

Sparse CSC tensors can be directly constructed by using the [`torch.sparse_csc_tensor()`](https://docs.pytorch.org/docs/stable/generated/torch.sparse_csc_tensor.html#torch.sparse_csc_tensor "torch.sparse_csc_tensor") function. The user must supply the row and column indices and values tensors separately where the column indices must be specified using the CSR compression encoding. The `size` argument is optional and will be deduced from the `row_indices` and `ccol_indices` tensors if it is not present.

```
<span></span><span>&gt;&gt;&gt; </span><span>ccol_indices</span> <span>=</span> <span>torch</span><span>.</span><span>tensor</span><span>([</span><span>0</span><span>,</span> <span>2</span><span>,</span> <span>4</span><span>])</span>
<span>&gt;&gt;&gt; </span><span>row_indices</span> <span>=</span> <span>torch</span><span>.</span><span>tensor</span><span>([</span><span>0</span><span>,</span> <span>1</span><span>,</span> <span>0</span><span>,</span> <span>1</span><span>])</span>
<span>&gt;&gt;&gt; </span><span>values</span> <span>=</span> <span>torch</span><span>.</span><span>tensor</span><span>([</span><span>1</span><span>,</span> <span>2</span><span>,</span> <span>3</span><span>,</span> <span>4</span><span>])</span>
<span>&gt;&gt;&gt; </span><span>csc</span> <span>=</span> <span>torch</span><span>.</span><span>sparse_csc_tensor</span><span>(</span><span>ccol_indices</span><span>,</span> <span>row_indices</span><span>,</span> <span>values</span><span>,</span> <span>dtype</span><span>=</span><span>torch</span><span>.</span><span>float64</span><span>)</span>
<span>&gt;&gt;&gt; </span><span>csc</span>
<span>tensor(ccol_indices=tensor([0, 2, 4]),</span>
<span>       row_indices=tensor([0, 1, 0, 1]),</span>
<span>       values=tensor([1., 2., 3., 4.]), size=(2, 2), nnz=4,</span>
<span>       dtype=torch.float64, layout=torch.sparse_csc)</span>
<span>&gt;&gt;&gt; </span><span>csc</span><span>.</span><span>to_dense</span><span>()</span>
<span>tensor([[1., 3.],</span>
<span>        [2., 4.]], dtype=torch.float64)</span>
```

Note

The sparse CSC tensor constructor function has the compressed column indices argument before the row indices argument.

The (0 + 2 + 0)-dimensional sparse CSC tensors can be constructed from any two-dimensional tensor using [`torch.Tensor.to_sparse_csc()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to_sparse_csc.html#torch.Tensor.to_sparse_csc "torch.Tensor.to_sparse_csc") method. Any zeros in the (strided) tensor will be interpreted as missing values in the sparse tensor:

```
<span></span><span>&gt;&gt;&gt; </span><span>a</span> <span>=</span> <span>torch</span><span>.</span><span>tensor</span><span>([[</span><span>0</span><span>,</span> <span>0</span><span>,</span> <span>1</span><span>,</span> <span>0</span><span>],</span> <span>[</span><span>1</span><span>,</span> <span>2</span><span>,</span> <span>0</span><span>,</span> <span>0</span><span>],</span> <span>[</span><span>0</span><span>,</span> <span>0</span><span>,</span> <span>0</span><span>,</span> <span>0</span><span>]],</span> <span>dtype</span><span>=</span><span>torch</span><span>.</span><span>float64</span><span>)</span>
<span>&gt;&gt;&gt; </span><span>sp</span> <span>=</span> <span>a</span><span>.</span><span>to_sparse_csc</span><span>()</span>
<span>&gt;&gt;&gt; </span><span>sp</span>
<span>tensor(ccol_indices=tensor([0, 1, 2, 3, 3]),</span>
<span>       row_indices=tensor([1, 1, 0]),</span>
<span>       values=tensor([1., 2., 1.]), size=(3, 4), nnz=3, dtype=torch.float64,</span>
<span>       layout=torch.sparse_csc)</span>
```

### Sparse BSR Tensor

The sparse BSR (Block compressed Sparse Row) tensor format implements the BSR format for storage of two-dimensional tensors with an extension to supporting batches of sparse BSR tensors and values being blocks of multi-dimensional tensors.

A sparse BSR tensor consists of three tensors: `crow_indices`, `col_indices` and `values`:

> -   The `crow_indices` tensor consists of compressed row indices. This is a (B + 1)-D tensor of shape `(*batchsize, nrowblocks + 1)`. The last element is the number of specified blocks, `nse`. This tensor encodes the index in `values` and `col_indices` depending on where the given column block starts. Each successive number in the tensor subtracted by the number before it denotes the number of blocks in a given row.
>     
> -   The `col_indices` tensor contains the column block indices of each element. This is a (B + 1)-D tensor of shape `(*batchsize, nse)`.
>     
> -   The `values` tensor contains the values of the sparse BSR tensor elements collected into two-dimensional blocks. This is a (1 + 2 + K)-D tensor of shape `(nse, nrowblocks, ncolblocks, *densesize)`.
>     

#### Construction of BSR tensors

Sparse BSR tensors can be directly constructed by using the [`torch.sparse_bsr_tensor()`](https://docs.pytorch.org/docs/stable/generated/torch.sparse_bsr_tensor.html#torch.sparse_bsr_tensor "torch.sparse_bsr_tensor") function. The user must supply the row and column block indices and values tensors separately where the row block indices must be specified using the CSR compression encoding. The `size` argument is optional and will be deduced from the `crow_indices` and `col_indices` tensors if it is not present.

```
<span></span><span>&gt;&gt;&gt; </span><span>crow_indices</span> <span>=</span> <span>torch</span><span>.</span><span>tensor</span><span>([</span><span>0</span><span>,</span> <span>2</span><span>,</span> <span>4</span><span>])</span>
<span>&gt;&gt;&gt; </span><span>col_indices</span> <span>=</span> <span>torch</span><span>.</span><span>tensor</span><span>([</span><span>0</span><span>,</span> <span>1</span><span>,</span> <span>0</span><span>,</span> <span>1</span><span>])</span>
<span>&gt;&gt;&gt; </span><span>values</span> <span>=</span> <span>torch</span><span>.</span><span>tensor</span><span>([[[</span><span>0</span><span>,</span> <span>1</span><span>,</span> <span>2</span><span>],</span> <span>[</span><span>6</span><span>,</span> <span>7</span><span>,</span> <span>8</span><span>]],</span>
<span>... </span>                       <span>[[</span><span>3</span><span>,</span> <span>4</span><span>,</span> <span>5</span><span>],</span> <span>[</span><span>9</span><span>,</span> <span>10</span><span>,</span> <span>11</span><span>]],</span>
<span>... </span>                       <span>[[</span><span>12</span><span>,</span> <span>13</span><span>,</span> <span>14</span><span>],</span> <span>[</span><span>18</span><span>,</span> <span>19</span><span>,</span> <span>20</span><span>]],</span>
<span>... </span>                       <span>[[</span><span>15</span><span>,</span> <span>16</span><span>,</span> <span>17</span><span>],</span> <span>[</span><span>21</span><span>,</span> <span>22</span><span>,</span> <span>23</span><span>]]])</span>
<span>&gt;&gt;&gt; </span><span>bsr</span> <span>=</span> <span>torch</span><span>.</span><span>sparse_bsr_tensor</span><span>(</span><span>crow_indices</span><span>,</span> <span>col_indices</span><span>,</span> <span>values</span><span>,</span> <span>dtype</span><span>=</span><span>torch</span><span>.</span><span>float64</span><span>)</span>
<span>&gt;&gt;&gt; </span><span>bsr</span>
<span>tensor(crow_indices=tensor([0, 2, 4]),</span>
<span>       col_indices=tensor([0, 1, 0, 1]),</span>
<span>       values=tensor([[[ 0.,  1.,  2.],</span>
<span>                       [ 6.,  7.,  8.]],</span>
<span>                      [[ 3.,  4.,  5.],</span>
<span>                       [ 9., 10., 11.]],</span>
<span>                      [[12., 13., 14.],</span>
<span>                       [18., 19., 20.]],</span>
<span>                      [[15., 16., 17.],</span>
<span>                       [21., 22., 23.]]]),</span>
<span>       size=(4, 6), nnz=4, dtype=torch.float64, layout=torch.sparse_bsr)</span>
<span>&gt;&gt;&gt; </span><span>bsr</span><span>.</span><span>to_dense</span><span>()</span>
<span>tensor([[ 0.,  1.,  2.,  3.,  4.,  5.],</span>
<span>        [ 6.,  7.,  8.,  9., 10., 11.],</span>
<span>        [12., 13., 14., 15., 16., 17.],</span>
<span>        [18., 19., 20., 21., 22., 23.]], dtype=torch.float64)</span>
```

The (0 + 2 + 0)-dimensional sparse BSR tensors can be constructed from any two-dimensional tensor using [`torch.Tensor.to_sparse_bsr()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to_sparse_bsr.html#torch.Tensor.to_sparse_bsr "torch.Tensor.to_sparse_bsr") method that also requires the specification of the values block size:

```
<span></span><span>&gt;&gt;&gt; </span><span>dense</span> <span>=</span> <span>torch</span><span>.</span><span>tensor</span><span>([[</span><span>0</span><span>,</span> <span>1</span><span>,</span> <span>2</span><span>,</span> <span>3</span><span>,</span> <span>4</span><span>,</span> <span>5</span><span>],</span>
<span>... </span>                      <span>[</span><span>6</span><span>,</span> <span>7</span><span>,</span> <span>8</span><span>,</span> <span>9</span><span>,</span> <span>10</span><span>,</span> <span>11</span><span>],</span>
<span>... </span>                      <span>[</span><span>12</span><span>,</span> <span>13</span><span>,</span> <span>14</span><span>,</span> <span>15</span><span>,</span> <span>16</span><span>,</span> <span>17</span><span>],</span>
<span>... </span>                      <span>[</span><span>18</span><span>,</span> <span>19</span><span>,</span> <span>20</span><span>,</span> <span>21</span><span>,</span> <span>22</span><span>,</span> <span>23</span><span>]])</span>
<span>&gt;&gt;&gt; </span><span>bsr</span> <span>=</span> <span>dense</span><span>.</span><span>to_sparse_bsr</span><span>(</span><span>blocksize</span><span>=</span><span>(</span><span>2</span><span>,</span> <span>3</span><span>))</span>
<span>&gt;&gt;&gt; </span><span>bsr</span>
<span>tensor(crow_indices=tensor([0, 2, 4]),</span>
<span>       col_indices=tensor([0, 1, 0, 1]),</span>
<span>       values=tensor([[[ 0,  1,  2],</span>
<span>                       [ 6,  7,  8]],</span>
<span>                      [[ 3,  4,  5],</span>
<span>                       [ 9, 10, 11]],</span>
<span>                      [[12, 13, 14],</span>
<span>                       [18, 19, 20]],</span>
<span>                      [[15, 16, 17],</span>
<span>                       [21, 22, 23]]]), size=(4, 6), nnz=4,</span>
<span>       layout=torch.sparse_bsr)</span>
```

### Sparse BSC Tensor

The sparse BSC (Block compressed Sparse Column) tensor format implements the BSC format for storage of two-dimensional tensors with an extension to supporting batches of sparse BSC tensors and values being blocks of multi-dimensional tensors.

A sparse BSC tensor consists of three tensors: `ccol_indices`, `row_indices` and `values`:

> -   The `ccol_indices` tensor consists of compressed column indices. This is a (B + 1)-D tensor of shape `(*batchsize, ncolblocks + 1)`. The last element is the number of specified blocks, `nse`. This tensor encodes the index in `values` and `row_indices` depending on where the given row block starts. Each successive number in the tensor subtracted by the number before it denotes the number of blocks in a given column.
>     
> -   The `row_indices` tensor contains the row block indices of each element. This is a (B + 1)-D tensor of shape `(*batchsize, nse)`.
>     
> -   The `values` tensor contains the values of the sparse BSC tensor elements collected into two-dimensional blocks. This is a (1 + 2 + K)-D tensor of shape `(nse, nrowblocks, ncolblocks, *densesize)`.
>     

#### Construction of BSC tensors

Sparse BSC tensors can be directly constructed by using the [`torch.sparse_bsc_tensor()`](https://docs.pytorch.org/docs/stable/generated/torch.sparse_bsc_tensor.html#torch.sparse_bsc_tensor "torch.sparse_bsc_tensor") function. The user must supply the row and column block indices and values tensors separately where the column block indices must be specified using the CSR compression encoding. The `size` argument is optional and will be deduced from the `ccol_indices` and `row_indices` tensors if it is not present.

```
<span></span><span>&gt;&gt;&gt; </span><span>ccol_indices</span> <span>=</span> <span>torch</span><span>.</span><span>tensor</span><span>([</span><span>0</span><span>,</span> <span>2</span><span>,</span> <span>4</span><span>])</span>
<span>&gt;&gt;&gt; </span><span>row_indices</span> <span>=</span> <span>torch</span><span>.</span><span>tensor</span><span>([</span><span>0</span><span>,</span> <span>1</span><span>,</span> <span>0</span><span>,</span> <span>1</span><span>])</span>
<span>&gt;&gt;&gt; </span><span>values</span> <span>=</span> <span>torch</span><span>.</span><span>tensor</span><span>([[[</span><span>0</span><span>,</span> <span>1</span><span>,</span> <span>2</span><span>],</span> <span>[</span><span>6</span><span>,</span> <span>7</span><span>,</span> <span>8</span><span>]],</span>
<span>... </span>                       <span>[[</span><span>3</span><span>,</span> <span>4</span><span>,</span> <span>5</span><span>],</span> <span>[</span><span>9</span><span>,</span> <span>10</span><span>,</span> <span>11</span><span>]],</span>
<span>... </span>                       <span>[[</span><span>12</span><span>,</span> <span>13</span><span>,</span> <span>14</span><span>],</span> <span>[</span><span>18</span><span>,</span> <span>19</span><span>,</span> <span>20</span><span>]],</span>
<span>... </span>                       <span>[[</span><span>15</span><span>,</span> <span>16</span><span>,</span> <span>17</span><span>],</span> <span>[</span><span>21</span><span>,</span> <span>22</span><span>,</span> <span>23</span><span>]]])</span>
<span>&gt;&gt;&gt; </span><span>bsc</span> <span>=</span> <span>torch</span><span>.</span><span>sparse_bsc_tensor</span><span>(</span><span>ccol_indices</span><span>,</span> <span>row_indices</span><span>,</span> <span>values</span><span>,</span> <span>dtype</span><span>=</span><span>torch</span><span>.</span><span>float64</span><span>)</span>
<span>&gt;&gt;&gt; </span><span>bsc</span>
<span>tensor(ccol_indices=tensor([0, 2, 4]),</span>
<span>       row_indices=tensor([0, 1, 0, 1]),</span>
<span>       values=tensor([[[ 0.,  1.,  2.],</span>
<span>                       [ 6.,  7.,  8.]],</span>
<span>                      [[ 3.,  4.,  5.],</span>
<span>                       [ 9., 10., 11.]],</span>
<span>                      [[12., 13., 14.],</span>
<span>                       [18., 19., 20.]],</span>
<span>                      [[15., 16., 17.],</span>
<span>                       [21., 22., 23.]]]), size=(4, 6), nnz=4,</span>
<span>       dtype=torch.float64, layout=torch.sparse_bsc)</span>
```

## Supported operations

### Linear Algebra operations

The following table summarizes supported Linear Algebra operations on sparse matrices where the operands layouts may vary. Here `T[layout]` denotes a tensor with a given layout. Similarly, `M[layout]` denotes a matrix (2-D PyTorch tensor), and `V[layout]` denotes a vector (1-D PyTorch tensor). In addition, `f` denotes a scalar (float or 0-D PyTorch tensor), `*` is element-wise multiplication, and `@` is matrix multiplication.

where “Sparse grad?” column indicates if the PyTorch operation supports backward with respect to sparse matrix argument. All PyTorch operations, except [`torch.smm()`](https://docs.pytorch.org/docs/stable/generated/torch.smm.html#torch.smm "torch.smm"), support backward with respect to strided matrix arguments.

Note

Currently, PyTorch does not support matrix multiplication with the layout signature `M[strided] @ M[sparse_coo]`. However, applications can still compute this using the matrix relation `D @ S == (S.t() @ D.t()).t()`.

### Torch functions specific to sparse Tensors
