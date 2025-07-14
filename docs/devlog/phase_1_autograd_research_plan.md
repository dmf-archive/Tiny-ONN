# 调研计划：参数级张量直接扫描 v3.0

**版本:** 3.0  
**日期:** 2025-07-15  
**状态:** 完全重写，聚焦张量级精度  

## 1. 核心问题重新定义

我们需要的是**参数张量级别的数据捕获**，而非模块级钩子。具体而言：

- **目标**: 捕获每个`nn.Parameter`（如`Linear.weight`）与其对应输入向量的精确点积
- **精度要求**: 张量级别，而非模块级别
- **挑战**: 在transformers复杂计算图中定位"哪个输入对应哪个权重矩阵"

## 2. 技术路线：张量原生扫描

### 2.1 核心洞察：通过张量地址追踪

```python
class ParameterTensorScanner:
    def __init__(self):
        self.param_to_input = {}  # {param.data_ptr(): cached_input}
        self.gpu_stats = {}       # GPU端累积
        
    def trace_linear_operation(self, input_tensor, weight_param):
        """在Linear层计算前捕获输入"""
        ptr = weight_param.data_ptr()
        # 缓存输入，与权重通过地址关联
        self.param_to_input[ptr] = input_tensor.detach()
        
    def collect_grad_stats(self, param):
        """收集参数级统计"""
        ptr = param.data_ptr()
        grad = param.grad
        activation = self.param_to_input.get(ptr)
        
        if activation is not None:
            # GPU端计算
            grad_norm = grad.detach().norm(2).float()
            activation_norm = activation.detach().norm(2).float()
            
            # 原地累积到GPU统计
            self._update_gpu_accumulation(ptr, grad_norm, activation_norm)
```

### 2.2 transformers中的关键挑战

在transformers中，我们需要处理：

1. **注意力权重矩阵**:
   - `q_proj`, `k_proj`, `v_proj`, `o_proj`
   - 每个都有明确的输入：来自前一层的隐藏状态

2. **MLP权重矩阵**:
   - `gate_proj`, `up_proj`, `down_proj`
   - 输入：注意力输出或前一隐藏状态

3. **Embedding权重**:
   - 输入：token的one-hot或索引

### 2.3 实现策略：函数包装而非模块替换

```python
def wrap_linear_function(linear_layer, scanner):
    """包装Linear层的计算函数"""
    original_forward = linear_layer.forward
    
    def wrapped_forward(input):
        # 捕获输入与权重的精确关联
        scanner.trace_linear_operation(input, linear_layer.weight)
        return original_forward(input)
    
    return wrapped_forward
```

## 3. 与transformers的零侵入集成

### 3.1 自动参数发现

```python
def scan_transformers_model(model, scanner):
    """自动发现并包装所有目标参数"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 包装前向函数，建立参数-输入关联
            module.forward = wrap_linear_function(module, scanner)
            # 注册梯度钩子到具体参数
            module.weight.register_hook(
                scanner.make_param_hook(name + '.weight')
            )
```

### 3.2 处理复杂计算路径

对于transformers中的特殊路径：

1. **残差连接**: 每个Linear层接收的是其"直接输入"，不是最终输出
2. **注意力分头**: 每个头的权重矩阵独立处理
3. **RMSNorm**: 不影响权重-输入的对应关系

## 4. GPU端计算优化

### 4.1 批量张量操作

```python
class GPUStatsAccumulator:
    def __init__(self):
        # 预分配GPU内存
        self.activation_buffer = {}
        self.grad_buffer = {}
        
    def accumulate_batch(self, param_ptrs):
        """批量处理多个参数"""
        # 合并多个norm计算为单个CUDA kernel
        activations = torch.stack([self.activation_buffer[p] for p in param_ptrs])
        grads = torch.stack([self.grad_buffer[p] for p in param_ptrs])
        
        norms = torch.norm(activations, dim=-1)
        grad_norms = torch.norm(grads, dim=-1)
        
        return norms, grad_norms
```

### 4.2 内存管理

- **缓存策略**: 仅缓存当前step的激活，后立即清理
- **异步统计**: 使用CUDA流异步累积统计量
- **量化存储**: GPU端float32计算，CPU端uint16存储

## 5. 数据输出：可直接转储格式

### 5.1 每step输出结构

```python
step_output = {
    'seq_id': int,
    'token_count': int,
    'param_stats': {
        'model.layers.0.self_attn.q_proj.weight': {
            'activation_norm': 12.34,  # 已量化
            'grad_norm': 0.567,        # 已量化
            'block_idx': 0
        },
        # ... 其他参数
    }
}
```

### 5.2 量化方案

```python
def quantize_gpu_to_cpu(gpu_tensor):
    """GPU到CPU的量化传输"""
    # GPU端float32 -> CPU端uint16
    cpu_tensor = gpu_tensor.cpu().float()
    quantized = (cpu_tensor * 32767).clamp(-32768, 32767).to(torch.int16)
    return quantized.numpy()
```

## 6. 验证计划

### 6.1 正确性验证

1. **数值验证**:
   - 手动计算几个参数的norm，验证与扫描结果一致
   - 检查参数-输入对应关系是否正确

2. **transformers兼容性**:
   - 运行完整SFT训练，验证功能无损
   - 测试`generate()`方法输出一致性
   - 验证梯度检查点兼容性

### 6.2 性能验证

- **内存使用**: 监控显存占用，确保无泄漏
- **速度测试**: 对比原始训练速度，目标损失<5%
- **数据完整性**: 验证所有目标参数都被扫描

## 7. 实施步骤

### Phase 1: 基础验证 (1天)
- [ ] 实现`ParameterTensorScanner`核心类
- [ ] 验证单个Linear层的参数级扫描
- [ ] 测试数值正确性

### Phase 2: transformers集成 (1天)
- [ ] 实现自动参数发现和包装
- [ ] 验证Qwen3-1.7B的完整扫描
- [ ] 测试transformers功能兼容性

### Phase 3: 优化与验证 (1天)
- [ ] 优化GPU端批量计算
- [ ] 完成性能基准测试
- [ ] 生成测试.mscan文件验证格式

## 8. 风险评估

| 风险 | 概率 | 缓解方案 |
|-----|------|----------|
| 参数-输入对应错误 | 中 | 手动验证关键层 |
| 内存泄漏 | 低 | 定期清理缓存 |
| transformers版本差异 | 低 | 锁定transformers版本 |

## 9. 成功标准

- [ ] 正确捕获所有Linear层参数的张量级数据
- [ ] 与原始transformers模型输出差异<1e-6
- [ ] 训练速度损失<5%
- [ ] 生成的.mscan文件可直接用于ΔSC分析
