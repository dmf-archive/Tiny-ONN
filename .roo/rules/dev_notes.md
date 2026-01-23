# Tiny-ONN 开发笔记 (Dev Notes)

---

## 第一部分：ARCTrainer & FARS 重构笔记

### 1. 性能对比分析：FARS vs SARS

| 维度 | SARS (旧版 `arc_dyntrm`) | FARS (新版 `arc_v2`) | 结论 |
| :--- | :--- | :--- | :--- |
| **计算路径** | 需捕获中间梯度 + 额外 `backward(meta_loss)` | 直接读取优化器 `exp_avg_sq` 状态 | **FARS 胜** |
| **耗时开销** | 显著（涉及二阶导数逻辑与张量同步） | **~3.10 ms** (仅占 Step 总时长的 0.15%) | **FARS 胜** |
| **显存占用** | 极高（需缓存所有 Expert 的激活与梯度） | **几乎为零** (复用已有的优化器状态) | **FARS 胜** |
| **信号质量** | 瞬时梯度噪声大，路由易抖动 | Fisher 信息代表长期不确定性，更平滑 | **FARS 胜** |

### 2. 核心心智模型：从“主动内省”到“历史反思”

- **SARS (Surprise-Aware)** 像是一个人在做每件事时都要停下来问自己：“我刚才这步做得有多难？”这种**主动内省**需要中断心流，记录大量细节，极其耗能。
- **FARS (Fisher-Aware)** 则像是通过**历史反思**：它直接查看 [`ARS2Neo`](src/optimizers/ars2_neo.py:69) 优化器在过去一段时间内对某个模块的“修改剧烈程度”（即二阶矩 $v_t$）。如果一个模块的 $v_t$ 很大，说明它对当前数据分布还不稳定（认知代价高），路由应避开它。

### 3. 踩坑经验 (Dev Log)

1.  **量纲对齐**: 在 [`RoutingShaper.calculate_meta_loss()`](src/tasks/arc/shaper.py:13) 中，我们将 `entropy` (无量纲) 与 `cost_fars` (基于 $\sqrt{v_t}$) 结合。目前的 `cost_alpha` 设为 0.1，这需要根据训练初期的 `meta_loss` 占比进行动态微调。
2.  **模块映射**: FARS 的高效性建立在“模块级”范数上。我们通过 [`model.named_modules()`](src/tasks/arc/shaper.py:26) 快速定位 `experts`，避免了逐参数计算的冗余。
3.  **SAM 兼容性**: 在 [`ARCTrainer.train_step`](src/tasks/arc/trainer.py:66) 中，`meta_loss` 必须被包含在 `closure()` 内部。这样在 SAM 的两次前向传播中，路由塑造信号能与平坦度约束协同工作。

---

## 第二部分：Transformers 自定义模型实现笔记

### 1. 生成功能的继承要求
为了支持 `.generate()` 方法，自定义模型必须同时继承 `PreTrainedModel` 和 `GenerationMixin`。
在最近的 `transformers` 版本（v4.50+）中，`PreTrainedModel` 不再自动继承 `GenerationMixin`。

```python
from transformers import PreTrainedModel, GenerationMixin

class MyModel(PreTrainedModel, GenerationMixin):
    ...
```

### 2. `generate()` 必需的方法
- `prepare_inputs_for_generation`: 为生成过程的每一步准备输入。
- `forward`: 必须返回包含 `logits` 的对象（或字典）。

### 3. `prepare_inputs_for_generation` 的实现
一个健壮的实现应该处理 `past_key_values`（KV 缓存），即使当前模型架构尚未使用它，以避免在生成循环中崩溃。

```python
def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
    # 如果使用 KV 缓存，将 input_ids 切片为仅保留最后一个 token
    if past_key_values is not None:
        input_ids = input_ids[:, -1:]
        
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": kwargs.get("position_ids", None),
    }
```

### 4. 输出格式
`forward` 方法理想情况下应返回 `transformers.modeling_outputs.CausalLMOutputWithPast` 或至少包含 `logits` 的字典。

---

## 第三部分：路由架构表达力评估 (2026-01-23)

### 1. 实验结论：MLP 路由的优越性
通过对 Linear, MLP, CAPR, CPR 等架构的 CPU 周期压测，结论如下：
- **表达力**: MLP (2-layer) > Linear > CAPR。MLP 是唯一能完美解决 XOR 逻辑路由的架构。
- **效率**: MLP 的 CPU 耗时与 Linear 相当，显著优于基于 Attention 的 CAPR。
- **一致性**: MLP 产生的路由 Logits 具有最低的熵（最清晰的决策边界）。

### 2. 策略转向
- **核心驱动**: 放弃复杂的 CAPR，转向 **MLP 路由 + FARS 塑造**。
- **泛化保障**: 依靠 **ARS2-Neo (AGA)** 的平坦度约束来防止 MLP 路由过拟合。
- **监控指标**: 引入 **路由一致性 Loss (Entropy)** 作为不参与 BP 的检测指标，用于评估专家分发的稳定性。
