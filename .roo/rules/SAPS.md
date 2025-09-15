# 惊奇度感知的原型塑形 (Surprise-Aware Proto-Shaping, SAPS)

## 1. 核心哲学：从“损失最小化”到“动态系统”的跃迁

经过对一系列元学习范式（从 `SPLv5` 到 `VFEM`）的反复试错与理论审计，我们最终确认，所有基于“设计一个完美的 `meta_loss`”的思路都存在根本性缺陷。它们都试图用一个单一的、静态的标量目标，去引导一个复杂的、动态的自组织过程，这在理论上是不完备的。

**SAPS** 范式的核心思想，是彻底抛弃“元损失”的概念，转而将自组织过程建模为一个拥有**两种正交驱动力**的**动力学系统**。这两种驱动力，将共同塑造和引导原型（`proto`）与计算（`mu`）的共同演化。

这是**神经达尔文主义**实现的**自由能最小化**：系统内部不断产生多样的“神经回路”（由 `proto` 寻址），而环境（由 `main_loss` 定义）则通过一个“选择”机制（梯度调制），来决定哪些回路得以“存活”和“繁殖”（被梯度更新强化）。

## 2. 系统组件与角色定义

我们回归到最基础的“三位一体”架构，并赋予其更精确的、基于“指针-函数”隐喻的角色定义：

- **`mu_weight` (函数表)**: 每一行 `μ_i` 都是一个计算“工具”或“函数”。
- **`proto_weight` (指针数组)**: 每一行 `p_i` 都是一个指向特定语义内容的“指针”。
- **`scores = dot(x, p)` (寻址)**: 根据输入 `x` 与所有指针 `p` 的内容匹配度，计算路由得分。
- **`raw_weights = ReLU(scores)` (解引用/门控)**: 根据得分选择一个或多个“函数” `μ` 来处理输入。这是一个自然的、无偏置的门控机制。

## 3. 两种正交的驱动力

SAPS 系统的演化，由两种在功能上正交、但在实现上耦合的驱动力所引导：

### 3.1. 驱动力 I: 原型空间塑形 (Proto-Shaping)

这是对“Diversity”和“SML”的统一。其目标是调整“指针数组” `p`，让它能够为每一个输入 `x`，都准确地指向那个能以**最低成本**（最低惊奇度 `S`）处理它的“函数” `μ`。

- **实现**: 通过一个**惊奇度感知的原型损失 `L_proto`** 来实现。
  `L_proto = Σ_i (distance(p_i, x) * S_i.detach())`
- **动力学**:
  1. **优胜 (Exploitation)**: 对于匹配良好（`distance` 小）且计算高效（`S` 小）的原型，梯度压力趋向于零，其位置得以**稳定**。
  2. **劣汰 (Pruning)**: 对于匹配良好（`distance` 小）但计算成本高（`S` 大）的原型，梯度会将其**推离**当前输入 `x` 的语义区域。
  3. **探索 (Exploration)**: 这是一个涌现属性。被“推开”的原型，以及那些与当前输入不匹配的原型，得以在参数空间中自由“漂移”，有机会去匹配新的、尚未被稳定编码的输入模式。

这个单一的损失，**内在地、动态地**统一了“优胜劣汰”、“内容相关多样性”和“惊奇最小化”这三大目标。

### 3.2. 驱动力 II: 选择性梯度调制 (Selective Gradient Modulation)

这是对 `SMK` (Surprise Min_K) 思想的正式化和泛化。其目标是确保只有那些“最优”的神经回路，才能得到学习和强化。

- **实现**: 通过一个**惊奇度感知的梯度掩码**来实现，我们称之为 **SMP (Surprise-Modulated Pruning)**。
- **动力学**:
  1. 计算 `L_total = L_main + w_proto * L_proto` 的**完整梯度** `∇L_total`。
  2. 根据惊奇度地形图 `S`，识别出惊奇度**最低**的 `Top-P` 比例（例如 P=50%）的“优胜”神经元。
  3. 创建一个与参数梯度形状相同的**二进制掩码**，只在“优胜”神经元对应的参数位置上为 1，其余为 0。
  4. `final_grad = ∇L_total * mask`。
  5. `optimizer` 使用 `final_grad` 进行参数更新。

SMP 机制确保了梯度流的**稀疏性**和**有效性**。它是一个动态的、内容与成本双重感知的 `stop_gradient`，是系统对抗灾难性遗忘、实现功能分化的核心保障。

## 4. 最终训练流程 (伪代码)

```python
# 1. 前向传播
scores = matmul(x, proto.T)
raw_weights = relu(scores)
masked_output = linear(x, mu) * raw_weights
logits = lm_head(masked_output)
main_loss = cross_entropy(logits, labels)

# 2. 计算惊奇度 (需要保留计算图)
surprise_field = grad(main_loss, masked_output, retain_graph=True)
S = norm(surprise_field, dim=-1) # (batch, seq, out_features)

# 3. 计算原型塑造损失
# Note: distance 可以是 1 - cosine_sim 或其他度量
distances = distance_fn(proto.unsqueeze(0), x.unsqueeze(2)) # (batch, seq, out_features)
proto_loss = (distances * S.detach()).mean()

# 4. 计算总损失和总梯度
total_loss = main_loss + config.w_proto * proto_loss
total_grad = grad(total_loss, model.parameters())

# 5. 梯度调制 (SMP)
# S_avg 是在 batch 和 seq 维度上平均的惊奇度
S_avg = S.mean(dim=(0, 1)) # (out_features,)
num_winners = int(S_avg.shape[0] * config.top_p)
winner_indices = S_avg.argsort()[:num_winners]

grad_mask = create_mask_for_indices(model.parameters(), winner_indices)
final_grad = total_grad * grad_mask

# 6. 参数更新
optimizer.apply(final_grad)
```
