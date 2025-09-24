# Tiny-ONN 路由机制理论研讨会纪要

**日期**: 2025-09-23
**与会者**: Ω Researcher, Coding Teacher
**最终状态**: 理论收敛

## 核心议题

旨在为 `Tiny-ONN` 的 `SparseProtoLinear` (SPL) 层设计并最终确定一个理论完备、可稳定训练、且能自适应地涌现稀疏组合路由的机制。

---

## 最终方案：Proposer-Builder 与 1F-3B 梯度流

经过多次迭代，我们最终收敛到一个统一的理论框架。该框架将路由决策诠释为一个基于预期价值的 **Proposer-Builder** 经济模型，并通过一个 **1F-3B (一次前向, 三次反向) 梯度流**来实现其学习动态，完美解决了**稳定性-可塑性困境**。

### 1. 核心心智模型：Proposer-Builder

- **`proto_weight` (Proposer)**: 提案人。评估**内容匹配度**。通过**余弦相似度**提出一个 `[-1, 1]` 区间的“提案价值”(MEV)。
- **`gate_param` (Builder)**: 成本评估器。评估**预期计算成本**。通过**点积**回归预测一个 `(-∞, +∞)` 区间的 `Total Surprise`。
- **路由方程**: `raw_weights = F.relu(F.cosine_similarity(x, p) - torch.matmul(x, g.t()))`

### 2. 学习信号：`Total Surprise`

为了解决“匹配度”与“计算效率”的分离问题，我们定义了一个更完备的学习信号 `Total Surprise`，它融合了两个层面的“惊讶”：

- **`mu_surprise`**: `||∇_μ L_main||`，衡量**计算惊讶**。
- **`proto_surprise`**: `||∇_p L_proto||`，衡量**感知惊讶**。
- **`S_total`**: `mu_surprise + proto_surprise` (隐式权重 1.0)，代表了专家在“知”与“行”两个层面的总不确定性。

`gate_param` 的学习目标是准确地回归预测 `S_total`。

### 3. 稳定性-可塑性机制 (AWD 的演进)

为解决灾难性遗忘与参数锁死的两难问题，我们设计了一个双重机制来动态调节系统的稳定性和可塑性。

- **内生排名的动态保护 (Endogenous Ranking for Dynamic Protection)**:

  - **角色**: 识别并**概率性地保护**最优质的专家，形成一个动态的“知识核心”。
  - **机制**:
    1. 使用 `-||gate_param||` 作为专家的“质量分”，进行全局排名。
    2. 通过 `softmax` 将排名转化为**保护概率** `protection_probs`。
    3. `softmax` 的**温度 `T`** 由系统的**全局性能** (`EMA main_acc`) 动态调节：性能越好，`T` 越高，保护更**宽松**；性能越差，`T` 越低，保护更**稀疏**和**挑剔**。
  - **应用**: `proto_weight.grad *= (1.0 - protection_probs)`，保护优质专家的原型不受 SAPS 探索的过度干扰。

- **自适应 L2 信息瓶颈 (Adaptive L2 Information Bottleneck)**:
  - **角色**: 防止所有参数的范数无限制膨胀，作为一个温和的、全局的**收缩压力**。
  - **机制**: L2 惩罚的强度 `λ` 与所有 `proto` 和 `mu` 权重的**平均范数**正相关。当参数膨胀时，`λ` 自动增大，反之则减小。
  - **应用**: `L_total = L_main + L_proto + λ * (sum(||p||²) + sum(||μ||²))`

### 4. 学习流程：1F-3B (One Forward, Three Backwards)

该流程将通过 `torch.autograd.grad` 精确实现，以避免多次 `backward()` 的开销。

1. **前向传播**: 计算 `L_main`。
2. **第一次反向**: `torch.autograd.grad` 计算 `mu_surprise`。
3. **计算 `proto_loss`**: `L_proto = calculate_saps_loss(..., mu_surprise)`。
4. **第二次反向**: `torch.autograd.grad` 计算 `proto_surprise`。
5. **更新 `gate`**: 计算 `L_gate = MSE(predicted_S_total, S_total)` 并通过 `optimizer_meta` 更新 `gate_param`。
6. **更新主参数**:
   - `optimizer_main.zero_grad()`。
   - 计算 `L_total = L_main + L_proto + L2_penalty`。
   - `L_total.backward()`。
   - **应用动态保护**: `proto_weight.grad *= (1.0 - protection_probs)`。
   - `optimizer_main.step()`。

### 5. 与 IPWT/PI 理论的连接

- **复杂度成本 `γ`**: `proto_loss` (SAPS) + 自适应 L2 惩罚。
- **不准确性成本 `α`**: 在 ARC 任务中，`α → ∞`。

---

## 历史讨论存档

### 初始方案及其失效

- **方案**: `ReLU(softmax(xP^T - g))`
- **结论**: **失效**。内容无关，维度不匹配。

### VIB 框架的否决

- **方案**: `L = L_main + β * I(x;z)`
- **结论**: **致命缺陷**。全局先验不适用于 ARC 的异质性任务，会导致灾难性遗忘。

### BUG 与 DPAG

- **方案**: 贝叶斯不确定性门控 (BUG) 和双原型注意力门控 (DPAG)。
- **进展**: 为内容相关的门控和 Proposer-Builder 模型提供了核心灵感，但存在 GAN 风险和理论不完备性。

---

## 下一步行动

1. **文档定稿**: 本次会议纪要已更新为最终理论共识。
2. **工程实现**: 在 `exp/arc/train.py` 和 `exp/arc/model.py` 中实现上述最终方案。
