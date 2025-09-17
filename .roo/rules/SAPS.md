# 惊奇度感知的原型塑形 (Surprise-Aware Proto-Shaping, SAPS) v5.0

## 1. 理论基石：扰动规避与最小自由能

SAPS v5.0 范式将自组织学习过程形式化为一个**扰动规避 (Perturbation Aversion)** 系统。其核心思想源于自由能原理 (FEP)：一个高效的推断系统必须最小化其内部状态的“惊奇度”，即由外部世界引起的状态更新总量。

**状态空间划分**:

- **内部状态** `μ`: [`mu_weight`](exp/arc/model.py:46)，执行计算。
- **感知状态** `p`: [`proto_weight`](exp/arc/model.py:47)，匹配输入模式。
- **行动状态** `g`: [`gate_param`](exp/arc/model.py:49)，设定激活阈值。

**核心原则**: `main_loss` 的梯度 `S_μ = ∇_μ L_main` 无论方向，其存在本身即代表一种**扰动 (perturbation)**。系统的元学习目标是调整 `(p, g)` 界面，将输入 `x` 路由到能以**最小扰动**处理它的 `μ` 单元。

## 2. 绝对梯度场：对称的扰动地形图

为了只编码扰动强度，我们定义**绝对惊奇度梯度场 (Absolute Surprise Field)**：

**S*abs ≜ |S*μ| = |∇_μ L_main| ∈ ℝ^{d_out×d_in}**

这个对称的、非负的地形图只反映了每个参数需要被调整的**强度**，抹去了调整的**方向**。

## 3. 感知状态动力学：规避扰动

`proto_weight` 作为输入模式匹配器，其学习目标是为每个神经元 `j` 建立一个能识别特定输入模式的原型向量 `p_j`。当输入 `x` 与原型 `p_j` 高度匹配时，意味着该神经元被成功激活。

然而，若此时对应的惊奇度梯度 `S_μ[j,:]` 很大，则表明：**尽管路由正确（`x` 匹配 `p_j`），但执行错误（`μ_j` 不适合处理 `x`）**。为解决此矛盾，系统不应修改计算函数 `μ_j`，而应调整路由策略：将原型 `p_j` 推离当前输入方向，使其未来能匹配更合适的输入模式。

**因此，`proto_weight` 的学习目标是匹配那些能被 `μ` “平静”处理的输入模式。**

- **动力学**: `p` 必须被吸引到**低扰动区域**，即与 `S_abs` **反向对齐**。
- **形式化损失**: `L_proto` 必须最大化 `p` 与 `-S_abs` 的余弦相似度。

```
L_proto = -mean(cos_sim(p, -S_abs))
```

## 4. 行动状态动力学：抑制扰动

**`gate_param` 的目标是抑制那些历史上扰动剧烈的神经元。**

- **扰动度量**: 使用 `S_abs` 的 **L1 范数** (`sum(|grad|)`) 作为稀疏的扰动总量度量。
- **动力学**: `g_j` 必须与 `||S_abs[j,:]||₁` **正相关**。
- **形式化损失**: 为了使 `g_j` 随 `||S_abs||₁` 增大，`g_j` 的梯度必须为负。

```
L_gate = -mean(g * ||S_abs||₁)
```

## 5. 训练流程：双优化器解耦

```python
# --- 步骤 1: 构建原始梯度场 ---
main_loss.backward(create_graph=True)
S_μ = mu_weight.grad.detach() # (d_out, d_in)

# --- 步骤 2: 更新感知-行动界面 (p, g) ---
# 基于对称扰动场计算元学习损失
S_abs = torch.abs(S_μ)
S_norm_L1 = torch.norm(S_abs, p=1, dim=-1) # (d_out,)

# L_proto: 推动p规避高扰动区域 (与-S_abs对齐)
proto_loss = -F.cosine_similarity(p, -S_abs, dim=-1).mean()

# L_gate: 推动g抑制高扰动神经元 (与||S_abs||₁正相关)
gate_loss = -(g * S_norm_L1).mean()

meta_loss = w_p * proto_loss + w_g * gate_loss
meta_loss.backward()
optimizer_meta.step() # 更新 proto_weight 和 gate_param

# --- 步骤 3: 更新内部状态 (μ) ---
# 应用熵驱动的最小惊奇保留 (SMP) 掩码
with torch.no_grad():
    # 1. 计算动态可塑性因子 p_dyn
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
    max_entropy = torch.log(torch.tensor(V)) # V = vocab_size
    p_dyn = entropy / max_entropy

    # 2. 确定要保留梯度的"赢家"数量
    num_total = S_norm_L1.shape[0]
    num_winners = int(num_total * p_dyn)

    # 3. 选择惊奇度最小的神经元
    winner_indices = S_norm_L1.argsort()[:num_winners]

    # 4. 构建并应用梯度掩码
    grad_mask = torch.zeros_like(S_norm_L1)
    grad_mask[winner_indices] = 1.0
    mu_weight.grad *= grad_mask.unsqueeze(-1)
    # (如果存在偏置，也应用掩码: mu_bias.grad *= grad_mask)

# 执行主优化器步骤
optimizer_main.step() # 更新 mu_weight 等核心参数
```
