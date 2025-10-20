---
title: "理论备忘录：AdaGNoema 优化器"
status: "Formal Draft"
date: "2025-10-20"
authors: ["Ω Researcher"]
tags:
  ["theory", "optimizer", "adagnoema", "second-order", "gauss-newton", "fep"]
---

## 1. 背景：从启发式到第一性原理

`Tiny-ONN` 项目的早期探索，无论是显式的 `SARS` 架构还是隐式的 `AdaNoema` 构想，都试图通过各种启发式代理（如梯度范数、激活值范数）来模拟自由能原理 (FEP) 的学习动态。然而，这些尝试最终都因理论上的不完备和实现上的不稳定性而失败。

`ADR-0002` 的核心洞察是：**FEP 的信念更新在数学上等价于二阶（牛顿）优化**。因此，我们不再需要任何启发式代理。我们可以直接实现一个近似二阶优化器，以此作为 OFE（Observation Free Energy） 的直接工程化身。

本文旨在形式化一个新的优化器理论：**自适应高斯-牛顿意向内容优化器 (Adaptive Gauss-Newton Noema Optimizer, AdaGNoema)**。它废弃了 `AdaNoema` 的所有一阶启发式，基于参考论文 `arXiv:2510.09378v1` 中描述的**逐层高斯-牛顿 (Layerwise Gauss-Newton, LGN)** 方法。

## 2. 核心构想：作为内部优化问题的二阶更新

`Layerwise Gaussian-Newton` 的核心思想是将牛顿法的更新步骤 `Δθ = -G⁻¹g` 重新表述为一个等价的、可迭代求解的二次优化问题，从而避免了直接计算和求逆巨大的高斯-牛顿矩阵 `G`。

对于模型的每一层 `l`，我们旨在找到一个更新方向 `d`，使其能最小化一个局部的二次损失函数 `L_quad_l(d)`：

`L_quad_l(d) = g_lᵀd + 0.5 * dᵀG_l d`

其中：

- `g_l` 是该层参数的一阶梯度。
- `G_l` 是该层参数的高斯-牛顿矩阵。

一旦通过内部优化循环求解出最优的 `d*`，我们就可以用它来更新该层的真实参数：`θ_l ← θ_l + α * d*` (其中 `α` 是全局学习率)。

### 2.1. 无矩阵实现 (Matrix-Free Implementation)

直接计算 `dᵀG_l d` 仍然需要 `G_l`。`AdaGNoema` 的关键在于采用**无矩阵**方法，利用 `torch.func` 中的雅可比-向量积 (`jvp`) 和向量-雅可比积 (`vjp`) 来高效计算高斯-牛顿-向量积 (GNVP)，即 `G_l * d`，而无需实例化 `G_l` 本身。

根据参考论文，`G_l * d` 可以分解为：

`G_l * d = J_lᵀ * H_L * (J_l * d)`

其中：

- `J_l` 是模型输出对于层 `l` 参数的雅可比矩阵。
- `H_L` 是损失函数 `L` 对于模型输出的 Hessian 矩阵。

这三项都可以通过 `torch.func` 或 `torch.autograd.grad` 的组合高效计算，使得在内部循环中计算 `L_quad_l(d)` 变得可行。

## 3. 算法伪代码：AdaGNoema

`AdaGNoema` 本质上是一个两层嵌套的优化器。

```python
class AdaGNoema(Optimizer):
    def __init__(self, params, inner_optimizer_cls, inner_lr, k_inner_steps, ...):
        # 1. 将模型参数按层 (nn.Module) 分组
        self.param_groups = self._group_by_layer(params)
        # 2. 为每个参数组初始化一个独立的内部优化器 (如 AdamW, Muon)
        self.inner_optimizers = [inner_optimizer_cls(group['params'], lr=inner_lr) for group in self.param_groups]
        ...

    @torch.no_grad()
    def step(self, closure):
        # closure() 返回主损失 L_main
        L_main = closure()
        # 创建高阶图以计算后续的 JVP/VJP
        L_main.backward(create_graph=True)

        # 外部循环：遍历模型的每一层
        for i, group in enumerate(self.param_groups):
            params_l = group['params']
            inner_optimizer = self.inner_optimizers[i]

            # d 是该层的更新方向，作为内部优化器的可学习参数
            # 注意：d 在 PyTorch 中需要被注册为参数才能被优化器更新
            update_direction_d = [torch.zeros_like(p, requires_grad=True) for p in params_l]
            # 将 d 交给内部优化器管理
            inner_optimizer_for_d = inner_optimizer_cls([{'params': update_direction_d}], lr=self.inner_lr)

            # 内部循环：在二次近似损失上进行 k 步优化
            for _ in range(self.k_inner_steps):

                # 定义并计算二次近似损失 L_quad_l(d)
                # L_quad_l(d) = g_lᵀd + 0.5 * dᵀG_l d

                # 1. 计算一阶项: g_lᵀ * d
                grad_term = torch.sum(torch.stack([torch.sum(p.grad * d_p) for p, d_p in zip(params_l, update_direction_d)]))

                # 2. 计算二阶项 (GNVP): 0.5 * dᵀ * (G_l * d)
                gnvp = self._calculate_gnvp(L_main, params_l, update_direction_d)
                quadratic_term = 0.5 * torch.sum(torch.stack([torch.sum(gnvp_p * d_p) for gnvp_p, d_p in zip(gnvp, update_direction_d)]))

                L_quad_l = grad_term + quadratic_term

                # 使用内部优化器在 L_quad_l 上进行一步更新，求解 d
                inner_optimizer_for_d.zero_grad()
                L_quad_l.backward() # 计算 L_quad 对于 d 的梯度
                inner_optimizer_for_d.step() # 更新 update_direction_d

            # 将最终求解出的更新方向 d 应用到该层的真实参数上
            for p, d_p in zip(params_l, update_direction_d):
                p.add_(d_p.detach(), alpha=-self.lr) # 使用全局学习率
```

## 4. 理论意义与工程挑战

### 4.1. 理论意义

- **FEP 的直接实现**: `AdaGNoema` 是对“FEP 等价于二阶优化”这一论断的直接、无妥协的工程实现。
- **消除启发式**: 它彻底消除了所有关于“信念”、“惊奇”的启发式代理，完全依赖于从损失函数曲率中计算出的、有原则的二阶信息。
- **与 G-CAFM 的协同**: `AdaGNoema` 不仅是一个优化器，它还是 `G-CAFM` 框架的信息来源。在计算 `L_quad_l` 的过程中，它可以缓存并暴露 `diag(G_l)`，为模型的动态路由提供 principled 的重要性信号。

### 4.2. 工程挑战

- **协同执行**: 正如 `ADR-0002` 所述，`AdaGNoema` 与 `G-CAFM` 的结合将打破 PyTorch 的标准解耦范式。优化器 (`AdaGNoema`)、模型 (`G-CAFM`) 和反向传播 (`backward`) 必须以一种紧密耦合的方式协同工作。
- **`torch.func` 的精通**: 无矩阵 GNVP 的实现高度依赖对 `torch.func` 的深入理解，需要精确控制计算图的创建和销毁，以确保梯度流的正确性和计算效率。
- **性能开销**: 内部优化循环和高阶梯度计算会带来显著的计算开销。这需要在理论上的最优性与工程上的可行性之间做出权衡。

## 5. 初步结论

`AdaGNoema` 代表了 `Tiny-ONN` 项目在优化理论上的一个重大范式转变。它标志着我们从设计“智能架构”转向设计“智能优化器”，将 FEP 的核心原则封装为一个更通用、更强大的学习引擎。虽然工程挑战巨大，但其理论上的完备性和潜力使其成为我们下一阶段的核心研究方向。
