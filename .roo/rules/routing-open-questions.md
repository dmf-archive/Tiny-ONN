## 摘要：MoE 路由塑造公式剖析

### 1. 讨论背景

- 目标：分析 FARS 路由塑造公式的理论正确性
- 对比对象：常规负载均衡损失 vs Bias-based Loss-Free Balancing

### 2. Loss-Free Balancing 核心机制（DeepSeek, 2024）

- **公式**：`g_i,t = s_i,t`（若 `s_i,t + b_i ∈ TopK`），否则为 0
- **关键设计**：bias 仅影响 Top-K 选择决策，不影响被选中后的 gating weight
- **更新规则**：`b_i = b_i + u · sign(c̄ - c_i)`，`u=0.001` 最优
- **优势**：bias 在梯度流之外更新，无干扰梯度
- **实验结果**（1B/3B 模型）：
  - Perplexity：9.50 vs 9.56（Loss-Controlled）
  - MaxVio_global：0.04 vs 0.72（18 倍更均衡）

### 3. FARS 机制分析

- **公式**：`ℒ_FARS = Σₑ Beliefₑ · Cost_FARS(e)`，`Cost_FARS(e) = ‖√vₑ‖`
- **信号来源**：Adam 优化器二阶矩，近似 Fisher 信息对角线
- **语义**：量化专家参数对损失的敏感度/学习活跃度
- **核心争议**：
  - 量纲不匹配：Fisher 代价（梯度 RMS）vs 频率（token 计数）
  - 目标区分：外部目标（硬件利用率）vs 内在目标（认知利用率）
- **结论**：FARS 作为辅助损失有理论必要性，与 Loss-Free 目标正交

### 4. DynMoE 方案对比（ICLR 2025）

- **路由公式**：`s(x) = ⟨x, Wg⟩ / (‖x‖ · ‖Wg‖)`（余弦相似度）+ 可训练阈值 `G`
- **决策边界**：线性超平面
- **与 MLP 路由对比**：
  - DynMoE：线性，有几何意义，可解释性强
  - DynSIHA：MLP 非线性，可表达 XOR 类决策边界
- **关键洞察**：可训练阈值 G 本身是 Loss-Free 风格的负载均衡机制

### 5. 未决理论问题

- 原型路由天然是线性的，引入非线性变换后失去几何意义
- 核心问题：Transformer 架构中路由是否需要非线性表达力？
- 假设：线性路由可能通过反向传播自动将路由决策逼近 low-rank 线性超球面，提供 MDL 认知压力

---

## 参考文献

1. Wang, L., Gao, H., Zhao, C., Sun, X., & Dai, D. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts. *arXiv:2408.15664*. <https://arxiv.org/abs/2408.15664>

2. Guo, Y., Cheng, Z., Tang, X., Tu, Z., & Lin, T. (2025). Dynamic Mixture of Experts: An Auto-Tuning Approach for Efficient Transformer Models. *ICLR 2025*. arXiv:2405.14297. <https://arxiv.org/abs/2405.14297>

3. Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. *JMLR, 23*(120):1-39.

4. Lepikhin, D., et al. (2020). GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. *arXiv:2006.16668*.
