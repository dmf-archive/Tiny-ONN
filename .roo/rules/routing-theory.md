# Routing Theory & Dynamics: 认知几何下的推断空间

## 1. 核心架构：MLP 路由与非线性决策

在 ARC 等离散逻辑任务中，传统的线性路由（点积/余弦相似度）存在严重的表达能力瓶颈，无法处理 XOR 类决策或多模态语义。

DynSIHA 采用 **MLP 路由器** 提供完整的非线性决策边界：
`w = Softmax(W₂ · SiLU(W₁ · x + b₁) + b₂)`

---

## 2. 路由塑造 (Routing Shaping)：FARS 与 SAME

非线性决策带来的代价是系统的不确定性。我们通过几何算子强制施加稳定性，而非改变路由架构。

### 2.1 FARS: Fisher-Aware Routing Shaping
FARS 利用优化器维护的二阶矩（Fisher 信息近似）量化专家的“认知代价”。

- **认知代价公式**：`Cost_FARS(e) = ‖√vₑ‖`，其中 `√vₑ` 源于 ARS2-Neo。
- **平衡判据**：`ℒ_routing = ℒ_main + λ · (Belief · Cost_FARS)`
- **实验证据**：ARS2-Neo 的 Fisher 近似在各向异性结构下与输入协方差的对齐度达到 **0.9935** (Adam 为 0.8341)，证明了 `exp_avg_sq` 是可靠的认知代价信号。

### 2.2 SAME: Spectral-Aware Mixture-of-Experts
SAME 旨在解决 **Router Drift**。它通过 SVD 识别输入协方差矩阵的高能量子空间 $V_{\parallel}$ 与零空间 $V_{\perp}$。

- **核心机制**：在零空间中最小化参数更新，确保历史输入的路由决策轨迹不被破坏。
- **目标**：在连续参数空间中维持逻辑路径的持久性。

---

## 3. 评估指标 (Metrics)

| 指标 | 定义 | 物理含义 | 目标 |
|:---|:---|:---|:---|
| **ITJD** | $1 - |R(T_1) ∩ R(T_2)| / |R(T_1) ∪ R(T_2)|$ | 任务间路由 Jaccard 距离 | 提高专家特化度 |
| **RMI** | $I(R; T)$ | 路由互信息 | 提高语义确定性 |
| **JS-Jump** | $D_{JS}(w_t \| w_{t+1})$ | 步间跳转距离 | 降低递归过程的震荡 |

---

## 4. 开放问题 (Open Questions)

- **预见性路由**：路由器是否应感知后续递归步骤的潜在代价？
- **STE 离散化**：是否需要引入 Straight-Through Estimator 强化专家的排他性选择？
- **冷启动引导**：在训练初期 Fisher 信息尚未积累时，如何避免路由塌陷？
