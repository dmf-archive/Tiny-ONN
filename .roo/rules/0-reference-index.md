# Tiny-ONN 项目索引 (Reference Index)

本索引用于梳理 Tiny-ONN 仓库中的实验性组件及其当前状态。

## 1. 核心优化器 (SOTA)

目前项目最稳定且性能最强的组件位于 `F3EO` 仓库。

- **ARS (AdaRMSuon SAM)**: `../F3EO/optimizer/ars.py`
  - **状态**: **生产就绪 (Stable)**。
  - **描述**: 在 AdaRMSuon 的基础上引入了流形感知扰动（SAM），解决了极速收敛带来的过拟合问题。是目前所有实验的首选优化器。
- **AdaRMSuon**: `../F3EO/optimizer/ada_rmsuon.py`
  - **状态**: 稳定。
  - **描述**: 实现了能量-几何解耦，通过 Newton-Schulz 正交化提供极高的收敛效率。

## 2. 架构实验 (Tiny-ONN/exp)

`Tiny-ONN` 目录下的架构大多处于早期原型或已搁置状态，需谨慎复用。

### 2.1 递归与动态合成 (Generation 3)

- **DynTRM / Recursive DynSIHA**: `./exp/arc_dyntrm/model.py`
  - **状态**: **实验中 / 已搁置**。
  - **问题**: 递归深度导致梯度不稳定，且在现有硬件上存在严重的访存瓶颈。
  - **价值**: 其中的 `CAPR` (交叉注意力路由) 逻辑具有研究价值。

### 2.2 动态 MLP 架构 (Generation 2)

- **ArcDynSNT (MLP-Attention + DynMoE)**: `./exp/arc_dyn_snt/model.py`
  - **状态**: **原型 (Prototype)**。
  - **描述**: 尝试用 MLP 增强 QKV 生成，并引入动态 MoE。尚未在大规模任务上验证稳定性。

### 2.3 稳定基线 (Generation 4)

- **ArcRMSuon**: `./exp/arc_rmsuon/model.py`
  - **状态**: **基线 (Baseline)**。
  - **描述**: 标准 Qwen3 结构 + RMSuon 优化器。对于ARC任务来说几乎无法有效学习。搁置。

## 3. 路由演进史 (Routing Evolution)

尽管 SARS 动力学在多个版本中均告失败，但路由机制的工程实现具有重要的参考价值。

- **CPR (Cosine Attention Prototype Routing)**: `./exp/arc_dyn_snt/model.py:45`
  - **机制**: 基于余弦相似度的原型匹配，结合阈值实现稀疏激活。
  - **评价**: DynMoE 的方案，梦开始的地方。
- **CAPR (Prototype Cross Attention Routing)**: `./exp/arc_dyntrm/model.py:50`
  - **机制**: 缩放点积交叉注意力路由。
  - **评价**: 将路由提升为注意力机制，显著增强了内容感知能力。虽然配合 SARS 训练依然困难，但作为架构组件是目前的最优解。退而求其次来说，可以考虑原版 DynMoE 的 SDL 辅助损失训练。

## 4. 理论文档

- **IPWT 2.0**: `./IPWT/` - 意识与信息整合的理论基石。
- **NSU Insight**: `../F3EO/.roo/rules/NSU.md` - 关于原生稀疏更新的最新理论洞察。
