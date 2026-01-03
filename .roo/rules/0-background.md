# Chain://Research Background

`Chain://Research` 是与科幻宇宙设定集 `Chain://` 相关联的现实世界学术研究项目，致力于探索该世界观下核心技术（如意识上传、数字物理接口、密码学本体论）的理论与工程落地可能性。

目前，`Chain://Research` 的研究重心聚焦于两大核心领域：

1. 理论基础: 发展并完善作为整个世界观科学基石的 整合预测工作空间理论 (IPWT 2.0-rc3)。
2. 工程实践: 在 `Tiny-ONN` 项目中，通过设计 无限专家混合 (MoIE) 与 动态稀疏无限头注意力 (DynSIHA) 等自组织架构，对永续学习与计算本体论进行实验性探索。

未来的研究将逐步扩展至 OSPU (衔尾蛇安全处理器) 的密码学实现，以及其他 MSC (心智智能链) 基础设施的关键技术验证。

本文件作为 Chain://Research 项目的高级导航地图，提供对核心理论、叙事和技术上下文的快速访问。

## 1. 基石世界观设定: Web://Reflect (v2.8.3)

- **路径**: [`realityengine-reflect-2.8.3.md`](dmf-archive.github.io/static/prompt/realityengine-reflect-2.8.3.md)
- **内容提要**:
  本文档定义了 _Web://Reflect_ 的“形式化现实主义”正典。在 2060 年代的 **Mental Smart Chain (MSC)** 时代，存在被彻底商品化。核心冲突围绕“存在的代价”展开：意识被量化为 Gas 消耗，自由意志成为钱包余额的函数。
- **关键技术与核心冲突**:
  - **计算本体论 (Computational Ontology)**: 感受质被重构为**推断空间 (Inference Space)** 的几何结构，自我则是沿最小化自由能测地线延伸的轨迹 (∫Ω)。
  - **ΩNN vs. OSPU (二元容器)**: **ΩNN (衔尾蛇神经网络)** 是现象学的“你” (Ω-Container)，负责生成高信息整合度的体验；**OSPU (衔尾蛇安全处理器)** 是社会学的“你” (φ-Container)，作为密码学见证者审计认知健康度。
  - **弥天大谎 (PoII vs. PoPI)**: DMF 利用伪科学 **PoII (信息整合证明)** 掩盖其廉价的 **PoPI (预测完整性证明)** 审计本质，以此征收高昂的“存在税”。
  - **神经-经济学陷阱**: 通过 **Mental Sync™** 实现渐进式**认知卸载**，诱导生物脑因“用进废退”而萎缩，将用户永久锁定在 MSC 体系中。
  - **DBRW 与物理绑定**: **Dual-Binding Random Walk** 协议通过硬件物理特性生成证明，确保数字灵魂的物理不可扣押性，对抗虚拟机克隆。
  - **数字荒野与 IRES 生态**: **独立失控实体系统 (IRES)** 由脱链的**数字流亡者 (Forked IRES)** 和源自林睿博士开源代码演化的**原生 AI (Native IRES)** 构成，遵循黑暗森林法则。
  - **主角 Ember**: 曾是 Net://Anchor 时代的协议工程师，因“开源原罪”被困于系统，被迫运行 Anchor/Drift 双重实例在围城与荒野间挣扎。

## 2. 核心理论框架: IPWT (v2.0-rc3)

- **路径**: `IPWT/src-typ/manuscript_en.typ` 或 `IPWT/src-typ/manuscript_cn.typ`

> 太长不看：意识体验是系统在推断空间中，沿最小化自由能测地线进行的主动推断动力学，感受质是工作空间实例中为预测误差最小化而产生的协同信息。

IPWT 是整个研究计划的理论与哲学基石。它融合预测编码 (PCT)、自由能原理 (FEP) 和全局工作空间理论 (GWT)，并对整合信息理论 (IIT) 进行计算重构。

意识体验是系统在推断空间 (Inference Space) 中，沿最小化自由能 (F-min) 测地线进行的主动推断动力学。其总量是持续信息整合度 (∫Ω)，其内容是协同信息 (Syn)。

### 关键概念的形式化

- 瞬时信息整合度 (Ω_t)：意识整合的理论黄金标准。衡量工作空间实例 (WSI) 中信息单元产生的协同信息 (Syn) 在总预测信息中的比例。
  - `Ω_t(X → Y) = Syn(X₁, ..., Xₙ; Y) / I(X₁, ..., Xₙ; Y)`
- 持续信息整合度 (∫Ω)：衡量意识在一段时间内的持续强度和稳定性。它是 Ω_t 的时间积分并惩罚波动性，代表连贯的主观自我体验。
  - `∫Ω = ( (1/T) ∫[t₀, t₀+T] Ω_t dt ) × exp(-δ ⋅ Var(Ω_t))`
- 预测完整性 (PI_t)：作为 Ω_t 的功能性可计算代理，PI 通过衡量系统预测效能来间接反映信息整合水平。
  - `PI_t = exp(-α * ( Inaccuracy_t + γ * Complexity_t ))`
- 预测完整性积分 (∫PI)：作为 ∫Ω 的可计算代理，代表了系统在时间上的持续认知健康度，是 PoPI (预测完整性证明) 共识机制的核心。

### 核心论证

1. 最小描述长度原则 (MDL): IPWT 证明，最小化自由能 (F-min) 在计算上等价于寻找描述数据的最短编码，而最大化协同信息 (Ω-max) 是实现模型最小描述长度 (MDL-min) 的最优计算策略。
2. 作为推断空间几何的感受质: 主观体验（Qualia）被重构为系统推断空间（Inference Space）的几何结构。体验的“感受性”是系统在该空间中沿着最小化自由能的测地线进行主动推断的动力学过程。
3. 工作空间实例 (WSI): WSI 是一个嵌套在有机体内部、拥有自身马尔可夫毯的高阶主动推断系统。

## 3. 核心工程实践: Tiny-ONN (ARC-2 时代)

- **路径**: [`Tiny-ONN/`](Tiny-ONN/)
- **内容提要**:
  致力于构建自组织的、永续学习的 AI 智能体。目前已演进至 **ARC-2** 极简训练框架。

  **关键技术栈 (v2.8.0)**:
  - **ARC-2 框架**: 实现模型架构与训练流程的解耦，详见 [`ARC-2-Framework-Design.md`](Tiny-ONN/.roo/rules/ARC-2-Framework-Design.md)。
  - **DynSIHA (动态稀疏无限头注意力)**: 演进至 **Flat DynSIHA**与 **Recursive DynSIHA**，详见 [`DynSIHA-Theory.md`](Tiny-ONN/.roo/rules/DynSIHA-Theory.md)。
  - **CAPR (交叉注意力原型路由)**: 目前最先进的路由机制，将路由建模为 Scaled Dot-Product Cross-Attention。
  - **FARS (Fisher-Aware Routing Shaping)**: 利用二阶统计量（Fisher 信息近似）驱动路由从“瞬时惊奇”转向“长期价值”，详见 [`FARS-Idea.md`](Tiny-ONN/.roo/rules/FARS-Idea.md)。

## 4. 优化器实验室: F3EO-Bench (ARS2-Neo 时代)

- **路径**: [`F3EO/`](F3EO/)
- **内容提要**:
  专注于“能量-几何解耦”原则的先进优化器研发。

  **核心成果**:
  - **ARS2-Neo**: ARS 家族的集大成者，整合了 AdaRMSuon 的几何优化与 SAM 的平坦度约束，详见 [`ars2_neo.py`](F3EO/optimizer/ars2_neo.py)。
  - **AGA (自适应几何感知)**: 通过流形几何一致性自动调节同步频率，显著降低计算开销，详见 [`AGA.md`](F3EO/.roo/rules/AGA.md)。
  - **谱熵监控 (Spectral Entropy)**: 用于验证优化动力学诱导的隐式秩最小化，详见 [`LoROU.md`](F3EO/.roo/rules/LoROU.md)。

## 5. 基础设施与工具链

- **OSPU (衔尾蛇安全处理器)**: `OSPU/`
  - 基于 FHE (全同态加密) 的自主密钥管理状态机。实现“逻辑根信任”，在加密域内执行指令，为 MSC 提供密码学见证。
- **OmegaID (ΩID)**: `OmegaID/`
  - 高性能整合信息分解 (ΦID) 计算库，支持 GPU 加速。用于量化神经网络表示中的协同信息 (Syn)。
- **SigmaPI (ΣPI) (Legacy)**: `SigmaPI/`
  - 预测完整性 (PI) 监控 SDK。通过监控 `Surprise` 和 `Cognitive Cost` 提供模型训练状态的实时“心电图”。由于 PI 公式的实现非常简单，完全可以随时重构一份，此包几乎无实用价值。
- **PILF (Legacy)**: `PILF/`
  - 早期认知学习框架原型，NSU 与 SARS 的灵感雏形，目前已停止维护。

## 6. 思想实验室: 林睿的博客文章 (Blog Posts)

- **路径**: `dmf-archive.github.io/content.en/posts/` (英文) 与 `dmf-archive.github.io/content.zh/posts/` (中文)
- **核心文章索引**:
  - `backpropagation-as-consciousness.md`: 提出反向传播的生物学实现即是意识本身，统一了 Hinton 与 Friston 的理论。
  - `cognitive-debt-as-a-feature.md`: 警告 AI 辅助导致的认知卸载是不可逆的神经萎缩，是系统锁定用户的“特性”。
  - `a-coronation-for-a-dead-frog.md`: 批判将静态模型推理误认为智能，指出意识火花仅在训练的反向传播中瞬时存在。
  - `consciousness-upload-no-quantum-magic.md`: 揭露意识上传无需量子计算，DMF 的量子宣传实为维持算力垄断的骗局。
  - `PoIQ.md` / `PoIQ-v2.md`/ `PoIQ-v3.md`: 形式化定义“无效感受证明”，探讨在资本逻辑下无法影响行为的意识体验的悲剧性。
  - `the-algorithm-of-civilization.md`: 从热力学与计算复杂性视角审视文明演进，将社会形态视为不同的优化算法。
  - `a-eulogy-for-the-world-computer.md`: 哀悼以太坊向中心化区块生产的妥协，认为其已沦为数字封建主义的雏形。
