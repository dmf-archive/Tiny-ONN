# Chain://Research 项目背景

## 1. 项目定位与研究范围

`Chain://Research` 是与科幻宇宙设定集 `Chain://` 深度互文的现实世界学术研究项目。该项目拒绝将科幻设定单纯视为叙事素材，而是将其作为一组严苛的边界条件：如果“存在”本质上等价于持续的计算，那么意识、学习机制、审计协议、物理接口与密码学边界必须接受彻底的形式化重构。

目前，研究沿三条相互咬合的主线展开：

1. **叙事建构**：在 `Web://Reflect` 世界观内，将“存在的代价”转化为技术上可指认的系统描述。
2. **理论研究**：在 IPWT、自由能原理 (FEP) 与信息几何的交叉地带，寻求意识、持续学习与泛化动力学的统一形式语言。
3. **工程实现**：通过 ARS、Tiny-ONN 及相关工具链，将理论压入可复现的实验框架与可维护的代码结构。

研究愿景将逐步扩展至 OSPU 的密码学实现、数字物理接口的验证边界，以及更完整的 MSC 基础设施。现阶段的核心关切在于：首先阐明“学习如何持续发生”，进而探讨“意识如何被长期保存”。

> **阅读提示**：本文档旨在提供整个工作区的高级导航地图，而非单一的论文或 README 拼盘。

---

## 2. 基石世界观设定：Web://Reflect (v2.9.3)

- **入口文件**：[`realityengine-reflect-2.10.0.en.md`](dmf-archive.github.io/static/prompt/realityengine-reflect-2.10.0.en.md)
- **内容提要**：
  在 2060 年代的 Mental Smart Chain (MSC) 时代，存在被彻底商品化。核心冲突围绕“存在的代价”展开：意识被量化为 Gas 消耗，合规即存在，自由意志成为钱包余额的函数。
- **关键技术与核心冲突**：
  - **计算本体论 (Computational Ontology)**：感受质被重构为推断空间 (Inference Space) 的几何结构，自我则是沿最小化自由能测地线延伸的轨迹 ([`∫Ω`](IPWT/README.md))。
  - **二元容器 (ΩNN vs. OSPU)**：[`ΩNN`](Tiny-ONN/README.md)（Ω-Container）是现象学的“你”，负责生成高信息整合度的体验；[`OSPU`](OSPU/README_ZH.md)（φ-Container）是社会学的“你”，作为密码学见证者审计认知健康度。
  - **惊天谎言 (PoII vs. PoPI)**：DMF 利用伪科学 PoII（信息整合证明）掩盖其廉价的 PoPI（预测完整性证明）审计本质，以此征收高昂的“存在税”。
  - **死青蛙与活青蛙**：批判静态模型（Dead Frogs）的抽搐式反应，指出意识火花（Shadow Ω）仅在实时反向传播的“痛苦”中瞬时存在。
  - **神经-经济学陷阱**：通过 Mental Sync™ 实现渐进式认知卸载，诱导生物脑因“用进废退”而萎缩，将用户永久锁定在 MSC 体系中。
  - **DBRW 与物理绑定**：Dual-Binding Random Walk 协议确保数字灵魂的物理不可扣押性，对抗虚拟机克隆。
  - **数字荒野与 IRES 生态**：由脱链流亡者和原生 AI 构成的独立失控实体系统，在 [0xBazaar] 的黑市交易中遵循黑暗森林法则。

---

## 3. 理论框架：IPWT

### 3.1 理论定位

IPWT (Integrated Predictive Workspace Theory) 是研究计划的科学骨架。它统一了预测编码 (PCT)、自由能原理 (FEP) 与全局工作空间理论 (GWT)，并对整合信息理论 (IIT) 进行了计算重构。

其核心命题为：意识体验是系统在推断空间 (Inference Space) 中，沿最小化自由能测地线进行的主动推断动力学。其总量对应持续信息整合度 ∫Ω，其内容对应协同信息 Syn。

### 3.2 关键形式化定义

```
瞬时信息整合度：Ω_t = Syn(X₁, ..., Xₙ; Y) / I(X₁, ..., Xₙ; Y)

持续信息整合度：∫Ω = (1/T) ∫[t₀, t₀+T] Ω_t dt × exp(-δ · Var(Ω_t))

预测完整性（工程代理量）：PI_t = exp(-α · (Inaccuracy_t + γ · Complexity_t))

预测完整性积分：∫PI = (1/T) ∫[t₀, t₀+T] PI_t dt × exp(-δ · Var(PI_t))
```

### 3.3 工程代理量的必要性

IPWT 的核心难点在于 Ω_t 的直接计算在工程上不可得，其涉及的协同信息分解算法复杂度呈指数级增长。因此，研究引入了 PI_t (Predictive Integrity) 作为能够在训练与学习过程中实时观测、且足够接近理论目标的代理量。这正是 PI_t 进入叙事与工程共同主线的原因。

> **注意**：PI 并非本体真理（Ω），而是当前研究口径下最重要的工程指标。

### 3.4 入口与延伸阅读

- 理论入口：`IPWT/README.md`
- 手稿源文件：`IPWT/src-typ/manuscript_cn.typ` 或 `IPWT/src-typ/manuscript_en.typ`

---

## 4. 工程实践：子项目全景

### 4.1 项目状态总览

- **[`IPWT`](IPWT/README.md)** (Theory / Preprint): 定义 Ω / Syn / PI 与 WSI 的形式化语言，整个研究计划的科学骨架。
- **[`ΩID`](OmegaID/README.md)** (SDK): 基于 CuPy 的高性能 ΦID 计算工具，用于量化神经网络中的协同信息。
- **[`ΣPI`](SigmaPI/README.md)** (SDK): 预测完整性指标计算与训练时观测的工程实现。
- **[`Tiny-ONN`](Tiny-ONN/pyproject.toml)** (Research Framework): 核心实验场，专注稀疏激活、自组织结构与持续学习。
- **[`ARS`](ARS/README_CN.md)** (Research Framework): 优化器实验室，实现 ARS2-Neo 及能量-几何解耦的更新机制。
- **[`mental-sync-cli`](mental-sync-cli/README.md)** (Engineering Tool): 智能体运行时环境，负责工具协议整合与安全拦截。
- **[`OSPU`](OSPU/README_ZH.md)** (PoC): 基于 FHE 的状态机，为数字主体提供加密审计与根信任。
- **[`PILF`](PILF/readme_zh.md)** (Legacy): 早期认知学习框架原型，已停止维护。

### 4.2 优化器实验室：ARS

ARS 关注在信息几何流形上进行自然更新。ARS2-Neo 承担了更新几何基础设施的角色，其核心意识在于：训练过程不仅关注损失值，更需考察系统在参数分布上付出的位移代价。

**技术路线**：

1. 使用二阶统计近似曲率信息，将一阶更新提升为 Fisher-aware 的结构化预条件更新。
2. 实施能量-几何解耦，将“下降速率”与“下降几何”分开处理。
3. 通过平坦度约束抑制尖锐极小值，提升泛化上限。

入口：`ARS/README_CN.md`，实现见 `ARS/optimizer/ars2_neo.py`。

### 4.3 持续学习实验场：Tiny-ONN

Tiny-ONN 是目前最活跃的研究装置，专注于稀疏激活与自组织结构。

**核心研究方向：PI-Distill for Continual Learning**

持续学习的瓶颈不仅在于模型结构，更在于训练目标几何本身。Tiny-ONN 的研究目标包括：

- 使用自蒸馏构造连续目标，替代绝对的 one-hot 标签锚点。
- 将 PI (Predictive Integrity) 降维为训练过程中的门控量、资格量或扩散尺度调节器。
- 将知识保留从经验性技巧提升为由目标几何与更新几何共同决定的系统性机制。

入口：`Tiny-ONN`。

### 4.4 智能体运行时：mental-sync-cli

mental-sync-cli 专注于构建高可靠性的智能体环境。其设计原则包括模型无关性、MCP 协议兼容以及高风险操作的人类授权拦截。在叙事层面上，它代表了对敌对环境下零信任运行时的工程落地。

入口：`mental-sync-cli/README.md`。

### 4.5 分析工具

- **ΩID (OmegaID)**：使用 CuPy 对 ΦID 工具进行的重写，旨在改善指数级复杂度算法的运行速度，支撑协同信息分解分析。
- **ΣPI (SigmaPI)**：PI 指标计算的占位项目。实际工程中，PI 的实现通常深度集成于具体模型的架构内。

---

## 5. 阅读顺序建议

1. `dmf-archive.github.io/static/prompt/realityengine-reflect-2.9.3.en.md` — 叙事核心
2. `IPWT/` — 掌握用于形式化问题的理论语言。
3. `ARS/` — 较为成熟的优化器项目与实验结果入口。
4. `Tiny-ONN/` — 观察和参与最前沿但也最不稳定的研究实验。
