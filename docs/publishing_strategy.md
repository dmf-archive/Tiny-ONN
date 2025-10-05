# Tiny-ONN 出版策略与论文大纲 (草案)

**核心策略**: 优先完成 ARC 奖项所需的论文（论文 2），同步推进实验以支持其论点。在 ARC 截止日期后，回头系统性地整理和发表关于理论框架（论文 1）的论文。

---

## 论文 1 (后续发表): 统一理论框架

- **拟定标题**: _Sparse Proto-Routing as Variational Free Energy Minimization: A Unified Framework for Deep Active Inference_
- **目标期刊/会议**: TMLR, ICLR, Nature Communications, NeuralPS
- **摘要 (Abstract)**:
  > We present a unified framework that operationalizes the Free-Energy Principle (FEP) for deep neural networks through a novel mechanism termed Sparse Proto-Routing. This framework treats meta-learning as an active inference process, where the network minimizes a tractable proxy for variational free energy by dynamically selecting and composing computational pathways. We introduce the Sparse Proto Linear (SPL) layer, a self-organizing building block that decouples parameters into a computational core, a perceptual pattern matcher, and a routing gate. The routing decisions are guided by a meta-learning objective, Surprise-Aware Routing Shaping (SARS), which aligns the network's routing distribution with an ideal expert utility distribution derived from the main task's gradient signals. This dual-optimizer architecture allows the agent to rapidly adapt its computational strategy (inference) while stably updating its world model (learning). We demonstrate that this approach not only provides a first-principles derivation for dynamic sparse computation but also naturally mitigates catastrophic forgetting, paving the way for more general and adaptive AI systems.
- **大纲 (Outline)**:
  1. **Introduction**: The challenge of building adaptive and continually learning AI. Introduce the Free-Energy Principle (FEP) as a guiding theory and propose our work as its computational operationalization.
  2. **Theoretical Foundation: From FEP to Surprise-Aware Routing Shaping (SARS)**:
     - 2.1. Variational Free Energy as the Objective.
     - 2.2. "Surprise" (Gradient Norm) as a Signal for Inference.
     - 2.3. SARS: Framing Meta-Learning as Distribution Alignment between routing policy `P` and expert utility `Q`.
  3. **The Computational Substrate: Sparse Proto Linear (SPL)**:
     - 3.1. Decoupling Computation, Perception, and Action (Routing).
     - 3.2. SPL as a VQ-VAE Analogue: The role of `mu`, `proto`, and `gate`.
  4. **The Unified Framework in Action**:
     - 4.1. The Dual-Optimizer Action-Perception Loop.
     - 4.2. How Sparse Proto-Routing Minimizes Free Energy.
     - 4.3. Emergent Properties: Dynamic sparsity, mitigation of catastrophic forgetting.
  5. **Experiments**: Validation on synthetic tasks demonstrating the core properties of the framework.
  6. **Conclusion**: Proposing this framework as a principled path towards deep active inference.

---

## 论文 2 (优先发表, 目标 ARC 奖): 应用与涌现

- **拟定标题**: _Emergent Differentiable Program Search for Abstract Reasoning in Transformers_
- **目标**: ARC Prize Paper Reward
- **摘要 (Abstract)**:
  > The Abstraction and Reasoning Corpus (ARC) presents a grand challenge for modern AI, requiring the ability to discover and apply abstract, algorithmic rules from a few examples. We present a novel Transformer-based architecture, built upon self-organizing Sparse Proto Linear (SPL) layers, that demonstrates the ability to solve a significant subset of ARC tasks through a process we term "emergent differentiable program search." Our model, guided by a meta-learning framework inspired by the Free-Energy Principle, dynamically composes specialized sub-networks (effective experts) for each input token, effectively learning to route information and synthesize computational functions on-the-fly. We introduce the "Key-in-Lock" optimization model to describe the unique, non-smooth loss landscape of ARC, and justify our "Single-View Grokking" training strategy. Through analysis of the learned sparse activation patterns and memory-based gate parameters, we show that our model learns to solidify discrete, reusable computational pathways analogous to symbolic programs, offering a new perspective on bridging the gap between connectionist and symbolic AI.
- **大纲 (Outline)**:
  1. **Introduction**: ARC 的挑战性及其对抽象推理能力的要求。现有方法的局限性。提出我们的核心论点：通过梯度下降，可以在一个专门设计的计算基质中涌现出可微的程序。
  2. **A Self-Organizing Computational Substrate**:
     - 2.1. Dynamic Function Composition with SPL.
     - 2.2. MoIE and DynSIHA: Fully dynamic Transformer blocks.
     - 2.3. Prototype Residual Connections for information flow.
  3. **Training Dynamics and Optimization**:
     - 3.1. The "Key-in-Lock" Model of the ARC Loss Landscape.
     - 3.2. Single-View Grokking: A strategy for pure gradient signals.
     - 3.3. PIMax: Guiding self-organization through surprise minimization.
  4. **Results on ARC-AGI**:
     - 4.1. Quantitative Results: 展示在 ARC-AGI-2 评估集上的性能，并与其他 SOTA 方法进行对比。
     - 4.2. Qualitative Analysis: 可视化具体任务的求解过程，展示动态生成的稀疏计算图。
     - 4.3. Causal Solidification: 分析 `proto` 的演化，展示模型如何“记住”并固化成功的计算路径。
  5. **Conclusion**: 总结我们的方法作为一种新的“可微程序搜索”范式的潜力，并讨论其对未来 AGI 研究的启示。
