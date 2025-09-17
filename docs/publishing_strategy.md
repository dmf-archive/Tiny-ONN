# Tiny-ONN 出版策略与论文大纲 (草案)

**核心策略**: 优先完成 ARC 奖项所需的论文（论文 3），同步推进实验以支持其论点。在 ARC 截止日期后，回头系统性地整理和发表关于底层架构（论文 1）和宏观理论框架（论文 2）的论文。

---

## 论文 1 (后续发表): 基础架构

- **拟定标题**: _Sparse Proto Linear (SPL): A Self-Organizing Building Block for Dynamic Neural Computation_
- **目标期刊/会议**: ICLR, NeurIPS
- **摘要 (Abstract)**:
  > We introduce Sparse Proto Linear (SPL), a novel, self-organizing linear layer designed to replace standard dense layers in neural networks. SPL decouples its parameters into a computational core (`mu_weight`), a perceptual pattern matcher (`proto_weight`), and a memory-based gate (`gate_param`). By employing a pure content-addressing mechanism via normalized cosine similarity and a memory-driven meta-learning rule (MSAPS), SPL can dynamically compose specialized computational functions for each input. Our formulation includes a parameter-free adaptive learning mechanism that effectively balances knowledge preservation and plasticity, mitigating catastrophic forgetting. We demonstrate that SPL serves as a robust and efficient building block for creating dynamic, sparse, and continually learning neural architectures.
- **大纲 (Outline)**:
  1. **Introduction**: 现有静态、同质化神经网络层的局限性。引出对动态、自组织计算的需求。
  2. **From SBL to SPL**: 介绍从稀疏贝叶斯线性层 (SBL) 到稀疏原型线性层 (SPL) 的演进，重点阐述为解决梯度冲突和灾难性遗忘所做的改进。
  3. **The Sparse Proto Linear (SPL) Layer**:
     - 3.1. Parameter Decoupling: `mu`, `proto`, and `gate` auras.
     - 3.2. Pure Content-Addressing: The role of L2 normalization in eliminating norm pollution.
     - 3.3. Memory-Surprise-Aware Prototype Shaping (MSAPS): The core meta-learning dynamic.
     - 3.4. Adaptive Dynamics: Inertia and Decay weights for parameter-free learning rate and weight decay adaptation.
  4. **Experiments**: 在合成任务上验证 SPL 的基本属性（稀疏性、抗遗忘、自组织聚类）。
  5. **Conclusion**: 总结 SPL 作为下一代神经网络基础构建块的潜力。

---

## 论文 2 (后续发表): 理论框架

- **拟定标题**: _Predictive Integrity Maximization: A Computational Framework for Meta-Learning Inspired by the Free-Energy Principle_
- **目标期刊/会议**: TMLR, Cognitive Science, Nature Communications
- **摘要 (Abstract)**:
  > Inspired by the Free-Energy Principle (FEP) in neuroscience, we propose Predictive Integrity Maximization (PIMax), a novel meta-learning framework for training intelligent agents. PIMax frames learning as an active inference process that seeks to maximize a computationally tractable proxy for the system's variational free energy, termed Predictive Integrity (PI). We operationalize this framework using a dual-optimizer architecture, where a fast meta-optimizer performs inference over the system's internal configuration (e.g., routing strategies) to minimize "surprise," while a slow main optimizer updates the system's world model. This dual-timescale dynamic allows the agent to rapidly adapt its computational strategy while stably accumulating knowledge. We argue that PIMax provides a unifying theoretical lens for understanding the emergence of complex, self-organizing behaviors in artificial neural networks.
- **大纲 (Outline)**:
  1. **Introduction**: AI 中元学习的挑战。介绍 FEP 作为生物智能的统一理论，并提出其在 AI 中的应用潜力。
  2. **Theoretical Foundation: Integrated Predictive Workspace Theory (IPWT)**:
     - 2.1. From FEP to Predictive Integrity (PI): 推导 PI 作为自由能的可计算代理。
     - 2.2. The PI Score: Formalizing inaccuracy and complexity costs.
  3. **The PIMax Framework**:
     - 3.1. Dual Optimizers as an Action-Perception Loop.
     - 3.2. "Surprise" as the Gradient Field: SAPS as the inference engine.
     - 3.3. The World Model as the Generative Model.
  4. **Connecting to Other Theories**: 讨论 PIMax 与现有元学习算法（如 MAML, Reptile）和持续学习方法的关系。
  5. **Conclusion**: 展望 PIMax 作为构建更通用、更自适应 AI 的理论基础。

---

## 论文 3 (优先发表, 目标 ARC 奖): 应用与涌现

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
     - 4.3. Causal Solidification: 分析 `gate_param` 的演化，展示模型如何“记住”并固化成功的计算路径。
  5. **Conclusion**: 总结我们的方法作为一种新的“可微程序搜索”范式的潜力，并讨论其对未来 AGI 研究的启示。
