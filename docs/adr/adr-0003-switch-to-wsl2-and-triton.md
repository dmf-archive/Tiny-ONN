---
title: "ADR-0003: 切换到 WSL2 与 Triton 使能的开发环境"
status: "Proposed"
date: "2025-10-11"
authors: "Ω Researcher"
tags: ["architecture", "decision", "environment", "wsl2", "triton", "python"]
supersedes: ""
superseded_by: ""
---

# ADR-0003: 切换到 WSL2 与 Triton 使能的开发环境

## 状态 (Status)

**Proposed** | Accepted | Rejected | Superseded | Deprecated

## 背景 (Context)

当前 `Tiny-ONN` 项目在原生 Windows 环境下进行开发。鉴于 ARC-AGI-2 的提交截止日期（2025-11-03）将至，进一步的架构探索已不具备时效性。因此，项目工作重心从冲刺短期目标，战略性地转向为长期研究和发展奠定更坚实的基础。ARC-AGI-2 仍将是核心基准测试，但当前的首要任务是提升开发环境的性能、标准化与行业对齐度。为了利用如 NVIDIA Triton 等先进的性能优化工具并提升环境的可复现性，有必要从原生 Windows 环境迁移至更符合高性能计算（HPC）和机器学习研究社区标准的 Linux 生态。

## 决策 (Decision)

1. 项目的主要开发环境将从原生 Windows 迁移至 **Windows Subsystem for Linux 2 (WSL2)**。
2. 项目的标准 Python 版本将统一为 **Python 3.11**。该决定基于一份外部权威报告，该报告证实了 Python 3.11 在 2025 年后期，对于包括 PyTorch、Transformers 和 Triton 在内的核心深度学习技术栈，具有最佳的稳定性和最广泛的官方支持。
3. 未来的性能优化工作将引入 **NVIDIA Triton**，用于编写自定义的高性能 GPU 内核。这一决策是迁移至 WSL2 的主要驱动力之一，因为 Triton 在 Linux 环境中拥有最佳支持。

## 后果 (Consequences)

### 积极 (Positive)

- **POS-001**: **性能优化解锁 (Performance Optimization Unlocked)**: 允许使用 Triton 对 GPU 内核进行深度优化，这有望显著提升模型训练和推理的速度。
- **POS-002**: **对齐行业标准 (Alignment with Industry Standards)**: 使项目技术栈与机器学习研究和高性能计算社区的主流标准保持一致，从而改善工具链的兼容性并简化与前沿技术的集成。
- **POS-003**: **增强可复现性 (Enhanced Reproducibility)**: 相比原生 Windows，WSL2 提供的 Linux 环境更为一致和隔离，能有效减少因环境差异导致的错误，提升实验的可复现性。
- **POS-004**: **扩展工具生态 (Expanded Tool Ecosystem)**: 能够无缝利用大量为 Linux 设计的高效开发、调试和系统管理工具。

### 消极 (Negative)

- **NEG-001**: **迁移成本 (Migration Overhead)**: 迁移过程需要投入时间来配置新的 WSL2 环境、验证项目依赖，并适配所有现有的工作流。
- **NEG-002**: **学习曲线 (Learning Curve)**: 对于不熟悉 Linux 或 WSL2 的团队成员，可能存在短暂的学习和适应期，短期内或对开发效率有一定影响。
- **NEG-003**: **潜在的驱动复杂性 (Potential Driver Complexity)**: 可能引入与 Windows 主机和 WSL2 虚拟机之间的 GPU 驱动程序直通相关的新配置问题，需要进行细致的设置和故障排查。

## 考虑的备选方案 (Alternatives Considered)

### 维持原生 Windows 开发环境 (Continue on Native Windows)

- **ALT-001**: **描述 (Description)**: 维持当前在原生 Windows 操作系统上的开发工作流。
- **ALT-002**: **拒绝理由 (Rejection Reason)**: 此方案严重限制甚至完全阻碍了如 Triton 等关键性能优化工具的使用。它加剧了项目与行业标准之间的脱节，使性能瓶颈问题无法得到根本解决。

### 使用双系统或专用 Linux 主机 (Dual-Boot or Dedicated Linux Machine)

- **ALT-003**: **描述 (Description)**: 配置一台独立的物理机或设置双系统，以运行专用的 Linux 发行版（如 Ubuntu）。
- **ALT-004**: **拒绝理由 (Rejection Reason)**: 虽然此方案能提供最佳的原生性能，但它要求在不同操作系统间频繁切换，带来了极大的不便。与 WSL2 相比，它与开发者主要的桌面工作环境集成度较低，为主要使用 Windows 的开发者设置了更高的门槛。

### 使用 Windows 上的 Docker Desktop (Use Docker Desktop on Windows)

- **ALT-005**: **描述 (Description)**: 利用 Docker Desktop 及其 WSL2 后端来将开发环境容器化。
- **ALT-006**: **拒绝理由 (Rejection Reason)**: 尽管这是一个可行的方案，但直接在 WSL2 中进行开发为调试和系统级操作提供了更大的灵活性。Docker 增加了一个额外的抽象层，对于研究和开发的快速迭代有时可能显得繁琐。本次决策的核心是转向 Linux 内核，WSL2 直接提供了这一点。

## 实施注意事项 (Implementation Notes)

- **IMP-001**: **环境标准化 (Environment Standardization)**: 所有开发者需安装配置 WSL2 及一个标准的 Linux 发行版（推荐 Ubuntu 24.04 LTS）。必须提供详尽的文档，指导如何设置 WSL2、NVIDIA 驱动、CUDA 工具包以及基于 `uv` 的 Python 3.11 环境。
- **IMP-002**: **项目迁移与验证 (Project Migration & Validation)**: 必须验证 `pyproject.toml` 中定义的所有依赖项在 WSL2 环境中能够正确安装和运行。任何特定于 Windows 的脚本或文件路径都需要被更新为 POSIX 兼容的格式。
- **IMP-003**: **成功标准 (Success Criteria)**: 当现有的训练脚本（如 `exp/algorithmic/train.py`）能够在 WSL2 环境中成功运行、利用 GPU 并得出与之前一致的结果时，迁移被视为成功。

## 参考文献 (References)

- **REF-001**: [`ADR-0001: 引入合成基准测试以进行架构能力分析`](adr-0001-synthetic-benchmarks-for-capability-analysis.md)
- **REF-002**: 外部报告: "Recommended Python Version for Deep Learning Stack in Late 2025"
- **REF-003**: NVIDIA Triton Inference Server Documentation
