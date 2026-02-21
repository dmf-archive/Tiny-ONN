"""
验证 Fisher 信息对角近似 (exp_avg_sq) 与输入协方差 C_{t-1} 的关系

理论假设:
    对于线性层 y = Wx + b，Fisher 信息矩阵 F 与输入协方差 C = E[xx^T] 存在对偶关系
    Adam 的 exp_avg_sq 近似于 diag(F)，因此应该与 diag(C) 成正比

验证方法:
    1. 生成具有特定协方差结构的输入数据
    2. 训练线性层，记录 Adam 的 exp_avg_sq
    3. 直接计算输入协方差的特征值/特征向量
    4. 比较两者的对齐程度
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class VerificationConfig:
    input_dim: int = 64
    output_dim: int = 32
    num_samples: int = 10000
    batch_size: int = 256
    num_epochs: int = 50
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    

def generate_correlated_data(
    num_samples: int,
    input_dim: int,
    correlation_structure: str = "random"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    生成具有特定协方差结构的合成数据
    
    Args:
        correlation_structure: "random", "low_rank", "diagonal", "anisotropic"
    """
    if correlation_structure == "random":
        # 随机协方差矩阵
        A = torch.randn(input_dim, input_dim)
        cov = A @ A.T + torch.eye(input_dim) * 0.1
        
    elif correlation_structure == "low_rank":
        # 低秩协方差（模拟真实数据的低维流形结构）
        rank = input_dim // 4
        U = torch.randn(input_dim, rank)
        cov = U @ U.T + torch.eye(input_dim) * 0.01
        
    elif correlation_structure == "diagonal":
        # 对角协方差（特征独立）
        eigenvalues = torch.rand(input_dim) * 5 + 0.1
        cov = torch.diag(eigenvalues)
        
    elif correlation_structure == "anisotropic":
        # 各向异性（特征值差异大）
        eigenvalues = torch.exp(torch.linspace(-3, 3, input_dim))
        Q, _ = torch.linalg.qr(torch.randn(input_dim, input_dim))
        cov = Q @ torch.diag(eigenvalues) @ Q.T
        
    else:
        raise ValueError(f"Unknown structure: {correlation_structure}")
    
    # 从多元高斯分布采样
    mean = torch.randn(input_dim)
    
    # 使用 Cholesky 分解生成样本
    L = torch.linalg.cholesky(cov + torch.eye(input_dim) * 1e-6)
    z = torch.randn(num_samples, input_dim)
    X = z @ L.T + mean
    
    # 生成目标（简单的线性关系加噪声）
    # 注意：y 的维度需要与模型输出一致，这里我们使用自编码器结构
    # 即目标 y 与输入 X 同维度
    y = X + torch.randn(num_samples, input_dim) * 0.1
    
    return X, y, cov


def compute_empirical_covariance(X: torch.Tensor) -> torch.Tensor:
    """计算经验协方差矩阵 C = E[xx^T]"""
    X_centered = X - X.mean(dim=0, keepdim=True)
    C = (X_centered.T @ X_centered) / X.shape[0]
    return C


def extract_adam_state(model: nn.Module, optimizer: optim.Adam) -> dict[str, torch.Tensor]:
    """提取 Adam 优化器的内部状态"""
    exp_avg_sq_dict = {}
    
    for name, param in model.named_parameters():
        if param in optimizer.state:
            state = optimizer.state[param]
            if 'exp_avg_sq' in state:
                # 对于权重矩阵，exp_avg_sq 的形状与参数相同
                exp_avg_sq_dict[name] = state['exp_avg_sq'].clone()
    
    return exp_avg_sq_dict


def compute_alignment_score(
    fisher_diag: torch.Tensor,
    cov_diag: torch.Tensor,
    method: str = "cosine"
) -> float:
    """
    计算两个对角向量的对齐程度
    
    Args:
        method: "cosine", "correlation", "relative_error"
    """
    # 展平并归一化
    f = fisher_diag.flatten().float()
    c = cov_diag.flatten().float()
    
    if method == "cosine":
        # 余弦相似度
        f_norm = f / (f.norm() + 1e-12)
        c_norm = c / (c.norm() + 1e-12)
        return (f_norm * c_norm).sum().item()
    
    elif method == "correlation":
        # Pearson 相关系数
        f_mean, c_mean = f.mean(), c.mean()
        f_std, c_std = f.std(), c.std()
        if f_std < 1e-12 or c_std < 1e-12:
            return 0.0
        return ((f - f_mean) * (c - c_mean)).mean().item() / (f_std * c_std)
    
    elif method == "relative_error":
        # 相对误差（假设线性关系）
        # 找到最佳比例因子
        scale = (f * c).sum() / (c * c).sum()
        error = (f - scale * c).abs().mean() / f.abs().mean()
        return 1.0 - error.item()
    
    else:
        raise ValueError(f"Unknown method: {method}")


def run_verification_experiment(config: VerificationConfig, structure: str) -> dict:
    """运行单次验证实验"""
    
    # 生成数据
    X_full, y_full, true_cov = generate_correlated_data(
        config.num_samples,
        config.input_dim,
        structure
    )
    
    # 计算真实的协方差特征分解（用于后续比较）
    eigvals_true, eigvecs_true = torch.linalg.eigh(true_cov)
    
    # 创建简单的线性模型（自编码器结构，输入输出维度相同）
    class SimpleLinear(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(dim, dim) * 0.01)
            self.bias = nn.Parameter(torch.zeros(dim))
            
        def forward(self, x):
            return x @ self.weight.T + self.bias
    
    model = SimpleLinear(config.input_dim)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=config.betas)
    criterion = nn.MSELoss()
    
    # 训练并收集统计信息
    alignment_history = []
    
    for epoch in range(config.num_epochs):
        # 随机打乱
        perm = torch.randperm(config.num_samples)
        
        for i in range(0, config.num_samples, config.batch_size):
            indices = perm[i:i + config.batch_size]
            X_batch = X_full[indices]
            y_batch = y_full[indices]
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
        
        # 每 10 个 epoch 计算一次对齐度
        if epoch % 10 == 0 or epoch == config.num_epochs - 1:
            # 获取当前批次的协方差
            X_batch = X_full[:config.batch_size]
            emp_cov = compute_empirical_covariance(X_batch)
            
            # 获取 Adam 状态
            adam_state = extract_adam_state(model, optimizer)
            
            # 对于权重矩阵，exp_avg_sq 累积的是梯度平方
            # 梯度 g = δ ⊗ x，其中 δ = output - target
            # Fisher 信息: F = E[δ^2] ⊗ C，即 exp_avg_sq[i,j] ∝ E[δ_i^2] * C_{jj}
            if 'weight' in adam_state:
                exp_sq = adam_state['weight']
                
                # 关键修正：使用 sum 而非 mean
                # 对每个输入维度 j，它在所有输出维度上的总 Fisher 信息与 C_{jj} 成正比
                # sum_i exp_avg_sq[i,j] ∝ sum_i E[δ_i^2] * C_{jj} = C_{jj} * sum_i E[δ_i^2]
                fisher_proxy = exp_sq.sum(dim=0)  # 沿输出维度求和
                cov_diag = emp_cov.diag()
                
                # 计算对齐度
                cosine_sim = compute_alignment_score(fisher_proxy, cov_diag, "cosine")
                correlation = compute_alignment_score(fisher_proxy, cov_diag, "correlation")
                
                alignment_history.append({
                    'epoch': epoch,
                    'cosine': cosine_sim,
                    'correlation': correlation,
                    'loss': loss.item()
                })
    
    return {
        'structure': structure,
        'alignment_history': alignment_history,
        'final_cosine': alignment_history[-1]['cosine'] if alignment_history else 0.0,
        'final_correlation': alignment_history[-1]['correlation'] if alignment_history else 0.0,
    }


def theoretical_analysis():
    """理论分析：证明 Fisher 信息与协方差的关系"""
    print("=" * 60)
    print("理论分析: Fisher 信息 vs 输入协方差")
    print("=" * 60)
    print()
    print("对于线性层 y = Wx + b，损失 L = 1/2 ||y - t||^2")
    print("梯度: ∂L/∂W = (y - t) ⊗ x = δ ⊗ x")
    print()
    print("Fisher 信息矩阵: F = E[ (∂L/∂W) (∂L/∂W)^T ]")
    print("                 = E[ (δ ⊗ x) (δ ⊗ x)^T ]")
    print("                 = E[ δ^2 ] ⊗ E[ x x^T ]  (假设 δ ⊥ x)")
    print("                 ∝ I ⊗ C")
    print()
    print("结论:")
    print("  1. Fisher 信息的对角元与协方差对角元成正比")
    print("  2. Adam 的 exp_avg_sq ≈ diag(F) ∝ diag(C)")
    print("  3. 非对角元不同：Fisher 有参数空间的结构，协方差是数据空间")
    print()


def main():
    """主验证流程"""
    torch.manual_seed(42)
    np.random.seed(42)
    
    config = VerificationConfig()
    
    theoretical_analysis()
    
    print("=" * 60)
    print("实验验证")
    print("=" * 60)
    print()
    
    structures = ["diagonal", "low_rank", "anisotropic", "random"]
    results = []
    
    for structure in structures:
        print(f"测试结构: {structure}")
        result = run_verification_experiment(config, structure)
        results.append(result)
        
        print(f"  最终余弦相似度: {result['final_cosine']:.4f}")
        print(f"  最终相关系数: {result['final_correlation']:.4f}")
        print()
    
    # 汇总
    print("=" * 60)
    print("验证结果汇总")
    print("=" * 60)
    print()
    print(f"{'结构':<15} {'余弦相似度':>12} {'相关系数':>12}")
    print("-" * 40)
    for r in results:
        print(f"{r['structure']:<15} {r['final_cosine']:>12.4f} {r['final_correlation']:>12.4f}")
    
    print()
    print("结论:")
    avg_cosine = np.mean([r['final_cosine'] for r in results])
    avg_corr = np.mean([r['final_correlation'] for r in results])
    print(f"  平均余弦相似度: {avg_cosine:.4f}")
    print(f"  平均相关系数: {avg_corr:.4f}")
    
    if avg_cosine > 0.8:
        print("  ✓ 强对齐：exp_avg_sq 可以作为协方差的有效代理")
    elif avg_cosine > 0.5:
        print("  △ 中等对齐：需要额外的缩放或变换")
    else:
        print("  ✗ 弱对齐：理论假设在此场景不成立")


if __name__ == "__main__":
    main()
