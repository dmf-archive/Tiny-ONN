"""
验证 ARS2-Neo 的 exp_avg_sq 与输入协方差 C_{t-1} 的关系

与 verify_fisher_covariance.py 的区别：
    - 使用 ARS2-Neo 而非普通 Adam
    - ARS2-Neo 通过 Newton-Schulz 正交化实现了 Full-rank Fisher Approximation
    - 理论上应该比普通 Adam 更好地对齐协方差结构

理论假设:
    ARS2-Neo: Diagonal Fisher + Orthogonalized Parameter Space ≈ Full-rank NGD
    这意味着 exp_avg_sq 应该比普通 Adam 更好地反映输入协方差
"""

import sys
sys.path.insert(0, 'e:/Dev/Chain/Tiny-ONN')

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

from src.optimizers.ars2_neo import SingleDeviceARS2Neo


@dataclass
class VerificationConfig:
    input_dim: int = 64
    hidden_dim: int = 128  # 使用 2D 权重矩阵 [hidden_dim, input_dim]
    num_samples: int = 5000
    batch_size: int = 256
    num_epochs: int = 50  # 减少epoch数，增加每epoch步数
    steps_per_epoch: int = 20  # 每epoch训练步数
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.95)
    ns_steps: int = 5
    rho: float = 0.0


def generate_correlated_data(
    num_samples: int,
    input_dim: int,
    correlation_structure: str = "random"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """生成具有特定协方差结构的合成数据"""
    
    if correlation_structure == "random":
        A = torch.randn(input_dim, input_dim)
        cov = A @ A.T + torch.eye(input_dim) * 0.1
        
    elif correlation_structure == "low_rank":
        rank = input_dim // 4
        U = torch.randn(input_dim, rank)
        cov = U @ U.T + torch.eye(input_dim) * 0.01
        
    elif correlation_structure == "diagonal":
        eigenvalues = torch.rand(input_dim) * 5 + 0.1
        cov = torch.diag(eigenvalues)
        
    elif correlation_structure == "anisotropic":
        eigenvalues = torch.exp(torch.linspace(-3, 3, input_dim))
        Q, _ = torch.linalg.qr(torch.randn(input_dim, input_dim))
        cov = Q @ torch.diag(eigenvalues) @ Q.T
        
    else:
        raise ValueError(f"Unknown structure: {correlation_structure}")
    
    L = torch.linalg.cholesky(cov + torch.eye(input_dim) * 1e-6)
    z = torch.randn(num_samples, input_dim)
    X = z @ L.T
    y = X + torch.randn(num_samples, input_dim) * 0.1
    
    return X, y, cov


def compute_empirical_covariance(X: torch.Tensor) -> torch.Tensor:
    """计算经验协方差矩阵"""
    X_centered = X - X.mean(dim=0, keepdim=True)
    C = (X_centered.T @ X_centered) / X.shape[0]
    return C


def extract_optimizer_state(model: nn.Module, optimizer) -> dict[str, torch.Tensor]:
    """提取优化器的内部状态"""
    exp_avg_sq_dict = {}
    
    # 调试：打印优化器状态和模型参数
    # print(f"优化器 state 键数: {len(optimizer.state)}")
    # print(f"模型参数: {list(model.named_parameters()).keys()}")
    
    # 通过 param_groups 遍历参数
    for group in optimizer.param_groups:
        for param in group['params']:
            if param in optimizer.state:
                state = optimizer.state[param]
                if 'exp_avg_sq' in state:
                    # 找到对应的参数名
                    for name, p in model.named_parameters():
                        if p is param:
                            exp_avg_sq_dict[name] = state['exp_avg_sq'].clone()
                            break
    
    return exp_avg_sq_dict


def compute_alignment_score(
    fisher_proxy: torch.Tensor,
    cov_diag: torch.Tensor,
    method: str = "cosine"
) -> float:
    """计算两个向量的对齐程度"""
    f = fisher_proxy.flatten().float()
    c = cov_diag.flatten().float()
    
    if method == "cosine":
        f_norm = f / (f.norm() + 1e-12)
        c_norm = c / (c.norm() + 1e-12)
        return (f_norm * c_norm).sum().item()
    
    elif method == "correlation":
        f_mean, c_mean = f.mean(), c.mean()
        f_std, c_std = f.std(), c.std()
        if f_std < 1e-12 or c_std < 1e-12:
            return 0.0
        return ((f - f_mean) * (c - c_mean)).mean().item() / (f_std * c_std)
    
    elif method == "relative_error":
        scale = (f * c).sum() / (c * c).sum()
        error = (f - scale * c).abs().mean() / f.abs().mean()
        return 1.0 - error.item()
    
    else:
        raise ValueError(f"Unknown method: {method}")


def run_verification_experiment(config: VerificationConfig, structure: str) -> dict:
    """运行单次验证实验"""
    
    X_full, y_full, true_cov = generate_correlated_data(
        config.num_samples,
        config.input_dim,
        structure
    )
    
    # 使用 2D 权重矩阵 [hidden_dim, input_dim] 以兼容 Newton-Schulz
    class SimpleLinear(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.01)
            self.bias = nn.Parameter(torch.zeros(out_dim))
            
        def forward(self, x):
            return x @ self.weight.T + self.bias
    
    # 编码器 + 解码器结构，保持输入输出同维度
    class AutoEncoder(nn.Module):
        def __init__(self, dim: int, hidden_dim: int):
            super().__init__()
            self.encoder = SimpleLinear(dim, hidden_dim)
            self.decoder = SimpleLinear(hidden_dim, dim)
            
        def forward(self, x):
            h = self.encoder(x)
            return self.decoder(h)
    
    model = AutoEncoder(config.input_dim, config.hidden_dim)
    
    # 使用 ARS2-Neo：2D 参数用 ARS2，1D 参数用 AdamW
    # 注意：需要先收集到列表，避免生成器耗尽
    all_params = list(model.parameters())
    ars2_params = [p for p in all_params if p.ndim >= 2]
    adamw_params = [p for p in all_params if p.ndim < 2]
    
    param_groups = []
    if ars2_params:
        param_groups.append({'params': ars2_params, 'is_rmsuon_group': True})
    if adamw_params:
        param_groups.append({'params': adamw_params, 'is_rmsuon_group': False})
    
    optimizer = SingleDeviceARS2Neo(
        param_groups,
        lr=config.lr,
        betas=config.betas,
        ns_steps=config.ns_steps,
        rho=config.rho
    )
    
    criterion = nn.MSELoss()
    alignment_history = []
    
    for epoch in range(config.num_epochs):
        for step in range(config.steps_per_epoch):
            indices = torch.randperm(config.num_samples)[:config.batch_size]
            X_batch = X_full[indices]
            y_batch = y_full[indices]
            
            def closure():
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                return loss
            
            loss = optimizer.step(closure)
        
        # 每 10 个 epoch 计算对齐度
        if epoch % 10 == 0 or epoch == config.num_epochs - 1:
            X_batch = X_full[:config.batch_size]
            emp_cov = compute_empirical_covariance(X_batch)
            
            opt_state = extract_optimizer_state(model, optimizer)
            
            # 使用 encoder.weight: shape [hidden_dim, input_dim]
            # 对应输入协方差结构
            if 'encoder.weight' in opt_state:
                exp_sq = opt_state['encoder.weight']
                
                # ARS2-Neo: exp_avg_sq shape [hidden_dim, input_dim]
                # 对 hidden_dim 求和，得到 input_dim 长度的向量
                fisher_proxy = exp_sq.sum(dim=0)
                cov_diag = emp_cov.diag()
                
                cosine_sim = compute_alignment_score(fisher_proxy, cov_diag, "cosine")
                correlation = compute_alignment_score(fisher_proxy, cov_diag, "correlation")
                
                alignment_history.append({
                    'epoch': epoch,
                    'cosine': cosine_sim,
                    'correlation': correlation,
                })
    
    return {
        'structure': structure,
        'alignment_history': alignment_history,
        'final_cosine': alignment_history[-1]['cosine'] if alignment_history else 0.0,
        'final_correlation': alignment_history[-1]['correlation'] if alignment_history else 0.0,
    }


def compare_adam_vs_ars2neo():
    """对比 Adam 和 ARS2-Neo 的对齐性能"""
    
    config = VerificationConfig()
    structures = ["diagonal", "low_rank", "anisotropic", "random"]
    
    print("=" * 70)
    print("对比实验: Adam vs ARS2-Neo")
    print("=" * 70)
    print()
    
    results = []
    
    for structure in structures:
        print(f"测试结构: {structure}")
        
        # ARS2-Neo 结果
        result_ars2 = run_verification_experiment(config, structure)
        
        print(f"  ARS2-Neo: 余弦={result_ars2['final_cosine']:.4f}, 相关={result_ars2['final_correlation']:.4f}")
        print()
        
        results.append({
            'structure': structure,
            'ars2_cosine': result_ars2['final_cosine'],
            'ars2_correlation': result_ars2['final_correlation'],
        })
    
    # 汇总
    print("=" * 70)
    print("结果汇总")
    print("=" * 70)
    print()
    print(f"{'结构':<15} {'ARS2余弦':>12} {'ARS2相关':>12}")
    print("-" * 40)
    for r in results:
        print(f"{r['structure']:<15} {r['ars2_cosine']:>12.4f} {r['ars2_correlation']:>12.4f}")
    
    print()
    print("结论:")
    avg_cosine = np.mean([r['ars2_cosine'] for r in results])
    avg_corr = np.mean([r['ars2_correlation'] for r in results])
    print(f"  ARS2-Neo 平均余弦相似度: {avg_cosine:.4f}")
    print(f"  ARS2-Neo 平均相关系数: {avg_corr:.4f}")
    
    if avg_cosine > 0.9:
        print("  ✓ 强对齐：ARS2-Neo 的 exp_avg_sq 是协方差的优秀代理")
    
    return results


def main():
    """主验证流程"""
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=" * 70)
    print("理论分析: ARS2-Neo 的特殊性质")
    print("=" * 70)
    print()
    print("ARS2-Neo 核心特性:")
    print("  1. Energy-Geometry Decoupling")
    print("     - 能量: Adam 二阶矩决定步长")
    print("     - 几何: Muon Newton-Schulz 正交化保证方向")
    print()
    print("  2. Full-rank Fisher Approximation")
    print("     - Diagonal Fisher + Orthogonalized Space ≈ Full-rank NGD")
    print("     - 理论上比纯对角 Fisher 更好地反映协方差结构")
    print()
    print("  3. 对 FARS 的意义")
    print("     - ARS2-Neo 的 exp_avg_sq 经过几何补偿")
    print("     - 应该比普通 Adam 更好地作为认知代价信号")
    print()
    
    results = compare_adam_vs_ars2neo()


if __name__ == "__main__":
    main()
