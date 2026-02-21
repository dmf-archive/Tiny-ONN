"""
验证非线性 FFN 下 Fisher-Covariance 对偶性

与线性版本的区别：
    - 使用 SiLU 激活的 FFN 结构替代线性 AutoEncoder
    - 测试非线性变换是否破坏 exp_avg_sq ∝ diag(C) 关系
    - 对比 Adam 与 ARS2-Neo 在非线性场景下的对齐质量
    - 增加 SAM (rho=0.1) 模式对比

理论预期:
    非线性层：grad = δ ⊗ σ'(Wx+b) ∘ x
    其中 σ' 是 SiLU 的导数 σ(x) · sigmoid(x)
    Fisher 对角元包含激活导数的影响，对齐关系可能弱化
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
    hidden_dim: int = 128
    num_samples: int = 5000
    batch_size: int = 256
    num_epochs: int = 50
    steps_per_epoch: int = 20
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.95)
    ns_steps: int = 5


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
    
    for group in optimizer.param_groups:
        for param in group['params']:
            if param in optimizer.state:
                state = optimizer.state[param]
                if 'exp_avg_sq' in state:
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
    
    else:
        raise ValueError(f"Unknown method: {method}")


class SiLUFFN(nn.Module):
    """SiLU 激活的 FFN 模块"""
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, in_dim, bias=True)
        self.act = nn.SiLU()
        
    def forward(self, x):
        h = self.act(self.fc1(x))
        return self.fc2(h)


def run_verification_experiment(
    config: VerificationConfig,
    structure: str,
    use_ars2: bool = False,
    rho: float = 0.0
) -> dict:
    """运行单次验证实验"""
    
    X_full, y_full, true_cov = generate_correlated_data(
        config.num_samples,
        config.input_dim,
        structure
    )
    
    model = SiLUFFN(config.input_dim, config.hidden_dim)
    criterion = nn.MSELoss()
    
    if use_ars2:
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
            rho=rho
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999))
    
    alignment_history = []
    
    for epoch in range(config.num_epochs):
        for step in range(config.steps_per_epoch):
            indices = torch.randperm(config.num_samples)[:config.batch_size]
            X_batch = X_full[indices]
            y_batch = y_full[indices]
            
            if use_ars2:
                def closure():
                    optimizer.zero_grad()
                    output = model(X_batch)
                    loss = criterion(output, y_batch)
                    return loss
                loss = optimizer.step(closure)
            else:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
        
        if epoch % 10 == 0 or epoch == config.num_epochs - 1:
            X_batch = X_full[:config.batch_size]
            emp_cov = compute_empirical_covariance(X_batch)
            
            opt_state = extract_optimizer_state(model, optimizer)
            
            if 'fc1.weight' in opt_state:
                exp_sq = opt_state['fc1.weight']
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


def compare_experiments():
    """对比 Adam, ARS2-Neo, ARS2-Neo+SAM 的对齐性能"""
    
    config = VerificationConfig()
    structures = ["diagonal", "low_rank", "anisotropic", "random"]
    
    print("=" * 80)
    print("非线性 FFN 对比实验: Adam vs ARS2-Neo vs ARS2-Neo+SAM (SiLU, rho=0.1)")
    print("=" * 80)
    print()
    
    results = []
    
    for structure in structures:
        print(f"测试结构: {structure}")
        
        result_adam = run_verification_experiment(config, structure, use_ars2=False)
        result_ars2 = run_verification_experiment(config, structure, use_ars2=True, rho=0.0)
        result_ars2_sam = run_verification_experiment(config, structure, use_ars2=True, rho=0.1)
        
        print(f"  Adam:         余弦={result_adam['final_cosine']:.4f}, 相关={result_adam['final_correlation']:.4f}")
        print(f"  ARS2-Neo:     余弦={result_ars2['final_cosine']:.4f}, 相关={result_ars2['final_correlation']:.4f}")
        print(f"  ARS2-Neo+SAM: 余弦={result_ars2_sam['final_cosine']:.4f}, 相关={result_ars2_sam['final_correlation']:.4f}")
        print()
        
        results.append({
            'structure': structure,
            'adam_cosine': result_adam['final_cosine'],
            'adam_correlation': result_adam['final_correlation'],
            'ars2_cosine': result_ars2['final_cosine'],
            'ars2_correlation': result_ars2['final_correlation'],
            'ars2_sam_cosine': result_ars2_sam['final_cosine'],
            'ars2_sam_correlation': result_ars2_sam['final_correlation'],
        })
    
    print("=" * 80)
    print("结果汇总")
    print("=" * 80)
    print()
    print(f"{'结构':<12} {'Adam':>10} {'ARS2':>10} {'ARS2+SAM':>10} | {'Adam':>10} {'ARS2':>10} {'ARS2+SAM':>10}")
    print(f"{'':12} {'余弦':>10} {'余弦':>10} {'余弦':>10} | {'相关':>10} {'相关':>10} {'相关':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['structure']:<12} {r['adam_cosine']:>10.4f} {r['ars2_cosine']:>10.4f} {r['ars2_sam_cosine']:>10.4f} | "
              f"{r['adam_correlation']:>10.4f} {r['ars2_correlation']:>10.4f} {r['ars2_sam_correlation']:>10.4f}")
    
    print()
    print("结论:")
    avg_adam_cos = np.mean([r['adam_cosine'] for r in results])
    avg_ars2_cos = np.mean([r['ars2_cosine'] for r in results])
    avg_ars2_sam_cos = np.mean([r['ars2_sam_cosine'] for r in results])
    avg_adam_corr = np.mean([r['adam_correlation'] for r in results])
    avg_ars2_corr = np.mean([r['ars2_correlation'] for r in results])
    avg_ars2_sam_corr = np.mean([r['ars2_sam_correlation'] for r in results])
    
    print(f"  Adam 平均:       余弦={avg_adam_cos:.4f}, 相关={avg_adam_corr:.4f}")
    print(f"  ARS2-Neo 平均:   余弦={avg_ars2_cos:.4f}, 相关={avg_ars2_corr:.4f}")
    print(f"  ARS2+SAM 平均:   余弦={avg_ars2_sam_cos:.4f}, 相关={avg_ars2_sam_corr:.4f}")
    print()
    
    if avg_ars2_sam_cos > avg_ars2_cos:
        print(f"  ✓ SAM 提升余弦对齐: {avg_ars2_sam_cos - avg_ars2_cos:+.4f}")
    else:
        print(f"  • SAM 余弦对齐变化: {avg_ars2_sam_cos - avg_ars2_cos:+.4f}")
    
    if avg_ars2_sam_corr > avg_ars2_corr:
        print(f"  ✓ SAM 提升相关对齐: {avg_ars2_sam_corr - avg_ars2_corr:+.4f}")
    else:
        print(f"  • SAM 相关对齐变化: {avg_ars2_sam_corr - avg_ars2_corr:+.4f}")
    
    return results


def main():
    """主验证流程"""
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=" * 80)
    print("理论分析: 非线性层 + SAM 平坦度约束下的 Fisher-Covariance 对偶性")
    print("=" * 80)
    print()
    print("核心问题:")
    print("  1. 非线性层: grad = δ ⊗ (σ'(Wx+b) ∘ x)")
    print("  2. SAM 扰动: 在流形度量下计算对抗方向，可能改变 Fisher 估计")
    print()
    print("预期:")
    print("  - SAM 的平坦度约束可能平滑损失地形，改善 Fisher 估计质量")
    print("  - 非线性场景下 SAM 可能缩小 Adam 与 ARS2-Neo 的性能差距")
    print()
    
    results = compare_experiments()


if __name__ == "__main__":
    main()
