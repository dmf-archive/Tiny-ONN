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
    X_centered = X - X.mean(dim=0, keepdim=True)
    C = (X_centered.T @ X_centered) / X.shape[0]
    return C


def extract_optimizer_state(model: nn.Module, optimizer) -> dict[str, torch.Tensor]:
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
    X_full, y_full, _ = generate_correlated_data(
        config.num_samples,
        config.input_dim,
        structure
    )

    class SimpleLinear(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.01)
            self.bias = nn.Parameter(torch.zeros(out_dim))
            
        def forward(self, x):
            return x @ self.weight.T + self.bias

    class AutoEncoder(nn.Module):
        def __init__(self, dim: int, hidden_dim: int):
            super().__init__()
            self.encoder = SimpleLinear(dim, hidden_dim)
            self.decoder = SimpleLinear(hidden_dim, dim)
            
        def forward(self, x):
            h = self.encoder(x)
            return self.decoder(h)

    init_seed = 42
    rng = torch.Generator().manual_seed(init_seed)
    model_init = AutoEncoder(config.input_dim, config.hidden_dim)
    model_init_state = {k: v.detach().clone() for k, v in model_init.state_dict().items()}

    total_steps = config.num_epochs * config.steps_per_epoch
    batch_indices = torch.randint(
        0,
        config.num_samples,
        (total_steps, config.batch_size),
        generator=rng,
    )

    def run_with_optimizer(optimizer_name: str) -> tuple[float, float, list[dict]]:
        model = AutoEncoder(config.input_dim, config.hidden_dim)
        model.load_state_dict(model_init_state, strict=True)

        if optimizer_name == "ars2neo":
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
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.999))
        else:
            raise ValueError(optimizer_name)

        criterion = nn.MSELoss()
        alignment_history = []

        step_cursor = 0
        for epoch in range(config.num_epochs):
            for _step in range(config.steps_per_epoch):
                indices = batch_indices[step_cursor]
                step_cursor += 1
                X_batch = X_full[indices]
                y_batch = y_full[indices]

                if optimizer_name == "ars2neo":
                    def closure():
                        optimizer.zero_grad()
                        output = model(X_batch)
                        loss = criterion(output, y_batch)
                        return loss

                    _loss = optimizer.step(closure)
                else:
                    optimizer.zero_grad()
                    output = model(X_batch)
                    loss = criterion(output, y_batch)
                    loss.backward()
                    optimizer.step()

            if epoch % 10 == 0 or epoch == config.num_epochs - 1:
                X_probe = X_full[:config.batch_size]
                emp_cov = compute_empirical_covariance(X_probe)
                opt_state = extract_optimizer_state(model, optimizer)
                if 'encoder.weight' in opt_state:
                    exp_sq = opt_state['encoder.weight']
                    fisher_proxy = exp_sq.sum(dim=0)
                    cov_diag = emp_cov.diag()
                    cosine_sim = compute_alignment_score(fisher_proxy, cov_diag, "cosine")
                    correlation = compute_alignment_score(fisher_proxy, cov_diag, "correlation")
                    alignment_history.append({
                        'epoch': epoch,
                        'cosine': cosine_sim,
                        'correlation': correlation,
                    })

        final_cosine = alignment_history[-1]['cosine'] if alignment_history else 0.0
        final_correlation = alignment_history[-1]['correlation'] if alignment_history else 0.0
        return final_cosine, final_correlation, alignment_history

    adam_cos, adam_corr, adam_hist = run_with_optimizer("adamw")
    ars2_cos, ars2_corr, ars2_hist = run_with_optimizer("ars2neo")

    return {
        'structure': structure,
        'adam_cosine': adam_cos,
        'adam_correlation': adam_corr,
        'ars2_cosine': ars2_cos,
        'ars2_correlation': ars2_corr,
        'adam_alignment_history': adam_hist,
        'ars2_alignment_history': ars2_hist,
    }


def compare_adam_vs_ars2neo():
    config = VerificationConfig()
    structures = ["diagonal", "low_rank", "anisotropic", "random"]
    
    print("=" * 70)
    print("对比实验: Adam vs ARS2-Neo")
    print("=" * 70)
    print()
    
    results = []
    
    for structure in structures:
        print(f"测试结构: {structure}")
        result_ars2 = run_verification_experiment(config, structure)

        print(f"  AdamW:    余弦={result_ars2['adam_cosine']:.4f}, 相关={result_ars2['adam_correlation']:.4f}")
        print(f"  ARS2-Neo: 余弦={result_ars2['ars2_cosine']:.4f}, 相关={result_ars2['ars2_correlation']:.4f}")
        print()
        results.append({
            'structure': structure,
            'adam_cosine': result_ars2['adam_cosine'],
            'adam_correlation': result_ars2['adam_correlation'],
            'ars2_cosine': result_ars2['ars2_cosine'],
            'ars2_correlation': result_ars2['ars2_correlation'],
        })
    
    # 汇总
    print("=" * 70)
    print("结果汇总")
    print("=" * 70)
    print()
    print(f"{'结构':<15} {'AdamW余弦':>12} {'AdamW相关':>12} {'ARS2余弦':>12} {'ARS2相关':>12}")
    print("-" * 64)
    for r in results:
        print(
            f"{r['structure']:<15}"
            f" {r['adam_cosine']:>12.4f}"
            f" {r['adam_correlation']:>12.4f}"
            f" {r['ars2_cosine']:>12.4f}"
            f" {r['ars2_correlation']:>12.4f}"
        )

    print()
    avg_adam_cosine = np.mean([r['adam_cosine'] for r in results])
    avg_adam_corr = np.mean([r['adam_correlation'] for r in results])
    avg_ars2_cosine = np.mean([r['ars2_cosine'] for r in results])
    avg_ars2_corr = np.mean([r['ars2_correlation'] for r in results])

    print(f"平均余弦相似度 | AdamW={avg_adam_cosine:.4f} | ARS2-Neo={avg_ars2_cosine:.4f}")
    print(f"平均相关系数   | AdamW={avg_adam_corr:.4f} | ARS2-Neo={avg_ars2_corr:.4f}")
    
    return results


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    results = compare_adam_vs_ars2neo()


if __name__ == "__main__":
    main()
