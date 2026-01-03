import torch

from src.configs.base import ExperimentConfig
from src.optimizers.ars import ARSOptimizer


def get_optimizer(model: torch.nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    opt_config = config.optimizer

    rmsuon_params = []
    adamw_params = []
    routing_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "proto" in name or "gate" in name:
            routing_params.append(p)
        elif p.ndim >= 2:
            rmsuon_params.append(p)
        else:
            adamw_params.append(p)

    param_groups = [
        {
            "params": rmsuon_params,
            "is_rmsuon_group": True,
            "lr": opt_config.lr,
            "weight_decay": opt_config.weight_decay,
            "ns_steps": opt_config.ns_steps,
            "rho": opt_config.ars.rho,
            "k": opt_config.ars.k,
            "alpha": opt_config.ars.alpha,
            "adaptive": opt_config.ars.adaptive,
        },
        {
            "params": adamw_params,
            "is_rmsuon_group": False,
            "lr": opt_config.lr,
            "weight_decay": opt_config.weight_decay,
            "rho": opt_config.ars.rho,
            "k": opt_config.ars.k,
            "alpha": opt_config.ars.alpha,
            "adaptive": opt_config.ars.adaptive,
        },
        {
            "params": routing_params,
            "is_rmsuon_group": False,
            "lr": opt_config.lr * 0.1,
            "weight_decay": 0.0,
            "rho": opt_config.ars.rho,
            "k": opt_config.ars.k,
            "alpha": opt_config.ars.alpha,
            "adaptive": opt_config.ars.adaptive,
        }
    ]

    if opt_config.type == "ARS":
        return ARSOptimizer(param_groups, betas=opt_config.betas, eps=opt_config.eps)
    elif opt_config.type == "AdamW":
        return torch.optim.AdamW(param_groups, betas=opt_config.betas, eps=opt_config.eps)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_config.type}")
