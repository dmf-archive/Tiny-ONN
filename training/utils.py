from typing import Any, Dict

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader

from tiny_onn.modular import TinyOnnForCausalLM


def get_optimizer(model: TinyOnnForCausalLM, config: Dict[str, Any]) -> Optimizer:
    return AdamW(model.parameters(), **config)


def get_scheduler(
    optimizer: Optimizer, config: Dict[str, Any], train_loader: DataLoader
) -> LRScheduler:
    num_training_steps = len(train_loader) * config.get("num_epochs", 3)

    def lr_lambda(current_step: int):
        if current_step < config.get("warmup_steps", 500):
            return float(current_step) / float(max(1, config.get("warmup_steps", 500)))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)
