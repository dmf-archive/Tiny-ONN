import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch

os.environ["HF_HOME"] = str(Path.cwd() / "weights")

from tiny_onn.modular import TinyOnnForCausalLM
from training.config import load_config
from training.data import get_dataloaders
from training.engine import TrainerEngine


def set_seed(seed):
    random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(config_path: Path):
    config = load_config(config_path)
    set_seed(config.system.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and config.system.device == "auto"
        else "cpu"
    )

    # Load the surgically modified model
    model = TinyOnnForCausalLM.from_pretrained(config.model.model_path).to(device)

    if config.model.use_torch_compile:
        model = torch.compile(model)

    # Setup optimizers
    expert_params = [p for name, p in model.named_parameters() if "gate" not in name]
    router_params = [p for name, p in model.named_parameters() if "gate" in name]

    optimizer_experts = torch.optim.AdamW(
        expert_params,
        lr=config.training.expert_learning_rate,
        weight_decay=config.training.weight_decay,
    )
    optimizer_router = torch.optim.AdamW(
        router_params, lr=config.training.gate_learning_rate
    )

    # Setup dataloaders
    train_loader, eval_loader = get_dataloaders(
        data_config=config.data,
        model_path=config.model.model_path,
        train_batch_size=config.training.per_device_train_batch_size,
        eval_batch_size=config.training.per_device_eval_batch_size,
        num_workers=config.training.dataloader_num_workers,
    )

    # Setup trainer
    trainer = TrainerEngine(
        config=config,
        model=model,
        optimizer_experts=optimizer_experts,
        optimizer_router=optimizer_router,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        device=device,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/meta_train_v1.yaml",
        help="Path to the training configuration file.",
    )
    args = parser.parse_args()
    main(Path(args.config))
