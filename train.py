import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

from tiny_onn.modular import TinyOnnForCausalLM
from training.config import load_config
from training.data import get_dataloaders
from training.engine import TrainerEngine


def main(config_path: str):
    config = load_config(Path(config_path))

    model = TinyOnnForCausalLM.from_pretrained(
        config.model.model_path, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model_name,
        model_max_length=config.data.max_seq_length,
        cache_dir="weights",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataloader, eval_dataloader = get_dataloaders(
        tokenizer=tokenizer,
        train_path="data/dummy_chat_data.jsonl",
        val_path="data/dummy_chat_data.jsonl",
        batch_size=config.training.per_device_train_batch_size,
        num_workers=config.training.dataloader_num_workers,
        max_length=config.data.max_seq_length,
    )

    expert_params = [
        p
        for n, p in model.named_parameters()
        if "gate" not in n and p.requires_grad
    ]
    router_params = [
        p for n, p in model.named_parameters() if "gate" in n and p.requires_grad
    ]

    optimizer_experts = AdamW(
        expert_params,
        lr=config.training.expert_learning_rate,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        eps=config.training.adam_epsilon,
        weight_decay=config.training.weight_decay,
    )
    optimizer_router = AdamW(
        router_params,
        lr=config.training.gate_learning_rate,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        eps=config.training.adam_epsilon,
        weight_decay=config.training.weight_decay,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    engine = TrainerEngine(
        config=config,
        model=model,
        optimizer_experts=optimizer_experts,
        optimizer_router=optimizer_router,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        device=device,
    )

    engine.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML config file."
    )
    args = parser.parse_args()
    main(args.config)
