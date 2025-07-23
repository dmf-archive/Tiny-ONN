import yaml
from pathlib import Path
from typing import Any, Dict

from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from tiny_onn.modular import TinyOnnForCausalLM
from training.data import get_dataloaders
from training.engine import TrainerEngine
from training.utils import get_optimizer, get_scheduler


class Config:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

    def __getattr__(self, name: str) -> Any:
        if name in self.config:
            if isinstance(self.config[name], dict):
                return DictWrapper(self.config[name])
            return self.config[name]
        raise AttributeError(f"'Config' object has no attribute '{name}'")


class DictWrapper:
    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def __getattr__(self, name: str) -> Any:
        if name in self.data:
            if isinstance(self.data[name], dict):
                return DictWrapper(self.data[name])
            return self.data[name]
        raise AttributeError(f"'DictWrapper' object has no attribute '{name}'")


def main(config_path: str):
    config = Config(config_path)

    model = TinyOnnForCausalLM.from_pretrained(config.model.model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model_name, cache_dir="weights"
    )

    train_loader, val_loader = get_dataloaders(
        tokenizer,
        config.data.dataset_name,
        config.data.dataset_name,
        config.training.per_device_train_batch_size,
        config.training.dataloader_num_workers,
    )

    optimizer = get_optimizer(model, config.optimizer)
    scheduler = get_scheduler(optimizer, config.scheduler, train_loader)

    engine = TrainerEngine(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    engine.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
