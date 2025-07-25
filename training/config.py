from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _from_dict(data_class, data):
    if data is None:
        return None
    return data_class(
        **{
            key: (
                _from_dict(field, data[key])
                if isinstance(data.get(key), dict)
                else data[key]
            )
            for key, field in data_class.__annotations__.items()
            if key in data
        }
    )


@dataclass
class DataConfig:
    mode: str = "local_json"
    train_path: str | None = None
    eval_path: str | None = None
    dataset_name: str | None = None
    dataset_subset: str | None = None
    validation_split_percentage: int = 5
    max_seq_length: int = 512


@dataclass
class ModelConfig:
    model_path: str
    resume_from_checkpoint: str | None = None


@dataclass
class TrainConfig:
    per_device_train_batch_size: int = 1
    num_train_epochs: int = 1
    expert_learning_rate: float = 1.0e-5
    gate_learning_rate: float = 1.0e-4
    sparsity_learning_rate: float = 1.0e-3
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1.0e-8
    lr_scheduler_warmup_steps: int = 10
    smk_loss_weight: float = 0.1


@dataclass
class ObserverConfig:
    output_dir: str
    log_interval: int = 1
    checkpoint_interval: int = 100
    pi_gamma: float = 0.5
    pi_alpha: float = 1.0


@dataclass
class FullConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    observer: ObserverConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FullConfig":
        return _from_dict(cls, data)


def load_config(config_path: Path) -> FullConfig:
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    return FullConfig.from_dict(config_data)
