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
class ModelConfig:
    model_path: str
    base_model_name: str
    teacher_model_name: str | None = None
    resume_from_checkpoint: str | None = None
    use_torch_compile: bool = True


@dataclass
class DataConfig:
    dataset_name: str
    dataset_subset: str
    validation_split_percentage: int
    max_seq_length: int


@dataclass
class TrainingConfig:
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    dataloader_num_workers: int
    expert_learning_rate: float
    gate_learning_rate: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    max_grad_norm: float
    lr_scheduler_warmup_steps: int = 0
    distillation_alpha: float = 0.5
    distillation_temperature: float = 2.0
    pi_gamma: float = 1.0
    pi_alpha: float = 1.0 # Surprise Budget
    activation_budget: float = 0.5
    moe_capacity_factor: float = 1.25
    moe_min_capacity: int = 4
    router_loss_lambda: float = 0.01


@dataclass
class LoggingConfig:
    log_interval: int
    eval_interval: int
    checkpoint_interval: int
    plot_interval: int
    rolling_checkpoint_count: int


@dataclass
class SystemConfig:
    device: str
    seed: int


@dataclass
class TrainConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    logging: LoggingConfig
    system: SystemConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        return _from_dict(cls, data)


def load_config(config_path: Path) -> TrainConfig:
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    return TrainConfig.from_dict(config_data)
