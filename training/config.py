from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Any
import yaml
from tiny_onn.config import TinyOnnConfig

def _from_dict(data_class, data):
    if data is None:
        return None

    kwargs = {}
    for field_name, field_type in data_class.__annotations__.items():
        if field_name in data:
            field_value = data[field_name]
            if is_dataclass(field_type):
                kwargs[field_name] = _from_dict(field_type, field_value)
            elif field_type is TinyOnnConfig:
                kwargs[field_name] = TinyOnnConfig(**field_value)
            else:
                try:
                    kwargs[field_name] = field_type(field_value)
                except (TypeError, ValueError):
                    kwargs[field_name] = field_value
    return data_class(**kwargs)

@dataclass
class DataConfig:
    mode: str = "local_json"
    train_path: str | None = None
    eval_path: str | None = None
    dataset_name: str | None = None
    dataset_subset: str | None = None
    validation_split_percentage: int = 5

@dataclass
class TrainConfig:
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1.0e-8
    lr_scheduler_warmup_steps: int = 10
    surprise_threshold_sigma: float = 2.0
    num_epochs: int = 1
    log_interval: int = 10

@dataclass
class ObserverConfig:
    output_dir: str
    pi_gamma: float = 0.5
    pi_alpha: float = 1.0

@dataclass
class FullConfig:
    data: DataConfig
    model: TinyOnnConfig
    train: TrainConfig
    observer: ObserverConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FullConfig":
        return _from_dict(cls, data)

def load_config(config_path: Path) -> FullConfig:
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    return FullConfig.from_dict(config_data)
