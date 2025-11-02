from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class DataConfig:
    data_dir: str = "./data"
    batch_size: int = 256
    num_workers: int = 4

@dataclass
class ModelConfig:
    model_type: str = "standard" # "standard" or "ffn_in_head"
    image_size: int = 28
    patch_size: int = 4
    in_channels: int = 1
    num_classes: int = 10
    embed_dim: int = 256
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1

@dataclass
class TrainConfig:
    epochs: int = 12
    lr: float = 1e-3
    optimizer: str = "AdamW"
    weight_decay: float = 0.05
    log_dir: str = "runs/f3eo_fashion/phase-0"
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints/f3eo_fashion/phase-0"
    resume: bool = True

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

def get_config():
    return Config()