from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 16
    hidden_size: int = 256
    num_layers: int = 8
    max_position_embeddings: int = 2048
    d_ffn_factor: int = 2

@dataclass
class DataConfig:
    data_path: str = "data/ARC-AGI-2/data"
    batch_size: int = 1
    num_workers: int = 2

@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    lr_mu: float = 3e-3
    lr_proto: float = 1e-4
    lr_gate: float = 5e-4

    ema_alpha_acc: float = 0.98
    w_gate: float = 1.0
    num_epochs: int = 20

    device: str = "cuda"
    seed: int = 42

    eval_interval: int = 10000
    log_interval: int = 10
    max_checkpoints: int = 3

@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
