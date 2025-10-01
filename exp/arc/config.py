from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 16
    hidden_size: int = 384
    num_layers: int = 5
    max_position_embeddings: int = 2048
    d_ffn_factor: int = 1

@dataclass
class DataConfig:
    data_path: str = "data/ARC-AGI-2/data"
    batch_size: int = 1
    num_workers: int = 2

@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    lr: float = 1e-3

    w_route_jsd: float = 1.0
    w_temporal_jsd: float = 1.0

    num_epochs: int = 20

    device: str = "cuda"
    seed: int = 42

    eval_interval: int = 2500
    log_interval: int = 10
    max_checkpoints: int = 3

@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    top_p: float = 0.5
