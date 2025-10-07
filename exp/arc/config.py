from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 16
    hidden_size: int = 512
    num_layers: int = 6
    max_position_embeddings: int = 4096
    d_ffn_factor: int = 1
    routing_gain: float = 1.0

@dataclass
class GenerationConfig:
    use_dfs: bool = False
    min_prob: float = 0.1


@dataclass
class DataConfig:
    data_path: str = "data/ARC-AGI-2/data"
    batch_size: int = 1
    num_workers: int = 4


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    lr: float = 1e-3

    w_route_jsd: float = 1.0

    num_epochs: int = 20
    num_augmentation_views: int = 2

    device: str = "cuda"
    seed: int = 42

    eval_interval: int = 1000
    log_interval: int = 10
    max_checkpoints: int = 3

