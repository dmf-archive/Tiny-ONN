from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 18 # 8 control tokens + 10 color tokens
    hidden_size: int = 256
    num_layers: int = 7
    max_position_embeddings: int = 4096
    d_ffn_factor: int = 1 # Trust MoIE's dynamic routing, expand hidden state instead.

@dataclass
class DataConfig:
    data_path: str = "data/ARC-AGI-2/data"
    batch_size: int = 1
    num_workers: int = 0

@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    base_learning_rate: float = 3e-3
    prior_learning_rate: float = 3e-2
    weight_decay: float = 0.0
    num_epochs: int = 100

    device: str = "cuda"
    seed: int = 42

    # SBL specific
    kl_prior_epsilon: float = 1e-9

    eval_interval: int = 500 # Eval less frequently
    log_interval: int = 10 # Log more frequently
    max_checkpoints: int = 3
