from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 16
    hidden_size: int = 512
    num_heads: int = 8
    num_key_value_heads: int = 8
    num_layers: int = 4
    max_position_embeddings: int = 4096
    ffn_scale: int = 4
    use_checkpoint: bool = False
    dropout: float = 0.9

@dataclass
class GenerationConfig:
    use_dfs: bool = False
    min_prob: float = 0.1


@dataclass
class DataConfig:
    data_path: str = "data/ARC-AGI-2/data"
    batch_size: int = 8
    num_workers: int = 4
    max_tokens_per_batch: int = 8192


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.1

    num_epochs: int = 100
    num_augmentation_views: int = 4
    max_steps_per_view: int = 10

    device: str = "cuda"
    seed: int = 42

    eval_interval: int = 1000
    log_interval: int = 10
    max_checkpoints: int = 3

    pi_early_stop_threshold: float = 0.99
    global_early_stop_steps: int = 100

