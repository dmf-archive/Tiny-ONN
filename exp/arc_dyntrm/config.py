from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 16
    hidden_size: int = 512
    max_refinement_steps: int = 8
    max_position_embeddings: int = 4096

    physical_num_heads: int = 4
    latent_attn_expert: int = 32

@dataclass
class GenerationConfig:
    use_dfs: bool = False
    min_prob: float = 0.1
    act_threshold: float = 0.6


@dataclass
class DataConfig:
    data_path: str = "data/ARC-AGI-2/data"
    batch_size: int = 1
    num_workers: int = 4
    warmup_dataset_ratio: float = 0.1


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    lr: float = 1e-3

    w_meta: float = 1.0
    w_act: float = 0.01

    num_epochs: int = 20
    num_augmentation_views: int = 4
    max_steps_per_view: int = 10

    device: str = "cuda"
    seed: int = 42

    eval_interval: int = 1000
    log_interval: int = 10
    max_checkpoints: int = 3

