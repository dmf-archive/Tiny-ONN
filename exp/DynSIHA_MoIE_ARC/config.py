from dataclasses import dataclass, field
from typing import Literal


@dataclass
class MoeConfig:
    num_experts: int = 16
    min_experts_per_tok: int = 12
    intermediate_size: int = 128


@dataclass
class ModelConfig:
    vocab_size: int = 19
    hidden_size: int = 128
    num_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    max_position_embeddings: int = 9000
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    
    moe: MoeConfig = field(default_factory=MoeConfig)


@dataclass
class DataConfig:
    data_path: str = "data/ARC-AGI-2/data"
    batch_size: int = 1
    num_workers: int = 0


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    use_consistency_augmentation: bool = True
    gradient_accumulation_steps: int = 8
    training_paradigm: Literal["tf", "eavi"] = "eavi"

    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    num_epochs: int = 1
    
    device: str = "cuda"
    seed: int = 42

    save_interval: int = 10
    max_checkpoints: int = 3

    def __post_init__(self):
        if self.training_paradigm not in ["tf", "eavi"]:
            raise ValueError("training_paradigm must be 'tf' or 'eavi'")
