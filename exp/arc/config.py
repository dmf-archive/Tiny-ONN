from typing import Optional
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

    eval_interval: int = 10000 # Eval less frequently
    log_interval: int = 10 # Log more frequently
    max_checkpoints: int = 3

@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    do_sample: bool = False
    num_beams: int = 1
    top_p: float = 1.0
    top_k: int = 50
    temperature: float = 1.0
    eos_token_id: Optional[int] = None
    num_return_sequences: int = 1
