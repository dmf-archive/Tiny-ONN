from dataclasses import dataclass, field
from typing import Literal

@dataclass
class ModelConfig:
    vocab_size: int = 9007 # 7 control tokens + 30*30*10 grid tokens
    hidden_size: int = 384
    num_layers: int = 4
    max_position_embeddings: int = 4096 # With the new tokenizer, the sequence length will be smaller.
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

    learning_rate: float = 5e-3
    weight_decay: float = 0.1
    kl_learning_rate: float = 2e-2 # Higher LR for sigma parameters
    num_epochs: int = 100

    device: str = "cuda"
    seed: int = 42
    
    # SBL specific
    kl_prior_epsilon: float = 1e-9

    eval_interval: int = 100 # Eval less frequently
    log_interval: int = 10 # Log more frequently
    max_checkpoints: int = 3
