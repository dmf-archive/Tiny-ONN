from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 16
    hidden_size: int = 384
    num_layers: int = 12
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

    lr_main: float = 5e-3
    lr_meta: float = 3e-2
    
    w_proto: float = 1.0
    w_gate: float = 1.0
    base_decay: float = 1e-4
    num_epochs: int = 100
    gate_sigmoid_temperature: float = 1.0

    device: str = "cuda"
    seed: int = 42

    eval_interval: int = 10000 # Eval less frequently
    log_interval: int = 10 # Log more frequently
    max_checkpoints: int = 3

@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    do_sample: bool = False
    num_beams: int = 1
    top_p: float = 1.0
    top_k: int = 1
    temperature: float = 1.0
    eos_token_id: int | None = None
    num_return_sequences: int = 1
