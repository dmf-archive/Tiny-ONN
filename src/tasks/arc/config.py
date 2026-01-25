from dataclasses import dataclass, field


@dataclass
class GenerationConfig:
    use_dfs: bool = False
    min_prob: float = 0.01
    max_new_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 50

@dataclass
class ModelConfig:
    vocab_size: int = 17
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    max_seq_len: int = 8192

@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    lr: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 10
    device: str = "cuda"
    data_dir: str = "data/ARC-AGI-2/data"
    # Optimizer settings
    k: int = 1
    adaptive_sync: bool = True
    eval_steps: int = 100
    save_steps: int = 500
    resume_from_checkpoint: Optional[str] = None
    # Routing Shaping
    w_meta: float = 0.5
    cost_alpha: float = 0.5
