from pydantic import BaseModel


class ModelConfig(BaseModel):
    model_type: str
    hidden_size: int
    num_layers: int
    num_heads: int
    physical_num_heads: int
    latent_attn_expert: int
    vocab_size: int
    max_position_embeddings: int
    ffn_scale: int

class ARSConfig(BaseModel):
    rho: float
    k: int
    alpha: float
    adaptive: bool

class OptimizerConfig(BaseModel):
    type: str
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    ns_steps: int
    ars: ARSConfig

class DataConfig(BaseModel):
    data_path: str
    batch_size: int
    num_workers: int
    warmup_dataset_ratio: float
    num_augmentation_views: int

class TrainingConfig(BaseModel):
    seed: int
    num_epochs: int
    max_steps_per_view: int
    w_meta: float
    max_checkpoints: int
    pi_threshold: float | None = None

class ExperimentConfig(BaseModel):
    model: ModelConfig
    optimizer: OptimizerConfig
    data: DataConfig
    training: TrainingConfig
