import torch


class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    use_object_finder: bool = True

    MAX_GRID_SIZE = 10

    vocab_size: int = 11
    hidden_size: int = 256
    num_hidden_layers: int = 16
    max_position_embeddings: int = (MAX_GRID_SIZE * (MAX_GRID_SIZE + 1)) * 2 + 1

    max_attention_experts: int = 32
    min_attention_experts: int = 8
    head_dim: int = 32

    max_moe_experts: int = 32
    min_moe_experts: int = 8
    intermediate_size: int = 32

    w_aux_smha: float = 1.0
    w_aux_moe: float = 1.0

    pi_alpha: float = 16.0
    pi_gamma: float = 0.5

    BATCH_SIZE = 32 if TRAINING_MODE == 0 else 16
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.01
    CLIP_GRAD_NORM = 1.0
    CHECKPOINT_DIR = "exp/tiny_onn_arc/checkpoints"
    EPOCHS = 1000
    LOG_INTERVAL = 1
    EVAL_INTERVAL = 30
    EVAL_BATCHES = 1
    MAX_CHECKPOINTS = 3

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)