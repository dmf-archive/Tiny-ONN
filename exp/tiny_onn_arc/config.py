import torch


class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    use_object_finder: bool = True

    MAX_GRID_SIZE = 10

    from .tokenizer import ArcTokenizer
    vocab_size: int = ArcTokenizer.VOCAB_SIZE
    hidden_size: int = 128
    num_hidden_layers: int = 16
    max_position_embeddings: int = (MAX_GRID_SIZE * (MAX_GRID_SIZE + 1)) * 2 + 1

    max_attention_experts: int = 16
    min_attention_experts: int = 4
    fallback_attention_k: int = 16
    head_dim: int = 32

    max_moe_experts: int = 16
    min_moe_experts: int = 4
    fallback_moe_k: int = 16
    intermediate_size: int = 32

    # SMHA specific loss weights
    w_smha_diversity: float = 0.1
    w_smha_sparsity: float = 0.1

    # MoE specific loss weights
    w_moe_diversity: float = 0.1
    w_moe_sparsity: float = 0.1

    pi_alpha: float = 16.0 # For observation only
    pi_gamma: float = 0.5 # For observation only

    w_consistency_loss: float = 1.0
    sampling_temperature: float = 0.0
    sampling_top_p: float = 0.3

    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.01
    CLIP_GRAD_NORM = 1.0
    CHECKPOINT_DIR = "exp/tiny_onn_arc/checkpoints"
    EPOCHS = 1000
    LOG_INTERVAL = 10
    EVAL_INTERVAL = 100
    EVAL_BATCHES = 1
    MAX_CHECKPOINTS = 3

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
