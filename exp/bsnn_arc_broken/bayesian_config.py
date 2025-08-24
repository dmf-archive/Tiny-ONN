import torch


class BayesianConfig:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16

    MAX_GRID_SIZE = 10

    vocab_size: int = 10
    hidden_size: int = 256
    num_hidden_layers: int = 12

    pi_alpha: int = 8
    pi_gamma: float = 0.5

    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.0
    CLIP_GRAD_NORM = 1.0
    CHECKPOINT_DIR = "exp/tiny_onn_arc/checkpoints_bayesian"
    EPOCHS = 1000
    LOG_INTERVAL = 1
    EVAL_INTERVAL = 50
    EVAL_BATCHES = 4
    MAX_CHECKPOINTS = 3

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
