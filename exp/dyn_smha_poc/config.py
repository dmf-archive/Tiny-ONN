from dataclasses import dataclass

import torch

DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class DynSMHAConfig:
    vocab_size: int = 5000
    max_seq_len: int = 256
    hidden_size: int = 128

    num_experts: int = 16
    head_dim: int = 32
    compress_block_size: int = 16

    bias: bool = False

    learning_rate: float = 1e-3
    epochs: int = 30
    batch_size: int = 4

    w_aux: float = 1.0
    w_ce: float = 1.0
    w_kl: float = 1.0
