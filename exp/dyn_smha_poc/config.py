from dataclasses import dataclass, field
import torch

DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class DynSMHAConfig:
    vocab_size: int = 32000
    max_seq_len: int = 256
    hidden_size: int = 128
    
    num_experts: int = 16 # Number of SHA experts
    head_dim: int = 32   # Dimension of each SHA expert
    
    bias: bool = False
    
    learning_rate: float = 1e-3
    epochs: int = 20
    k_fallback: int = 8
    batch_size: int = 4
    
    # SurpriseMin Loss Weights
    w_aux: float = 1.0
    w_ce: float = 1.0
    w_kl: float = 0.8
    compress_ratio: int = 4