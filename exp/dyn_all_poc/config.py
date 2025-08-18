from dataclasses import dataclass

import torch

DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class UnifiedConfig:
    vocab_size: int = 5000
    max_seq_len: int = 256
    hidden_size: int = 256
    num_hidden_layers: int = 3
    bias: bool = False
    learning_rate: float = 1e-3
    epochs: int = 50
    batch_size: int = 1

    # DynSMHA specific
    max_attention_experts: int = 32
    min_attention_experts: int = 8
    head_dim: int = 32

    # DynMoE specific
    max_moe_experts: int = 32
    min_moe_experts: int = 8
    moe_intermediate_size: int = 32
    
    # Expert reborn specific
    k_reborn_experts: int = 32

    # Loss weights
    w_aux_smha: float = 1.0
    w_ce_smha: float = 1.0
    w_kl_smha: float = 1.0

    w_aux_moe: float = 1.0
    w_ce_moe: float = 1.0
    w_kl_moe: float = 1.0

    w_aux_se: float = 1.0
    w_ce_se: float = 1.0
    w_kl_se: float = 1.0

    # Predictive Integrity Score specific
    pi_alpha: float = 32
    pi_gamma: float = 0.5
