from dataclasses import dataclass, field
import torch

DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class DenseConfig:
    model_type: str = "dense"
    vocab_size: int = 50257
    max_seq_len: int = 256
    hidden_size: int = 128
    intermediate_size: int = 512
    num_attention_heads: int = 4
    
    learning_rate: float = 1e-4
    epochs: int = 50

@dataclass
class DynNSAConfig(DenseConfig):
    model_type: str = "dyn_nsa_v2"
    num_attention_heads: int = 32 # Increased from 4
    head_dim: int = 16

    sliding_window_size: int = 32
    compress_block_size: int = 8
    compress_block_sliding_stride: int = 4
    selection_block_size: int = 8
    num_selected_blocks: int = 8

    num_heads: int = field(init=False)
    
    gradient_accumulation_steps: int = 8
    w_aux: float = 1.0
    w_ce: float = 1.0
    w_kl: float = 1.0

    def __post_init__(self):
        self.num_heads = self.num_attention_heads