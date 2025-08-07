import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import PretrainedConfig
from einops import rearrange
from typing import Optional

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16

@dataclass
class Config(PretrainedConfig):
    model_type = "dyn_nsa_poc"
    hidden_size: int = 32
    num_attention_heads: int = 4
    head_dim: int = 8
    vocab_size: int = 50257
    max_seq_len: int = 256
    learning_rate: float = 1e-3
    epochs: int = 10
    selection_block_size: int = 32
    max_selected_blocks: int = 8
    w_aux: float = 0.1
    w_entropy: float = 0.1
    w_sparse: float = 0.1

class DenseAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads)
        
        if attention_mask is not None and attention_mask.shape[-1] != q.shape[-2]:
             attention_mask = attention_mask[:,:,:q.shape[-2],:q.shape[-2]]

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        
        attn_output = rearrange(attn_output, 'b h t d -> b t (h d)')
        
        output = self.o_proj(attn_output)
        return output, torch.tensor(0.0, device=DEVICE)

class DenseModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.attn = DenseAttention(config)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        B, T = input_ids.shape
        causal_mask = torch.tril(torch.ones(T, T, device=DEVICE)).view(1, 1, T, T).bool()
        
        hidden_states = self.embedding(input_ids)
        attn_output, _ = self.attn(self.ln1(hidden_states), attention_mask=causal_mask)
        hidden_states = hidden_states + attn_output
        ffn_output = self.ffn(self.ln2(hidden_states))
        hidden_states = hidden_states + ffn_output
        logits = self.lm_head(hidden_states)
        return logits, torch.tensor(0.0, device=DEVICE)