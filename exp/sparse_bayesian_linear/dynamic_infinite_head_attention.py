import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_bayesian_linear import SparseBayesianLinear

class DynamicInfiniteHeadAttention(nn.Module):
    def __init__(self, d_model: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_model = d_model
        self.sbl_qkv = SparseBayesianLinear(d_model, 3 * d_model, dtype=dtype)
        self.sbl_o = SparseBayesianLinear(d_model, d_model, dtype=dtype)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple, tuple]:
        qkv, s_qkv, m_qkv = self.sbl_qkv(x)
        q, k, v = torch.split(qkv, self.d_model, dim=-1)
        
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y, s_o, m_o = self.sbl_o(attn_out)
        return y, (s_qkv, s_o), (m_qkv, m_o)