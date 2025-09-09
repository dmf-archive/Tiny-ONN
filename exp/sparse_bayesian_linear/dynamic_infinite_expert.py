import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_bayesian_linear import SparseBayesianLinear

class DynamicInfiniteExpert(nn.Module):
    def __init__(self, d_model: int, d_ffn_factor: int = 4, dtype: torch.dtype = torch.float32):
        super().__init__()
        d_ffn = d_model * d_ffn_factor
        self.sbl1 = SparseBayesianLinear(d_model, d_ffn, dtype=dtype)
        self.sbl2 = SparseBayesianLinear(d_ffn, d_model, dtype=dtype)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple, tuple]:
        h, s1, m1 = self.sbl1(x)
        h_act = F.silu(h)
        y, s2, m2 = self.sbl2(h_act)
        return y, (s1, s2), (m1, m2)