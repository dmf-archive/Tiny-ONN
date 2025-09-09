import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple

from .config import CONFIG
from .bcat import BCAT

class SparseBayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.gate_param = nn.Parameter(torch.empty(out_features, dtype=dtype))
        self.mu_bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.mu_weight, a=math.sqrt(5))
        nn.init.normal_(self.sigma_weight, mean=0.0, std=0.5)
        nn.init.constant_(self.gate_param, -0.1)
        nn.init.zeros_(self.mu_bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        x_reshaped = x.view(-1, self.in_features)
        
        keys = self.mu_weight * F.softplus(self.sigma_weight)
        scores = torch.matmul(x_reshaped, keys.t()) / math.sqrt(self.in_features)
        raw_weights = F.relu(scores - self.gate_param.unsqueeze(0))

        bcat_obj = BCAT.cluster(
            raw_weights, 
            threshold=CONFIG["BCAT_CLUSTER_THRESHOLD"], 
            min_block_size=CONFIG["BCAT_MIN_BLOCK_SIZE"]
        )

        output = torch.zeros_like(raw_weights)
        for i, meta in enumerate(bcat_obj.block_meta):
            r_start, c_start, r_len, c_len = meta
            block_x = x_reshaped[r_start : r_start + r_len]
            block_mu = self.mu_weight[c_start : c_start + c_len]
            
            block_out = F.linear(block_x, block_mu)
            block_out *= bcat_obj.block_values[i]
            
            output[r_start : r_start + r_len, c_start : c_start + c_len] += block_out

        output += self.mu_bias.unsqueeze(0)
        
        final_output = output.view(*original_shape[:-1], self.out_features)
        return final_output, scores, output

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

class MoIETransformerBlock(nn.Module):
    def __init__(self, d_model, d_ffn_factor, dropout=0.1, dtype=torch.float32):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, dtype=dtype)
        self.attn = DynamicInfiniteHeadAttention(d_model, dtype=dtype)
        self.ln2 = nn.LayerNorm(d_model, dtype=dtype)
        self.ffn = DynamicInfiniteExpert(d_model, d_ffn_factor, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_in = self.ln1(x)
        attn_out, _, attn_masked_tuple = self.attn(attn_in)
        x = x + self.dropout(attn_out)
        
        ffn_in = self.ln2(x)
        ffn_out, _, ffn_masked_tuple = self.ffn(ffn_in)
        x = x + self.dropout(ffn_out)
        
        return x, list(attn_masked_tuple) + list(ffn_masked_tuple)

class ReferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(CONFIG["VOCAB_SIZE"], CONFIG["D_MODEL"], dtype=CONFIG["DTYPE"])
        self.pos_embedding = nn.Embedding(CONFIG["SEQ_LEN"], CONFIG["D_MODEL"], dtype=CONFIG["DTYPE"])
        self.blocks = nn.ModuleList([MoIETransformerBlock(
            CONFIG["D_MODEL"], CONFIG["D_FFN_FACTOR"], dtype=CONFIG["DTYPE"]
        ) for _ in range(CONFIG["NUM_TRANSFORMER_BLOCKS"])])
        self.lm_head = nn.Linear(CONFIG["D_MODEL"], CONFIG["VOCAB_SIZE"], dtype=CONFIG["DTYPE"])

    def forward(self, x):
        tok_emb = self.embedding(x)
        pos = torch.arange(0, x.size(1), device=x.device)
        pos_emb = self.pos_embedding(pos)
        x = tok_emb + pos_emb
        
        all_masked_outputs = []
        for block in self.blocks:
            x, masked_outputs_from_block = block(x)
            all_masked_outputs.extend(masked_outputs_from_block)
            
        return self.lm_head(x), all_masked_outputs