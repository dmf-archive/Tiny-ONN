import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

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

        activation_rate = (raw_weights > 0).float().mean()
        
        # Use BCAT only when sparse, otherwise fallback to dense
        if self.training and activation_rate < 0.75 and activation_rate > 0.0:
            bcat_obj = BCAT.cluster(raw_weights, grid_size=CONFIG["BCAT_GRID_SIZE"])
            
            if bcat_obj.block_meta.numel() > 0:
                block_r_starts, block_c_starts, block_r_lens, block_c_lens = bcat_obj.block_meta.T.long()
                
                # Gather indices for x and mu_weight
                x_indices = torch.cat([torch.arange(start, start + length, device=x.device) for start, length in zip(block_r_starts, block_r_lens)])
                mu_indices = torch.cat([torch.arange(start, start + length, device=x.device) for start, length in zip(block_c_starts, block_c_lens)])
                
                gathered_x = x_reshaped.index_select(0, x_indices)
                gathered_mu = self.mu_weight.index_select(0, mu_indices)
                
                # Pad for bmm
                x_splits = torch.split(gathered_x, block_r_lens.tolist())
                mu_splits = torch.split(gathered_mu, block_c_lens.tolist())
                padded_x = torch.nn.utils.rnn.pad_sequence(x_splits, batch_first=True)
                padded_mu = torch.nn.utils.rnn.pad_sequence(mu_splits, batch_first=True)
                
                # Batched matrix multiplication
                bmm_output = torch.bmm(padded_x, padded_mu.transpose(1, 2))
                
                # Mask and weight the output
                # Manually pad values to max_c_len
                max_r_len = block_r_lens.max().item()
                max_c_len = block_c_lens.max().item()
                padded_values_list = []
                value_offset = 0
                for r_len, c_len in zip(block_r_lens, block_c_lens):
                    num_block_values = r_len * c_len
                    block_vals = bcat_obj.block_values[value_offset : value_offset + num_block_values].view(r_len, c_len)
                    padded_block = F.pad(block_vals, (0, max_c_len - c_len, 0, 0))
                    padded_values_list.append(padded_block)
                    value_offset += num_block_values
                
                padded_values = torch.nn.utils.rnn.pad_sequence(padded_values_list, batch_first=True)
                bmm_output *= padded_values
                
                # Scatter add back to output tensor
                output = torch.zeros_like(raw_weights)
                
                # Create destination indices for scatter_add_
                rows = torch.cat([torch.arange(r_start, r_start + r_len, device=x.device).repeat_interleave(c_len) for r_start, r_len, c_len in zip(block_r_starts, block_r_lens, block_c_lens)])
                cols = torch.cat([torch.arange(c_start, c_start + c_len, device=x.device).repeat(r_len) for c_start, r_len, c_len in zip(block_c_starts, block_r_lens, block_c_lens)])
                
                # Flatten values to scatter
                bmm_values = torch.cat([bmm_output[i, :r_len, :c_len].flatten() for i, (r_len, c_len) in enumerate(zip(block_r_lens, block_c_lens))])

                # Use index_put_ for a more direct scatter operation
                output.index_put_((rows, cols), bmm_values, accumulate=True)

                masked_output = output + self.mu_bias.unsqueeze(0)
            else:
                 # Fallback if clustering yields no blocks
                computation_output = F.linear(x_reshaped, self.mu_weight)
                masked_output = computation_output * raw_weights
                masked_output += self.mu_bias.unsqueeze(0)
        else:
            computation_output = F.linear(x_reshaped, self.mu_weight)
            masked_output = computation_output * raw_weights
            masked_output += self.mu_bias.unsqueeze(0)
        
        final_output = masked_output.view(*original_shape[:-1], self.out_features)
        return final_output, scores, masked_output

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