import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

from .config import CONFIG
from typing import Optional

# ==== Standalone RoPE Implementation ====
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, dtype=torch.float32):   
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))       
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device, dtype=dtype)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

# ==== End Standalone RoPE Implementation ====

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

    def forward(self, x: torch.Tensor, prior_std: float, kl_epsilon: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        x_reshaped = x.view(-1, self.in_features)
        
        keys = self.mu_weight * F.softplus(self.sigma_weight)
        scores = torch.matmul(x_reshaped, keys.t()) / math.sqrt(self.in_features)
        raw_weights = F.relu(scores - self.gate_param.unsqueeze(0))

        computation_output = F.linear(x_reshaped, self.mu_weight, self.mu_bias)
        masked_output = computation_output * raw_weights
        
        new_shape = list(original_shape[:-1]) + [self.out_features]
        output = masked_output.view(new_shape)

        mu_q = self.mu_weight
        sigma_q = F.softplus(self.sigma_weight)
        var_q = sigma_q.pow(2)
        
        mu_p = torch.zeros_like(mu_q)
        var_p = torch.full_like(sigma_q, prior_std).pow(2)

        kl_div = 0.5 * (torch.log(var_p / (var_q + kl_epsilon)) + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1)
        kl_loss = kl_div.mean()
        
        return output, scores, masked_output, kl_loss

class DynamicInfiniteHeadAttention(nn.Module):
    def __init__(self, d_model: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_model = d_model
        self.sbl_qkv = SparseBayesianLinear(d_model, 3 * d_model, dtype=dtype)
        self.sbl_o = SparseBayesianLinear(d_model, d_model, dtype=dtype)

    def forward(self, x: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor], prior_std: float, kl_epsilon: float) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        qkv, s_qkv, m_qkv, kl_qkv = self.sbl_qkv(x, prior_std, kl_epsilon)
        
        batch_size, seq_len, _ = x.shape
        q, k, v = torch.split(qkv, self.d_model, dim=-1)

        cos, sin = position_embeddings
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        q = q.view(batch_size, seq_len, 1, self.d_model)
        k = k.view(batch_size, seq_len, 1, self.d_model)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = q.view(batch_size, seq_len, self.d_model)
        k = k.view(batch_size, seq_len, self.d_model)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y, s_o, m_o, kl_o = self.sbl_o(attn_out, prior_std, kl_epsilon)
        total_kl = kl_qkv + kl_o
        return y, (s_qkv, s_o), (m_qkv, m_o), total_kl

class DynamicInfiniteExpert(nn.Module):
    def __init__(self, d_model: int, d_ffn_factor: int = 4, dtype: torch.dtype = torch.float32):
        super().__init__()
        d_ffn = d_model * d_ffn_factor
        self.sbl1 = SparseBayesianLinear(d_model, d_ffn, dtype=dtype)
        self.sbl2 = SparseBayesianLinear(d_ffn, d_model, dtype=dtype)

    def forward(self, x: torch.Tensor, prior_std: float, kl_epsilon: float) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        h, s1, m1, kl1 = self.sbl1(x, prior_std, kl_epsilon)
        h_act = F.silu(h)
        y, s2, m2, kl2 = self.sbl2(h_act, prior_std, kl_epsilon)
        total_kl = kl1 + kl2
        return y, (s1, s2), (m1, m2), total_kl

class MoIETransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ffn_factor: int, dropout: float = 0.1, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, dtype=dtype)
        self.attn = DynamicInfiniteHeadAttention(d_model, dtype=dtype)
        self.ln2 = nn.LayerNorm(d_model, dtype=dtype)
        self.ffn = DynamicInfiniteExpert(d_model, d_ffn_factor, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor], prior_std: float, kl_epsilon: float) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        
        def run_attn(x, pos_emb, p_std, kl_eps):
            attn_in = self.ln1(x)
            attn_out, attn_scores, attn_masked, attn_kl = self.attn(attn_in, pos_emb, p_std, kl_eps)       
            return x + self.dropout(attn_out), attn_masked, attn_kl

        def run_ffn(x, p_std, kl_eps):
            ffn_in = self.ln2(x)
            ffn_out, ffn_scores, ffn_masked, ffn_kl = self.ffn(ffn_in, p_std, kl_eps)
            return x + self.dropout(ffn_out), ffn_masked, ffn_kl

        if self.training and self.gradient_checkpointing:
            x.requires_grad_()
            
            def create_custom_forward(module_fn):
                def custom_forward(*inputs):
                    output, masked_list, kl_val = module_fn(*inputs)
                    return (output, kl_val) + tuple(masked_list)
                return custom_forward

            def unpack_custom_output(outputs):
                output, kl_val = outputs[0], outputs[1]
                masked_list = list(outputs[2:])
                return output, masked_list, kl_val

            attn_outputs = checkpoint(create_custom_forward(lambda *inputs: run_attn(*inputs)), x, position_embeddings, torch.tensor(prior_std), torch.tensor(kl_epsilon), use_reentrant=False)
            x, attn_masked_tuple, attn_kl = unpack_custom_output(attn_outputs)

            ffn_outputs = checkpoint(create_custom_forward(lambda *inputs: run_ffn(*inputs)), x, torch.tensor(prior_std), torch.tensor(kl_epsilon), use_reentrant=False)
            x, ffn_masked_tuple, ffn_kl = unpack_custom_output(ffn_outputs)

        else:
            x, attn_masked_tuple, attn_kl = run_attn(x, position_embeddings, prior_std, kl_epsilon)        
            x, ffn_masked_tuple, ffn_kl = run_ffn(x, prior_std, kl_epsilon)

        total_kl = attn_kl + ffn_kl
        masked_outputs = list(attn_masked_tuple) + list(ffn_masked_tuple)
        return x, masked_outputs, total_kl

class ReferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(CONFIG["VOCAB_SIZE"], CONFIG["D_MODEL"], dtype=CONFIG["DTYPE"])      
        self.rotary_emb = RotaryEmbedding(
            dim=CONFIG["D_MODEL"],
            max_position_embeddings=CONFIG["SEQ_LEN"],
            device=CONFIG["DEVICE"],
            dtype=CONFIG["DTYPE"]
        )
        self.blocks = nn.ModuleList([MoIETransformerBlock(
            CONFIG["D_MODEL"], CONFIG["D_FFN_FACTOR"], dtype=CONFIG["DTYPE"]
        ) for _ in range(CONFIG["NUM_TRANSFORMER_BLOCKS"])])
        self.lm_head = nn.Linear(CONFIG["D_MODEL"], CONFIG["VOCAB_SIZE"], dtype=CONFIG["DTYPE"])
        self.norm = nn.LayerNorm(CONFIG["D_MODEL"], dtype=CONFIG["DTYPE"])

    def gradient_checkpointing_enable(self):
        for block in self.blocks:
            block.gradient_checkpointing = True

    def forward(self, x, prior_std: float, kl_epsilon: float):
        x = self.embedding(x)
        seq_len = x.shape[1]
        
        position_embeddings = self.rotary_emb(x, seq_len=seq_len)

        all_masked_outputs = []
        total_kl_loss = torch.tensor(0.0, device=x.device, dtype=CONFIG["DTYPE"])

        for block in self.blocks:
            x, masked_outputs_from_block, kl_from_block = block(x, position_embeddings, prior_std, kl_epsilon)
            all_masked_outputs.extend(masked_outputs_from_block)
            total_kl_loss += kl_from_block

        x = self.norm(x)
        return self.lm_head(x), all_masked_outputs, total_kl_loss