import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .config import CONFIG

class BitSBL(nn.Module):
    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu_weight_latent = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.gate_param = nn.Parameter(torch.empty(out_features, dtype=dtype))
        self.mu_bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.mu_weight_latent, a=math.sqrt(5))
        nn.init.normal_(self.sigma_weight, mean=0.0, std=0.5)
        nn.init.constant_(self.gate_param, -0.1)
        nn.init.zeros_(self.mu_bias)

    def forward(self, x: torch.Tensor, prior_std: float, kl_epsilon: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        x_reshaped = x.view(-1, self.in_features)
        
        keys = self.mu_weight_latent * F.softplus(self.sigma_weight)
        scores = torch.matmul(x_reshaped, keys.t()) / math.sqrt(self.in_features)
        raw_weights = F.relu(scores - self.gate_param.unsqueeze(0))

        w_binary = torch.sign(self.mu_weight_latent)
        w_binary_ste = (w_binary - self.mu_weight_latent).detach() + self.mu_weight_latent
        
        computation_output = F.linear(x_reshaped, w_binary_ste, self.mu_bias)
        masked_output = computation_output * raw_weights
        
        new_shape = list(original_shape[:-1]) + [self.out_features]
        output = masked_output.view(new_shape)

        mu_q = self.mu_weight_latent
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
        self.sbl_qkv = torch.jit.script(BitSBL(d_model, 3 * d_model, dtype=dtype))
        self.sbl_o = torch.jit.script(BitSBL(d_model, d_model, dtype=dtype))

    def forward(self, x: torch.Tensor, prior_std: float, kl_epsilon: float) -> tuple[torch.Tensor, tuple, tuple, torch.Tensor]:
        qkv, s_qkv, m_qkv, kl_qkv = self.sbl_qkv(x, prior_std, kl_epsilon)
        q, k, v = torch.split(qkv, self.d_model, dim=-1)
        
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y, s_o, m_o, kl_o = self.sbl_o(attn_out, prior_std, kl_epsilon)
        total_kl = kl_qkv + kl_o
        return y, (s_qkv, s_o), (m_qkv, m_o), total_kl

class DynamicInfiniteExpert(nn.Module):
    def __init__(self, d_model: int, d_ffn_factor: int = 4, dtype: torch.dtype = torch.float32):
        super().__init__()
        d_ffn = d_model * d_ffn_factor
        self.sbl1 = torch.jit.script(BitSBL(d_model, d_ffn, dtype=dtype))
        self.sbl2 = torch.jit.script(BitSBL(d_ffn, d_model, dtype=dtype))

    def forward(self, x: torch.Tensor, prior_std: float, kl_epsilon: float) -> tuple[torch.Tensor, tuple, tuple, torch.Tensor]:
        h, s1, m1, kl1 = self.sbl1(x, prior_std, kl_epsilon)
        h_act = F.silu(h)
        y, s2, m2, kl2 = self.sbl2(h_act, prior_std, kl_epsilon)
        total_kl = kl1 + kl2
        return y, (s1, s2), (m1, m2), total_kl

class MoIETransformerBlock(nn.Module):
    def __init__(self, d_model, d_ffn_factor, dropout=0.1, dtype=torch.float32):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, dtype=dtype)
        self.attn = DynamicInfiniteHeadAttention(d_model, dtype=dtype)
        self.ln2 = nn.LayerNorm(d_model, dtype=dtype)
        self.ffn = DynamicInfiniteExpert(d_model, d_ffn_factor, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, prior_std: float, kl_epsilon: float):
        attn_in = self.ln1(x)
        attn_out, _, attn_masked_tuple, attn_kl = self.attn(attn_in, prior_std, kl_epsilon)
        x = x + self.dropout(attn_out)
        
        ffn_in = self.ln2(x)
        ffn_out, _, ffn_masked_tuple, ffn_kl = self.ffn(ffn_in, prior_std, kl_epsilon)
        x = x + self.dropout(ffn_out)
        
        total_kl = attn_kl + ffn_kl
        return x, list(attn_masked_tuple) + list(ffn_masked_tuple), total_kl

class ReferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(CONFIG["VOCAB_SIZE"], CONFIG["D_MODEL"], dtype=CONFIG["DTYPE"])
        self.pos_embedding = nn.Embedding(CONFIG["SEQ_LEN"], CONFIG["D_MODEL"], dtype=CONFIG["DTYPE"])
        self.blocks = nn.ModuleList([MoIETransformerBlock(
            CONFIG["D_MODEL"], CONFIG["D_FFN_FACTOR"], dtype=CONFIG["DTYPE"]
        ) for _ in range(CONFIG["NUM_TRANSFORMER_BLOCKS"])])
        self.lm_head = nn.Linear(CONFIG["D_MODEL"], CONFIG["VOCAB_SIZE"], dtype=CONFIG["DTYPE"])

    def forward(self, x, prior_std: float, kl_epsilon: float):
        tok_emb = self.embedding(x)
        pos = torch.arange(0, x.size(1), device=x.device)
        pos_emb = self.pos_embedding(pos)
        x = tok_emb + pos_emb
        
        all_masked_outputs = []
        total_kl_loss = torch.tensor(0.0, device=x.device, dtype=CONFIG["DTYPE"])
        for block in self.blocks:
            x, masked_outputs_from_block, kl_from_block = block(x, prior_std, kl_epsilon)
            all_masked_outputs.extend(masked_outputs_from_block)
            total_kl_loss += kl_from_block
            
        return self.lm_head(x), all_masked_outputs, total_kl_loss