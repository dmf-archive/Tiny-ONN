import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

@torch.jit.script
def ste(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return gate.detach() - value.detach() + value

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

@torch.jit.script
def spl_forward(
    x: torch.Tensor,
    effective_proto: torch.Tensor,
    mu_weight: torch.Tensor,
    mu_bias: torch.Tensor,
    gate_param: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_norm = F.normalize(x, p=2.0, dim=-1)
    proto_norm = F.normalize(effective_proto, p=2.0, dim=-1)
    match_values = F.linear(x_norm, proto_norm)
    predicted_cost = torch.abs(torch.matmul(x, gate_param.t()))
    raw_weights = F.relu(match_values - predicted_cost)
    gate_binary = (raw_weights > 0).to(x.dtype)
    gated_raw_weights = ste(gate_binary, raw_weights)
    computation_output = F.linear(x, mu_weight, mu_bias)
    gated_output = computation_output * gated_raw_weights
    return gated_output, computation_output, raw_weights, predicted_cost

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000, device: torch.device | None = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).to(dtype=x.dtype)
        sin = emb.sin().unsqueeze(0).to(dtype=x.dtype)
        return cos, sin

class SparseProtoLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.proto_weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.mu_bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        self.gate_param = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.mu_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.proto_weight, a=math.sqrt(5))
        nn.init.zeros_(self.mu_bias)
        nn.init.kaiming_uniform_(self.gate_param, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, effective_proto: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return spl_forward(x, effective_proto, self.mu_weight, self.mu_bias, self.gate_param)

class DynamicInfiniteHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_model = config.hidden_size
        self.sbl_qkv = SparseProtoLinear(self.d_model, 3 * self.d_model, dtype=dtype)
        self.sbl_o = SparseProtoLinear(self.d_model, self.d_model, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        effective_protos: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list, list, list, list, list, tuple]:
        if effective_protos is None: raise ValueError("effective_protos cannot be None")
        m_qkv, c_qkv, rw_qkv, pc_qkv = self.sbl_qkv(x, effective_protos["attn_qkv"])
        q, k, v = torch.split(m_qkv, self.d_model, dim=-1)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=past_key_value is None)
        m_o, c_o, rw_o, pc_o = self.sbl_o(attn_out, effective_protos["attn_o"])
        return m_o, [m_qkv, m_o], [c_qkv, c_o], [rw_qkv, rw_o], [x, attn_out], [pc_qkv, pc_o], (k, v)

class DynamicInfiniteExpert(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        d_ffn = config.hidden_size * config.d_ffn_factor
        self.sbl1 = SparseProtoLinear(config.hidden_size, d_ffn, dtype=dtype)
        self.sbl2 = SparseProtoLinear(d_ffn, config.hidden_size, dtype=dtype)

    def forward(self, x: torch.Tensor, effective_protos: dict[str, torch.Tensor]) -> tuple[torch.Tensor, list, list, list, list, list]:
        if effective_protos is None: raise ValueError("effective_protos cannot be None")
        m1, c1, rw1, pc1 = self.sbl1(x, effective_protos["ffn_sbl1"])
        h_act = F.relu(m1)
        m2, c2, rw2, pc2 = self.sbl2(h_act, effective_protos["ffn_sbl2"])
        return m2, [m1, m2], [c1, c2], [rw1, rw2], [x, h_act], [pc1, pc2]

class MoIETransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, dtype=dtype)
        self.attn = DynamicInfiniteHeadAttention(config, dtype=dtype)
        self.ln2 = nn.LayerNorm(config.hidden_size, dtype=dtype)
        self.ffn = DynamicInfiniteExpert(config, dtype=dtype)
        d_ffn = config.hidden_size * config.d_ffn_factor
        self.proto_transforms = nn.ModuleDict({
            "attn_qkv": nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=dtype),
            "attn_o": nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=dtype),
            "ffn_sbl1": nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=dtype),
            "ffn_sbl2": nn.Linear(d_ffn, d_ffn, bias=False, dtype=dtype),
        })
        self.proto_layernorms = nn.ModuleDict({
            "attn_qkv": nn.LayerNorm(config.hidden_size, dtype=dtype),
            "attn_o": nn.LayerNorm(config.hidden_size, dtype=dtype),
            "ffn_sbl1": nn.LayerNorm(config.hidden_size, dtype=dtype),
            "ffn_sbl2": nn.LayerNorm(d_ffn, dtype=dtype),
        })

    def forward(self, x: torch.Tensor, pos_emb: tuple, past_kv: tuple | None = None, prev_protos: dict | None = None) -> tuple:
        effective_protos = {}
        sbl_modules = {"attn_qkv": self.attn.sbl_qkv, "attn_o": self.attn.sbl_o, "ffn_sbl1": self.ffn.sbl1, "ffn_sbl2": self.ffn.sbl2}
        for name, module in sbl_modules.items():
            if prev_protos is not None and name in prev_protos:
                residual = self.proto_layernorms[name](self.proto_transforms[name](prev_protos[name]))
                effective_protos[name] = module.proto_weight + residual
            else:
                effective_protos[name] = module.proto_weight
        
        attn_out, attn_m, attn_c, attn_rw, attn_in, attn_pc, present_kv = self.attn(self.ln1(x), pos_emb, past_kv, effective_protos)
        x = x + attn_out
        ffn_out, ffn_m, ffn_c, ffn_rw, ffn_in, ffn_pc = self.ffn(self.ln2(x), effective_protos)
        x_out = x + ffn_out
        
        return x_out, attn_m + ffn_m, attn_c + ffn_c, attn_rw + ffn_rw, attn_in + ffn_in, attn_pc + ffn_pc, present_kv, effective_protos

class ArcTransformer(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device | str):
        super().__init__()
        self.config, self.device, dtype = config, device, torch.bfloat16
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.rotary_emb = RotaryEmbedding(dim=config.hidden_size, max_position_embeddings=config.max_position_embeddings, dtype=dtype, device=device)
        self.blocks = nn.ModuleList([MoIETransformerBlock(config, dtype=dtype) for _ in range(config.num_layers)])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, dtype=dtype)

    def forward(self, input_ids: torch.Tensor, past_key_values: list | None = None):
        x = self.embedding(input_ids)
        pos_emb = self.rotary_emb(x, seq_len=input_ids.size(1))
        past_key_values = past_key_values if past_key_values is not None else [None] * len(self.blocks)
        
        all_masked, all_comp, all_sbl_in, all_raw, all_protos, all_pred_costs, presents = [], [], [], [], [], [], []
        prev_protos = None
        
        for i, block in enumerate(self.blocks):
            x, masked, comp, raw, sbl_inputs, pred_costs, present_kv, effective_protos = block(x, pos_emb, past_key_values[i], prev_protos)
            presents.append(present_kv)
            all_masked.extend(masked)
            all_comp.extend(comp)
            all_sbl_in.extend(sbl_inputs)
            all_raw.extend(raw)
            all_protos.append(effective_protos)
            all_pred_costs.extend(pred_costs)
            prev_protos = effective_protos

        logits = self.lm_head(x)
        return logits, x, all_masked, all_comp, all_protos, all_sbl_in, all_raw, all_pred_costs, presents
