import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


@torch.jit.script
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@torch.jit.script
def mas_normalize(logits: torch.Tensor) -> torch.Tensor:
    max_abs_val = torch.max(torch.abs(logits), dim=-1, keepdim=True).values
    scaled_logits = logits / (max_abs_val + 1e-9)
    return F.relu(scaled_logits)


@torch.jit.script
def mas_normalize_negative(logits: torch.Tensor) -> torch.Tensor:
    return -F.relu(-mas_normalize(logits))


@torch.jit.script
def spl_forward(
    x: torch.Tensor,
    effective_proto: torch.Tensor,
    mu_weight: torch.Tensor,
    mu_bias: torch.Tensor,
    gate_param: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model_dtype = mu_weight.dtype
    x = x.to(model_dtype)
    effective_proto = effective_proto.to(model_dtype)
    gate_param = gate_param.to(model_dtype)

    match_values = F.linear(x, effective_proto) / math.sqrt(x.size(-1))
    gate_logit = torch.matmul(x, gate_param.t())
    computation_output = F.linear(x, mu_weight, mu_bias)

    return computation_output, match_values, gate_logit


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
        )
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
        nn.init.zeros_(self.gate_param)

    def forward(
        self, x: torch.Tensor, effective_proto: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return spl_forward(x, effective_proto, self.mu_weight, self.mu_bias, self.gate_param)


class DynamicInfiniteHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_model = config.hidden_size
        self.spl_q = SparseProtoLinear(self.d_model, self.d_model, dtype=dtype)
        self.spl_k = SparseProtoLinear(self.d_model, self.d_model, dtype=dtype)
        self.spl_v = SparseProtoLinear(self.d_model, self.d_model, dtype=dtype)
        self.spl_o = SparseProtoLinear(self.d_model, self.d_model, dtype=dtype)


class DynamicInfiniteExpert(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        d_ffn = config.hidden_size * config.d_ffn_factor
        self.spl1 = SparseProtoLinear(config.hidden_size, d_ffn, dtype=dtype)
        self.spl2 = SparseProtoLinear(d_ffn, config.hidden_size, dtype=dtype)


class MoIETransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.attn = DynamicInfiniteHeadAttention(config, dtype=dtype)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.ffn = DynamicInfiniteExpert(config, dtype=dtype)
        self.routing_gain = config.routing_gain
        d_ffn = config.hidden_size * config.d_ffn_factor
        self.proto_transforms = nn.ModuleDict(
            {
                "attn_q": nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=dtype),
                "attn_k": nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=dtype),
                "attn_v": nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=dtype),
                "attn_o": nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=dtype),
                "ffn_spl1": nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=dtype),
                "ffn_spl2": nn.Linear(d_ffn, d_ffn, bias=False, dtype=dtype),
            }
        )
        self.proto_layernorms = nn.ModuleDict(
            {
                "attn_q": nn.LayerNorm(config.hidden_size, eps=1e-5),
                "attn_k": nn.LayerNorm(config.hidden_size, eps=1e-5),
                "attn_v": nn.LayerNorm(config.hidden_size, eps=1e-5),
                "attn_o": nn.LayerNorm(config.hidden_size, eps=1e-5),
                "ffn_spl1": nn.LayerNorm(config.hidden_size, eps=1e-5),
                "ffn_spl2": nn.LayerNorm(d_ffn, eps=1e-5),
            }
        )

    def forward(
        self, x: torch.Tensor, pos_emb: tuple, past_kv: tuple | None = None, prev_protos: dict | None = None
    ) -> tuple:
        effective_protos = {}
        spl_modules = {
            "attn_q": self.attn.spl_q,
            "attn_k": self.attn.spl_k,
            "attn_v": self.attn.spl_v,
            "attn_o": self.attn.spl_o,
            "ffn_spl1": self.ffn.spl1,
            "ffn_spl2": self.ffn.spl2,
        }
        for name, module in spl_modules.items():
            if prev_protos is not None and name in prev_protos:
                residual = self.proto_layernorms[name](self.proto_transforms[name](prev_protos[name]))
                effective_protos[name] = module.proto_weight + residual
            else:
                effective_protos[name] = module.proto_weight

        ln1_out = self.ln1(x)
        c_q, mv_q, pc_q = self.attn.spl_q(ln1_out, effective_protos["attn_q"])
        c_q = ln1_out + c_q
        c_k, mv_k, pc_k = self.attn.spl_k(ln1_out, effective_protos["attn_k"])
        c_k = ln1_out + c_k
        c_v, mv_v, pc_v = self.attn.spl_v(ln1_out, effective_protos["attn_v"])
        c_v = ln1_out + c_v

        ln2_out = self.ln2(x)
        c1, mv1, pc1 = self.ffn.spl1(ln2_out, effective_protos["ffn_spl1"])
        c1 = ln2_out + c1

        dummy_h_act = torch.zeros(
            ln2_out.shape[0],
            ln2_out.shape[1],
            self.ffn.spl2.in_features,
            device=x.device,
            dtype=x.dtype,
        )
        _, _, pc2_pre = self.ffn.spl2(dummy_h_act, effective_protos["ffn_spl2"])

        dummy_attn_out = torch.zeros_like(x)
        _, _, pc_o_pre = self.attn.spl_o(dummy_attn_out, effective_protos["attn_o"])


        all_masked, all_comp, all_raw, all_routing_logits, all_spl_inputs = [], [], [], [], []

        comp_qkv, match_qkv, costs_qkv = [c_q, c_k, c_v], [mv_q, mv_k, mv_v], [pc_q, pc_k, pc_v]
        q, k, v = torch.zeros_like(c_q), torch.zeros_like(c_k), torch.zeros_like(c_v)

        for i in range(3):
            cost_score = mas_normalize(costs_qkv[i])
            routing_logits = (match_qkv[i] - cost_score) * self.routing_gain
            raw_weights = mas_normalize(routing_logits)
            masked = comp_qkv[i] * raw_weights
            if i == 0: q = masked
            elif i == 1: k = masked
            else: v = masked
            all_masked.append(masked)
            all_comp.append(comp_qkv[i])
            all_raw.append(raw_weights)
            all_routing_logits.append(routing_logits)

        cos, sin = pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=1)
            v = torch.cat([past_kv[1], v], dim=1)
        present_kv = (k, v)
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=past_kv is None)
        attn_out = attn_out.squeeze(1)

        c_o, mv_o, pc_o = self.attn.spl_o(attn_out, effective_protos["attn_o"])
        c_o = attn_out + c_o
        cost_score_o = mas_normalize(pc_o)
        routing_logits_o = (mv_o - cost_score_o) * self.routing_gain
        rw_o = mas_normalize(routing_logits_o)
        m_o = c_o * rw_o
        x = x + m_o

        cost_score_f1 = mas_normalize(pc1)
        routing_logits_f1 = (mv1 - cost_score_f1) * self.routing_gain
        rw_f1 = mas_normalize(routing_logits_f1)
        m1 = c1 * rw_f1
        h_act = F.relu(m1)

        c2, mv2, pc2 = self.ffn.spl2(h_act, effective_protos["ffn_spl2"])
        c2 = h_act + c2
        cost_score_f2 = mas_normalize(pc2)
        routing_logits_f2 = (mv2 - cost_score_f2) * self.routing_gain
        rw_f2 = mas_normalize(routing_logits_f2)
        m2 = c2 * rw_f2
        x_out = x + m2

        all_masked.extend([m_o, m1, m2])
        all_comp.extend([c_o, c1, c2])
        all_raw.extend([rw_o, rw_f1, rw_f2])
        all_routing_logits.extend([routing_logits_o, routing_logits_f1, routing_logits_f2])
        all_spl_inputs.extend([ln1_out] * 3 + [attn_out, ln2_out, h_act])

        return x_out, all_masked, all_comp, all_raw, all_spl_inputs, all_routing_logits, present_kv, effective_protos


class ArcEmbedding(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.color_embedding = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.row_embedding = nn.Embedding(31, config.hidden_size, dtype=dtype)
        self.col_embedding = nn.Embedding(31, config.hidden_size, dtype=dtype)

    def forward(self, input_ids: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        color_embed = self.color_embedding(input_ids)

        row_coords = coords[..., 0]
        col_coords = coords[..., 1]

        row_embed = self.row_embedding(row_coords.clamp(min=0, max=30))
        col_embed = self.col_embedding(col_coords.clamp(min=0, max=30))

        is_special_token = (coords[..., 0] == -1).unsqueeze(-1)

        pos_embed = row_embed + col_embed
        final_embed = color_embed + torch.where(is_special_token, torch.zeros_like(pos_embed), pos_embed)

        return final_embed

class ArcTransformer(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device | str):
        super().__init__()
        self.config, self.device = config, device
        dtype = torch.float32
        self.embedding = torch.jit.script(ArcEmbedding(config, dtype=dtype))
        self.rotary_emb = RotaryEmbedding(
            dim=config.hidden_size, max_position_embeddings=config.max_position_embeddings, device=device, dtype=dtype
        )
        self.blocks = nn.ModuleList([MoIETransformerBlock(config, dtype=dtype) for _ in range(config.num_layers)])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype)

    def forward(self, input_ids: torch.Tensor, coords: torch.Tensor, past_key_values: list | None = None, return_dict: bool = False):
        x = self.embedding(input_ids, coords)
        pos_emb = self.rotary_emb(x, seq_len=input_ids.size(1))
        past_key_values = past_key_values if past_key_values is not None else [None] * len(self.blocks)

        all_masked, all_comp, all_spl_in, all_raw, all_protos, all_routing_logits, presents = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        prev_protos = None

        for i, block in enumerate(self.blocks):
            (
                x,
                masked,
                comp,
                raw,
                spl_inputs,
                routing_logits,
                present_kv,
                effective_protos,
            ) = block(x, pos_emb, past_key_values[i], prev_protos)
            presents.append(present_kv)
            all_masked.extend(masked)
            all_comp.extend(comp)
            all_spl_in.extend(spl_inputs)
            all_raw.extend(raw)
            all_protos.append(effective_protos)
            all_routing_logits.extend(routing_logits)
            prev_protos = effective_protos

        logits = self.lm_head(x)

        if not return_dict:
            return logits, x, all_masked, all_comp, all_protos, all_spl_in, all_raw, all_routing_logits, presents

        return {
            "logits": logits,
            "hidden_states": x,
            "masked_outputs": all_masked,
            "computation_outputs": all_comp,
            "proto_states": all_protos,
            "spl_inputs": all_spl_in,
            "raw_weights": all_raw,
            "routing_logits": all_routing_logits,
            "past_key_values": presents,
        }
