import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


@torch.jit.script
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

@torch.jit.script
def apply_rope(q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor, hidden_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Standard 1D RoPE
    seq_len = position_ids.max() + 1
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, hidden_size, 2, device=q.device).float() / hidden_size))
    t = torch.arange(seq_len, device=q.device).type_as(inv_freq)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    
    cos = emb.cos()[position_ids, :].unsqueeze(0).unsqueeze(0)
    sin = emb.sin()[position_ids, :].unsqueeze(0).unsqueeze(0)

    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


@torch.jit.script
def mas_normalize_jit(logits: torch.Tensor) -> torch.Tensor:
    max_abs_val = torch.max(torch.abs(logits), dim=-1, keepdim=True).values
    scaled_logits = logits / (max_abs_val + 1e-9)
    return scaled_logits


@torch.jit.script
def spl_forward(
    x: torch.Tensor,
    proto_state: torch.Tensor,
    mu_weight: torch.Tensor,
    mu_bias: torch.Tensor,
    gate_param: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model_dtype = mu_weight.dtype
    x = x.to(model_dtype)
    proto_state = proto_state.to(model_dtype)
    gate_param = gate_param.to(model_dtype)

    match_values = F.linear(x, proto_state) / math.sqrt(x.size(-1))
    gate_logit = gate_param
    computation_output = F.silu(F.linear(x, mu_weight, mu_bias))

    return computation_output, match_values, gate_logit


class SparseProtoLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.proto_weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.mu_bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        self.gate_param = nn.Parameter(torch.empty(out_features, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.mu_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.proto_weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.mu_bias)
        nn.init.zeros_(self.gate_param)

    def forward(
        self, x: torch.Tensor, proto_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return spl_forward(x, proto_state, self.mu_weight, self.mu_bias, self.gate_param)


class DynamicInfiniteHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_model = config.hidden_size
        self.spl_q = SparseProtoLinear(self.d_model, self.d_model, dtype=dtype)
        self.spl_k = SparseProtoLinear(self.d_model, self.d_model, dtype=dtype)
        self.spl_v = SparseProtoLinear(self.d_model, self.d_model, dtype=dtype)
        self.spl_o = SparseProtoLinear(self.d_model, self.d_model, dtype=dtype)


class MoIETransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.d_model = config.hidden_size
        self.attn = DynamicInfiniteHeadAttention(config, dtype=dtype)
        self.proto_transforms = nn.ModuleDict(
            {
                "attn_q": nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=dtype),
                "attn_k": nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=dtype),
                "attn_v": nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=dtype),
                "attn_o": nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=dtype),
            }
        )
        self.proto_layernorms = nn.ModuleDict(
            {
                "attn_q": nn.LayerNorm(config.hidden_size, eps=1e-5),
                "attn_k": nn.LayerNorm(config.hidden_size, eps=1e-5),
                "attn_v": nn.LayerNorm(config.hidden_size, eps=1e-5),
                "attn_o": nn.LayerNorm(config.hidden_size, eps=1e-5),
            }
        )

    def forward(
        self,
        x: torch.Tensor,
        coords_2d: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_kv: tuple | None = None,
        incoming_proto_state: dict | None = None,
    ) -> tuple:
        outgoing_proto_state = {}
        spl_modules = {
            "attn_q": self.attn.spl_q,
            "attn_k": self.attn.spl_k,
            "attn_v": self.attn.spl_v,
            "attn_o": self.attn.spl_o,
        }
        for name, module in spl_modules.items():
            if incoming_proto_state is not None and name in incoming_proto_state:
                prc_residual = self.proto_layernorms[name](self.proto_transforms[name](incoming_proto_state[name]))
                outgoing_proto_state[name] = module.proto_weight + prc_residual
            else:
                outgoing_proto_state[name] = module.proto_weight

        ln1_out = self.ln1(x)
        
        # --- Sequence-Level Routing ---
        context_vector = ln1_out.mean(dim=1) # (B, D)
        
        # We need to compute sequence-level logits for all SPL modules
        # This requires calling them with the context_vector
        # Let's compute computation outputs first, as they are per-token
        c_q, _, _ = self.attn.spl_q(ln1_out, outgoing_proto_state["attn_q"])
        c_k, _, _ = self.attn.spl_k(ln1_out, outgoing_proto_state["attn_k"])
        c_v, _, _ = self.attn.spl_v(ln1_out, outgoing_proto_state["attn_v"])

        # Now compute sequence-level routing logits
        _, mv_q_seq, pc_q_seq = self.attn.spl_q(context_vector, outgoing_proto_state["attn_q"])
        _, mv_k_seq, pc_k_seq = self.attn.spl_k(context_vector, outgoing_proto_state["attn_k"])
        _, mv_v_seq, pc_v_seq = self.attn.spl_v(context_vector, outgoing_proto_state["attn_v"])

        all_masked, all_comp, all_masked_routing_logits, all_routing_logits, all_spl_inputs = [], [], [], [], []

        comp_qkv = [c_q, c_k, c_v]
        match_qkv_seq = [mv_q_seq, mv_k_seq, mv_v_seq]
        costs_qkv_seq = [pc_q_seq, pc_k_seq, pc_v_seq]
        q, k, v = torch.zeros_like(c_q), torch.zeros_like(c_k), torch.zeros_like(c_v)

        for i in range(3):
            # Sequence-level logits
            routing_logits_seq = match_qkv_seq[i] + costs_qkv_seq[i] # Shape: (B, D_out)
            gating_weights_seq = mas_normalize_jit(routing_logits_seq) # Shape: (B, D_out)

            # Broadcast gating weights to all tokens
            masked_output = gating_weights_seq.unsqueeze(1) * comp_qkv[i] # (B, 1, D_out) * (B, T, D_out)

            if i == 0:
                q = masked_output
            elif i == 1:
                k = masked_output
            else:
                v = masked_output

            all_masked.append(masked_output)
            all_comp.append(comp_qkv[i])
            all_masked_routing_logits.append(gating_weights_seq)
            all_routing_logits.append(routing_logits_seq)

        batch_size, seq_len, _ = q.shape
        num_heads = 1
        head_dim = self.d_model

        if position_ids is not None:
            q, k = apply_rope(q, k, position_ids, self.d_model)

        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        present_kv = (k, v)

        is_causal = past_kv is None
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Sequence-level routing for output projection
        context_attn_out = attn_out.mean(dim=1)
        c_o, _, _ = self.attn.spl_o(attn_out, outgoing_proto_state["attn_o"])
        _, mv_o_seq, pc_o_seq = self.attn.spl_o(context_attn_out, outgoing_proto_state["attn_o"])
        
        routing_logits_o_seq = mv_o_seq + pc_o_seq
        gating_weights_o_seq = mas_normalize_jit(routing_logits_o_seq)
        masked_output_o = gating_weights_o_seq.unsqueeze(1) * c_o

        x_out = x + masked_output_o

        all_masked.append(masked_output_o)
        all_comp.append(c_o)
        all_masked_routing_logits.append(gating_weights_o_seq)
        all_routing_logits.append(routing_logits_o_seq)
        all_routing_logits.append(routing_logits_o_seq)
        all_spl_inputs.extend([ln1_out] * 3 + [attn_out])


        return x_out, all_masked, all_comp, all_masked_routing_logits, all_spl_inputs, all_routing_logits, present_kv, outgoing_proto_state


class ArcEmbedding(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.color_embedding = nn.Embedding(config.vocab_size, self.hidden_size, dtype=dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.color_embedding(input_ids)

class ArcTransformer(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device | str):
        super().__init__()
        self.config, self.device = config, device
        dtype = torch.float32
        self.embedding = ArcEmbedding(config, dtype=dtype)
        self.blocks = nn.ModuleList([MoIETransformerBlock(config, dtype=dtype) for _ in range(config.num_layers)])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        coords: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list | None = None,
        return_dict: bool = False,
    ):
        x = self.embedding(input_ids)

        if position_ids is None:
            past_seen_tokens = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + x.shape[1], device=x.device
            ).unsqueeze(0)

        past_key_values = past_key_values if past_key_values is not None else [None] * len(self.blocks)
        all_masked, all_comp, all_spl_in, all_masked_routing_logits, all_protos, all_routing_logits, presents = (
            [], [], [], [], [], [], [],
        )
        incoming_proto_state = None

        for i, block in enumerate(self.blocks):
            (
                x,
                masked,
                comp,
                masked_routing_logits,
                spl_inputs,
                routing_logits,
                present_kv,
                outgoing_proto_state,
            ) = block(
                x,
                coords,
                position_ids,
                past_key_values[i],
                incoming_proto_state,
            )
            presents.append(present_kv)
            all_masked.extend(masked)
            all_comp.extend(comp)
            all_spl_in.extend(spl_inputs)
            all_masked_routing_logits.extend(masked_routing_logits)
            all_protos.append(outgoing_proto_state)
            all_routing_logits.extend(routing_logits)
            incoming_proto_state = outgoing_proto_state

        logits = self.lm_head(x)

        if not return_dict:
            return logits, x, all_masked, all_comp, all_protos, all_spl_in, all_masked_routing_logits, all_routing_logits, presents

        return {
            "logits": logits,
            "hidden_states": x,
            "masked_outputs": all_masked,
            "computation_outputs": all_comp,
            "proto_states": all_protos,
            "spl_inputs": all_spl_in,
            "masked_routing_logits": all_masked_routing_logits,
            "routing_logits": all_routing_logits,
            "past_key_values": presents,
        }
