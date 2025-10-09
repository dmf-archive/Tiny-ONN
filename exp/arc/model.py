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
    return scaled_logits


@torch.jit.script
def mas_normalize_negative(logits: torch.Tensor) -> torch.Tensor:
    return -F.relu(-mas_normalize(logits))


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
    computation_output = F.linear(F.silu(x), mu_weight, mu_bias)

    return computation_output, match_values, gate_logit



@torch.jit.script
def get_masked_routing_logits(routing_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(routing_logits)
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
        self.routing_gain = config.routing_gain
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
        pos_emb: tuple,
        past_kv: tuple | None = None,
        incoming_proto_state: dict | None = None,
        captured_spl_inputs: list | None = None,
        captured_masked_grad_outputs: list | None = None,
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
        c_q, mv_q, pc_q = self.attn.spl_q(ln1_out, outgoing_proto_state["attn_q"])
        c_k, mv_k, pc_k = self.attn.spl_k(ln1_out, outgoing_proto_state["attn_k"])
        c_v, mv_v, pc_v = self.attn.spl_v(ln1_out, outgoing_proto_state["attn_v"])

        all_masked, all_comp, all_masked_routing_logits, all_routing_logits, all_spl_inputs = [], [], [], [], []

        comp_qkv, match_qkv, costs_qkv = [c_q, c_k, c_v], [mv_q, mv_k, mv_v], [pc_q, pc_k, pc_v]
        q, k, v = torch.zeros_like(c_q), torch.zeros_like(c_k), torch.zeros_like(c_v)

        for i in range(3):
            cost_score = mas_normalize(costs_qkv[i])
            routing_logits = (match_qkv[i] - cost_score) * self.routing_gain
            computation_output = comp_qkv[i]

            masked_routing_logits = get_masked_routing_logits(routing_logits)
            masked = computation_output * masked_routing_logits
            if i == 0:
                q = masked
            elif i == 1:
                k = masked
            else:
                v = masked
            all_masked.append(masked)
            all_comp.append(computation_output)
            all_masked_routing_logits.append(masked_routing_logits)
            all_routing_logits.append(routing_logits)

        cos, sin = pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        batch_size, seq_len, _ = q.shape
        num_heads = 1
        head_dim = self.d_model

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

        c_o, mv_o, pc_o = self.attn.spl_o(attn_out, outgoing_proto_state["attn_o"])
        cost_score_o = mas_normalize(pc_o)
        routing_logits_o = (mv_o - cost_score_o) * self.routing_gain

        computation_output_o = c_o

        masked_routing_logits_o = get_masked_routing_logits(routing_logits_o)
        m_o = computation_output_o * masked_routing_logits_o
        x_out = x + m_o

        all_masked.append(m_o)
        all_comp.append(computation_output_o)
        all_masked_routing_logits.append(masked_routing_logits_o)
        all_routing_logits.append(routing_logits_o)
        all_spl_inputs.extend([ln1_out] * 3 + [attn_out])

        if self.training:
            if captured_spl_inputs is not None:
                captured_spl_inputs.append(ln1_out.clone().detach())
                captured_spl_inputs.append(ln1_out.clone().detach())
                captured_spl_inputs.append(ln1_out.clone().detach())
                captured_spl_inputs.append(attn_out.clone().detach())

            if captured_masked_grad_outputs is not None:
                all_masked[0].register_hook(lambda grad: captured_masked_grad_outputs.insert(0, grad.clone().detach()))
                all_masked[1].register_hook(lambda grad: captured_masked_grad_outputs.insert(0, grad.clone().detach()))
                all_masked[2].register_hook(lambda grad: captured_masked_grad_outputs.insert(0, grad.clone().detach()))
                all_masked[3].register_hook(lambda grad: captured_masked_grad_outputs.insert(0, grad.clone().detach()))

        return x_out, all_masked, all_comp, all_masked_routing_logits, all_spl_inputs, all_routing_logits, present_kv, outgoing_proto_state


class ArcEmbedding(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.color_embedding = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)

    def forward(self, input_ids: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        return self.color_embedding(input_ids)

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

    def forward(
        self,
        input_ids: torch.Tensor,
        coords: torch.Tensor | None = None,
        past_key_values: list | None = None,
        return_dict: bool = False,
        captured_spl_inputs: list | None = None,
        captured_masked_grad_outputs: list | None = None,
    ):
        x = self.embedding(input_ids)
        pos_emb = self.rotary_emb(x, seq_len=input_ids.size(1))
        past_key_values = past_key_values if past_key_values is not None else [None] * len(self.blocks)

        all_masked, all_comp, all_spl_in, all_masked_routing_logits, all_protos, all_routing_logits, presents = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
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
                pos_emb,
                past_key_values[i],
                incoming_proto_state,
                captured_spl_inputs=captured_spl_inputs,
                captured_masked_grad_outputs=captured_masked_grad_outputs,
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
