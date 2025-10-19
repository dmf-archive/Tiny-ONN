import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .circle_rope_arc import (
    apply_arc_rope,
)
from .config import ModelConfig



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
            max_abs_val = torch.max(torch.abs(costs_qkv[i]), dim=-1, keepdim=True).values
            cost_score = costs_qkv[i] / (max_abs_val + 1e-9)
            routing_logits = match_qkv[i] - cost_score
            computation_output = comp_qkv[i]

            mask_active = (routing_logits > 0).float()
            masked_routing_logits = F.relu(routing_logits)

            masked_output = computation_output * mask_active

            if i == 0:
                q = masked_output
            elif i == 1:
                k = masked_output
            else:
                v = masked_output
            all_masked.append(masked_output)
            all_comp.append(computation_output)
            all_masked_routing_logits.append(masked_routing_logits)
            all_routing_logits.append(routing_logits)

        batch_size, seq_len, _ = q.shape
        num_heads = 1
        head_dim = self.d_model

        if coords_2d is not None and position_ids is not None:
            q, k = apply_arc_rope(q, k, coords_2d, position_ids, self.d_model)

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
        max_abs_val_o = torch.max(torch.abs(pc_o), dim=-1, keepdim=True).values
        cost_score_o = pc_o / (max_abs_val_o + 1e-9)
        routing_logits_o = mv_o - cost_score_o

        computation_output_o = c_o

        mask_active_o = (routing_logits_o > 0).float()
        masked_routing_logits_o = F.relu(routing_logits_o)

        masked_output_o = computation_output_o * mask_active_o

        x_out = x + masked_output_o
 
        all_masked.append(masked_output_o)
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
        captured_spl_inputs: list | None = None,
        captured_masked_grad_outputs: list | None = None,
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
