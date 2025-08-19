import math
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import TinyOnnConfig

ExpertID = tuple[str, int, int]

@dataclass
class CausalLMOutputWithAux(CausalLMOutputWithPast):
    aux_outputs: dict[ExpertID, Any] | None = None

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, scores: torch.Tensor) -> torch.Tensor:
        return (scores > 0).to(scores.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output

@torch.jit.script
def _gating_logic(hidden_states: torch.Tensor, sim_matrix: torch.Tensor, gates: torch.Tensor, max_experts: int, min_experts: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b, t, c = hidden_states.shape
    flat_hidden_states = hidden_states.view(b * t, c)
    logits = torch.matmul(F.normalize(flat_hidden_states, dim=-1), F.normalize(sim_matrix, dim=0)) - torch.sigmoid(gates)
    gated_logits = F.relu(logits)
    activation_mask = STEFunction.apply(gated_logits)
    inactive_mask = torch.sum(activation_mask, dim=1) == 0
    if inactive_mask.any():
        inactive_logits = logits[inactive_mask]
        fallback_indices = torch.topk(inactive_logits, min_experts, dim=-1).indices
        inactive_b_indices = torch.where(inactive_mask)[0]
        activation_mask.index_put_((inactive_b_indices.unsqueeze(1).expand(-1, min_experts), fallback_indices), torch.tensor(1.0, device=hidden_states.device, dtype=activation_mask.dtype))
    gated_logits_masked = torch.where(activation_mask > 0, gated_logits, torch.tensor(-1e9, dtype=gated_logits.dtype, device=gated_logits.device))
    return F.softmax(gated_logits_masked, dim=-1), logits, activation_mask, gated_logits

class GatingNetwork(nn.Module):
    def __init__(self, config: TinyOnnConfig, max_experts: int, min_experts: int):
        super().__init__()
        self.config = config
        self.max_experts = max_experts
        self.min_experts = min_experts
        self.sim_matrix = nn.Parameter(torch.randn(config.hidden_size, max_experts))
        self.gates = nn.Parameter(torch.zeros(max_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        active_expert_probs, logits, activation_mask, gated_logits = _gating_logic(hidden_states, self.sim_matrix, self.gates, self.max_experts, self.min_experts)
        return active_expert_probs, {"logits": logits, "activation_mask": activation_mask, "gated_logits": gated_logits}


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000, device: Any = None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len: int, device: Any, dtype: Any) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: Any, seq_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len is not None and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: Any) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DynSMHALayer(nn.Module):
    def __init__(self, config: TinyOnnConfig):
        super().__init__()
        self.config = config
        self.max_experts = config.max_attention_experts
        self.gating_network = GatingNetwork(config, self.max_experts, config.min_attention_experts)
        self.q_proj = nn.Parameter(torch.empty(self.max_experts, config.hidden_size, config.head_dim))
        self.k_proj = nn.Parameter(torch.empty(self.max_experts, config.hidden_size, config.head_dim))
        self.v_proj = nn.Parameter(torch.empty(self.max_experts, config.hidden_size, config.head_dim))
        self.o_proj = nn.Parameter(torch.empty(self.max_experts, config.head_dim, config.hidden_size))
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.max_position_embeddings, base=config.rope_theta)
        for i in range(self.max_experts):
            nn.init.xavier_uniform_(self.q_proj[i])
            nn.init.xavier_uniform_(self.k_proj[i])
            nn.init.xavier_uniform_(self.v_proj[i])
            nn.init.xavier_uniform_(self.o_proj[i])

    def forward_gating(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.gating_network(hidden_states)

    def forward_main(self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, gate_cache: dict[str, torch.Tensor], position_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        B, T, C = hidden_states.shape
        flat_hidden_states = hidden_states.view(B * T, C)
        activation_mask = gate_cache["activation_mask"]

        token_indices, expert_indices = torch.where(activation_mask > 0)

        if token_indices.numel() == 0:
            final_output = torch.zeros_like(hidden_states)
        else:
            selected_hidden_states = flat_hidden_states[token_indices]
            weights_for_active = routing_weights[token_indices, expert_indices].unsqueeze(-1)

            q_parts = torch.bmm(selected_hidden_states.unsqueeze(1), self.q_proj[expert_indices]).squeeze(1)
            k_parts = torch.bmm(selected_hidden_states.unsqueeze(1), self.k_proj[expert_indices]).squeeze(1)
            v_parts = torch.bmm(selected_hidden_states.unsqueeze(1), self.v_proj[expert_indices]).squeeze(1)

            weighted_q_parts = q_parts * weights_for_active
            weighted_k_parts = k_parts * weights_for_active
            weighted_v_parts = v_parts * weights_for_active

            q_flat = torch.zeros(B * T, self.config.head_dim, device=hidden_states.device, dtype=hidden_states.dtype)
            k_flat = torch.zeros(B * T, self.config.head_dim, device=hidden_states.device, dtype=hidden_states.dtype)
            v_flat = torch.zeros(B * T, self.config.head_dim, device=hidden_states.device, dtype=hidden_states.dtype)

            q_flat.scatter_add_(0, token_indices.unsqueeze(-1).expand_as(weighted_q_parts), weighted_q_parts)
            k_flat.scatter_add_(0, token_indices.unsqueeze(-1).expand_as(weighted_k_parts), weighted_k_parts)
            v_flat.scatter_add_(0, token_indices.unsqueeze(-1).expand_as(weighted_v_parts), weighted_v_parts)

            q = q_flat.view(B, T, -1)
            k = k_flat.view(B, T, -1)
            v = v_flat.view(B, T, -1)
            
            cos, sin = self.rotary_emb(v, seq_len=T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

            attn_output = F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), is_causal=True).squeeze(1)

            attn_output_flat = attn_output.view(B * T, -1)
            selected_attn_output = attn_output_flat[token_indices]

            o_parts = torch.bmm(selected_attn_output.unsqueeze(1), self.o_proj[expert_indices]).squeeze(1)
            weighted_o_parts = o_parts * weights_for_active

            final_output_flat = torch.zeros_like(flat_hidden_states)
            final_output_flat.scatter_add_(0, token_indices.unsqueeze(-1).expand_as(weighted_o_parts), weighted_o_parts)
            final_output = final_output_flat.view(B, T, C)

        cache = {
            "gate_cache": gate_cache,
            "routing_weights": routing_weights,
            "final_output": final_output,
            "num_active_tokens": torch.sum(activation_mask).item(),
            "B": B, "T": T,
        }
        return final_output, cache

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        routing_weights, gate_cache = self.forward_gating(hidden_states)
        return self.forward_main(hidden_states, routing_weights, gate_cache, position_ids)


class DynamicMoELayer(nn.Module):
    def __init__(self, config: TinyOnnConfig):
        super().__init__()
        self.config = config
        self.max_experts = config.max_moe_experts
        self.gating_network = GatingNetwork(config, self.max_experts, config.min_moe_experts)
        self.w1 = nn.Parameter(torch.empty(self.max_experts, config.hidden_size, config.intermediate_size))
        self.w2 = nn.Parameter(torch.empty(self.max_experts, config.intermediate_size, config.hidden_size))
        for i in range(self.max_experts):
            nn.init.kaiming_uniform_(self.w1[i], a=5**0.5)
            nn.init.kaiming_uniform_(self.w2[i], a=5**0.5)

    def forward_gating(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.gating_network(hidden_states)

    def forward_main(self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, gate_cache: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        B, T, C = hidden_states.shape
        flat_hidden_states = hidden_states.view(B * T, C)
        activation_mask = gate_cache["activation_mask"]

        token_indices, expert_indices = torch.where(activation_mask > 0)

        if token_indices.numel() == 0:
            final_output = torch.zeros_like(hidden_states)
        else:
            selected_hidden_states = flat_hidden_states[token_indices]

            selected_w1 = self.w1[expert_indices]
            selected_w2 = self.w2[expert_indices]

            intermediate_activated = F.gelu(torch.bmm(selected_hidden_states.unsqueeze(1), selected_w1))
            output_flat = torch.bmm(intermediate_activated, selected_w2).squeeze(1)

            weights_for_active = routing_weights[token_indices, expert_indices]
            weighted_output = output_flat * weights_for_active.unsqueeze(-1)

            final_output_flat = torch.zeros_like(flat_hidden_states)
            final_output_flat.scatter_add_(0, token_indices.unsqueeze(-1).expand_as(weighted_output), weighted_output)
            final_output = final_output_flat.view(B, T, C)

        cache = {
            "gate_cache": gate_cache,
            "routing_weights": routing_weights,
            "final_output": final_output,
            "num_active_tokens": torch.sum(activation_mask).item(),
            "B": B, "T": T,
        }
        return final_output, cache

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        routing_weights, gate_cache = self.forward_gating(hidden_states)
        return self.forward_main(hidden_states, routing_weights, gate_cache)


class Block(nn.Module):
    def __init__(self, config: TinyOnnConfig, layer_index: int):
        super().__init__()
        self.layer_index = layer_index
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.smha_layer = DynSMHALayer(config)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.moe_layer = DynamicMoELayer(config)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, dict[ExpertID, Any]]:
        residual = hidden_states
        normed_hs_smha = self.ln1(hidden_states)
        smha_routing_weights, smha_gate_cache = self.smha_layer.forward_gating(normed_hs_smha)

        def smha_checkpointed_fn(hs_norm: torch.Tensor, pos_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
            return self.smha_layer.forward_main(hs_norm, smha_routing_weights, smha_gate_cache, pos_ids)

        attn_output, smha_cache = checkpoint(smha_checkpointed_fn, normed_hs_smha, position_ids, use_reentrant=False)
        smha_cache["normed_hs"] = normed_hs_smha
        hidden_states = residual + attn_output

        residual = hidden_states
        normed_hs_moe = self.ln2(hidden_states)
        moe_routing_weights, moe_gate_cache = self.moe_layer.forward_gating(normed_hs_moe)

        def moe_checkpointed_fn(hs_norm: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
            return self.moe_layer.forward_main(hs_norm, moe_routing_weights, moe_gate_cache)

        moe_output, moe_cache = checkpoint(moe_checkpointed_fn, normed_hs_moe, use_reentrant=False)
        moe_cache["normed_hs"] = normed_hs_moe
        hidden_states = residual + moe_output

        block_cache = {
            ("smha", self.layer_index, 0): smha_cache,
            ("moe", self.layer_index, 0): moe_cache,
        }
        return hidden_states, block_cache


class TinyOnnPreTrainedModel(PreTrainedModel):
    config_class = TinyOnnConfig
    base_model_prefix = "model"


class TinyOnnModel(TinyOnnPreTrainedModel):
    def __init__(self, config: TinyOnnConfig):
        super().__init__(config)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.embedding_size)
        self.embedding_proj_in = nn.Linear(config.embedding_size, config.hidden_size, bias=config.bias)
        self.layers = nn.ModuleList([Block(config, i) for i in range(config.num_hidden_layers)])
        self.final_ln = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, dict[ExpertID, Any]]:
        hidden_states = self.tok_embeddings(input_ids)
        hidden_states = self.embedding_proj_in(hidden_states)

        B, T, C = hidden_states.shape
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)

        flat_forward_cache: dict[ExpertID, Any] = {}
        for layer in self.layers:
            hidden_states, block_cache = layer(hidden_states, position_ids)
            flat_forward_cache.update(block_cache)
        hidden_states = self.final_ln(hidden_states)
        return hidden_states, flat_forward_cache

class TinyOnnForCausalLM(TinyOnnPreTrainedModel):
    def __init__(self, config: TinyOnnConfig):
        super().__init__(config)
        self.model = TinyOnnModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.tok_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Module) -> None:
        self.model.tok_embeddings = new_embeddings
    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> CausalLMOutputWithAux:
        hidden_states, flat_forward_cache = self.model(input_ids, **kwargs)
        final_logits = self.lm_head(hidden_states)
        return CausalLMOutputWithAux(logits=final_logits, aux_outputs=flat_forward_cache)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 30, eos_token_id: int | None = None) -> torch.Tensor:
        generated_ids = input_ids
        for _ in range(max_new_tokens):
            outputs = self.forward(input_ids=generated_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        return generated_ids
