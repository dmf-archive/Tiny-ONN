import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from .config import TinyOnnArcConfig

ExpertID = tuple[str, int, int]


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, scores: torch.Tensor) -> torch.Tensor:
        return (scores > 0).to(scores.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


@torch.jit.script
def _gating_logic(
    hidden_states: torch.Tensor, sim_matrix: torch.Tensor, gates: torch.Tensor, max_experts: int, min_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b, t, c = hidden_states.shape
    flat_hidden_states = hidden_states.view(b * t, c)
    logits = torch.matmul(F.normalize(flat_hidden_states, dim=-1), F.normalize(sim_matrix, dim=0)) - torch.sigmoid(
        gates
    )
    gated_logits = F.relu(logits)
    activation_mask = STEFunction.apply(gated_logits)
    inactive_mask = torch.sum(activation_mask, dim=1) == 0
    if inactive_mask.any():
        inactive_logits = logits[inactive_mask]
        fallback_indices = torch.topk(inactive_logits, min_experts, dim=-1).indices
        inactive_b_indices = torch.where(inactive_mask)[0]
        activation_mask.index_put_(
            (inactive_b_indices.unsqueeze(1).expand(-1, min_experts), fallback_indices),
            torch.tensor(1.0, device=hidden_states.device, dtype=activation_mask.dtype),
        )
    gated_logits_masked = torch.where(
        activation_mask > 0,
        gated_logits,
        torch.tensor(-torch.inf, dtype=gated_logits.dtype, device=gated_logits.device),
    )
    return F.softmax(gated_logits_masked, dim=-1), logits, activation_mask, gated_logits


class GatingNetwork(nn.Module):
    def __init__(self, config: TinyOnnArcConfig, max_experts: int, min_experts: int):
        super().__init__()
        self.config = config
        self.max_experts = max_experts
        self.min_experts = min_experts
        self.sim_matrix = nn.Parameter(torch.randn(config.hidden_size, max_experts))
        self.gates = nn.Parameter(torch.zeros(max_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        active_expert_probs, logits, activation_mask, gated_logits = _gating_logic(
            hidden_states, self.sim_matrix, self.gates, self.max_experts, self.min_experts
        )
        return active_expert_probs, {"logits": logits, "activation_mask": activation_mask, "gated_logits": gated_logits}


class DynSMHALayer(nn.Module):
    def __init__(self, config: TinyOnnArcConfig, is_causal: bool = False):
        super().__init__()
        self.config = config
        self.max_experts = config.max_attention_experts
        self.is_causal = is_causal
        self.gating_network = GatingNetwork(config, self.max_experts, config.min_attention_experts)
        self.q_proj = nn.Parameter(torch.empty(self.max_experts, config.hidden_size, config.head_dim))
        self.k_proj = nn.Parameter(torch.empty(self.max_experts, config.hidden_size, config.head_dim))
        self.v_proj = nn.Parameter(torch.empty(self.max_experts, config.hidden_size, config.head_dim))
        self.o_proj = nn.Parameter(torch.empty(self.max_experts, config.head_dim, config.hidden_size))

        for i in range(self.max_experts):
            nn.init.xavier_uniform_(self.q_proj[i])
            nn.init.xavier_uniform_(self.k_proj[i])
            nn.init.xavier_uniform_(self.v_proj[i])
            nn.init.xavier_uniform_(self.o_proj[i])

    def forward_gating(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.gating_network(hidden_states)

    def forward_main(self, hidden_states: torch.Tensor, routing_weights: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        B, T, C = hidden_states.shape
        routing_weights_reshaped = rearrange(routing_weights, "(b t) e -> b t e", b=B)

        q_experts = torch.einsum("btc,ech->bteh", hidden_states, self.q_proj)
        k_experts = torch.einsum("btc,ech->bteh", hidden_states, self.k_proj)
        v_experts = torch.einsum("btc,ech->bteh", hidden_states, self.v_proj)

        q_agg = torch.einsum("bteh,bte->bth", q_experts, routing_weights_reshaped)
        k_agg = torch.einsum("bteh,bte->bth", k_experts, routing_weights_reshaped)
        v_agg = torch.einsum("bteh,bte->bth", v_experts, routing_weights_reshaped)

        q = rearrange(q_agg, "b t h -> b 1 t h")
        k = rearrange(k_agg, "b t h -> b 1 t h")
        v = rearrange(v_agg, "b t h -> b 1 t h")

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        attn_output = rearrange(attn_output, "b 1 t h -> b t h")

        output_experts = torch.einsum("bth,ehc->btec", attn_output, self.o_proj)
        final_output = torch.einsum("btec,bte->btc", output_experts, routing_weights_reshaped)

        cache = {"final_output": final_output, "routing_weights": routing_weights, "B": B, "T": T}
        return final_output, cache


class DynamicMoELayer(nn.Module):
    def __init__(self, config: TinyOnnArcConfig):
        super().__init__()
        self.config = config
        self.max_experts = config.max_moe_experts
        self.gating_network = GatingNetwork(config, self.max_experts, config.min_moe_experts)
        self.w1 = nn.Parameter(torch.empty(self.max_experts, config.hidden_size, config.intermediate_size))
        self.w2 = nn.Parameter(torch.empty(self.max_experts, config.intermediate_size, config.hidden_size))
        for i in range(self.max_experts):
            nn.init.kaiming_uniform_(self.w1[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w2[i], a=math.sqrt(5))

    def forward_gating(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.gating_network(hidden_states)

    def forward_main(self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, gate_cache: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        B, T, C = hidden_states.shape
        routing_weights_reshaped = rearrange(routing_weights, "(b t) e -> b t e", b=B)

        intermediate_experts = F.gelu(torch.einsum("btc,eci->btei", hidden_states, self.w1))
        output_experts = torch.einsum("btei,eic->btec", intermediate_experts, self.w2)

        final_output = torch.einsum("btec,bte->btc", output_experts, routing_weights_reshaped)

        cache = {"final_output": final_output, "gate_cache": gate_cache, "routing_weights": routing_weights, "B": B, "T": T, "normed_hs": hidden_states, "layer": self}
        return final_output, cache


class Block(nn.Module):
    def __init__(self, config: TinyOnnArcConfig, layer_index: int):
        super().__init__()
        self.layer_index = layer_index
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.smha_layer = DynSMHALayer(config, is_causal=True)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.moe_layer = DynamicMoELayer(config)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[ExpertID, Any]]:
        residual = hidden_states
        normed_hs_smha = self.ln1(hidden_states)
        smha_routing_weights, smha_gate_cache = self.smha_layer.forward_gating(normed_hs_smha)
        B, T, C = hidden_states.shape
        smha_routing_weights_flat = smha_routing_weights.view(B * T, -1)

        def smha_checkpointed_fn(hs_norm: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
            return self.smha_layer.forward_main(hs_norm, smha_routing_weights_flat)

        attn_output, smha_cache = checkpoint(smha_checkpointed_fn, normed_hs_smha, use_reentrant=False)
        smha_cache["gate_cache"] = smha_gate_cache
        smha_cache["normed_hs"] = normed_hs_smha
        smha_cache["layer"] = self.smha_layer
        hidden_states = residual + attn_output

        residual = hidden_states
        normed_hs_moe = self.ln2(hidden_states)
        moe_routing_weights, moe_gate_cache = self.moe_layer.forward_gating(normed_hs_moe)
        moe_routing_weights_flat = moe_routing_weights.view(B * T, -1)

        def moe_checkpointed_fn(hs_norm: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
            return self.moe_layer.forward_main(hs_norm, moe_routing_weights_flat, moe_gate_cache)

        moe_output, moe_cache = checkpoint(moe_checkpointed_fn, normed_hs_moe, use_reentrant=False)
        moe_cache["normed_hs"] = normed_hs_moe
        hidden_states = residual + moe_output

        block_cache = {("smha", self.layer_index, 0): smha_cache, ("moe", self.layer_index, 0): moe_cache}
        return hidden_states, block_cache


class AutoregressiveEmbedding(nn.Module):
    def __init__(self, config: TinyOnnArcConfig):
        super().__init__()
        self.tok_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        pos_ids = self.position_ids[:seq_len]
        
        tok_embeds = self.tok_embed(input_ids)
        pos_embeds = self.pos_embed(pos_ids)
        
        return tok_embeds + pos_embeds


class TinyOnnModel(nn.Module):
    def __init__(self, config: TinyOnnArcConfig):
        super().__init__()
        self.config = config
        self.embeddings = AutoregressiveEmbedding(config)
        self.layers = nn.ModuleList([Block(config, i) for i in range(config.num_hidden_layers)])
        self.final_ln = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, dict[ExpertID, Any]]:
        hidden_states = self.embeddings(input_ids)

        flat_forward_cache: dict[ExpertID, Any] = {}
        for layer in self.layers:
            hidden_states, block_cache = layer(hidden_states)
            flat_forward_cache.update(block_cache)
            
        hidden_states = self.final_ln(hidden_states)
        return hidden_states, flat_forward_cache


class TinyOnnForArcReconstruction(nn.Module):
    def __init__(self, config: TinyOnnArcConfig):
        super().__init__()
        self.model = TinyOnnModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, dict[ExpertID, Any]]:
        hidden_states, flat_forward_cache = self.model(input_ids=input_ids, **kwargs)
        final_logits = self.lm_head(hidden_states)
        return final_logits, flat_forward_cache
