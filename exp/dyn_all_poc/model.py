from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import DTYPE, UnifiedConfig

ExpertID = tuple[int, int, int]

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, scores: torch.Tensor) -> torch.Tensor:
        return (scores > 0).to(scores.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output

@torch.jit.script
def _gating_logic(hidden_states: torch.Tensor, sim_matrix: torch.Tensor, gates: torch.Tensor, max_experts: int, min_experts: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return F.softmax(gated_logits_masked, dim=-1), logits, activation_mask, gated_logits, inactive_mask

class GatingNetwork(nn.Module):
    def __init__(self, config: UnifiedConfig, max_experts: int, min_experts: int):
        super().__init__()
        self.config = config
        self.max_experts = max_experts
        self.min_experts = min_experts
        self.sim_matrix = nn.Parameter(torch.randn(config.hidden_size, max_experts, dtype=DTYPE))
        self.gates = nn.Parameter(torch.zeros(max_experts, dtype=DTYPE))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        active_expert_probs, logits, activation_mask, gated_logits, inactive_mask = _gating_logic(hidden_states, self.sim_matrix, self.gates, self.max_experts, self.min_experts)
        return active_expert_probs, {"logits": logits, "activation_mask": activation_mask, "gated_logits": gated_logits, "inactive_mask": inactive_mask}


@torch.jit.script
def sparse_weighted_sum(routing_weights: torch.Tensor, activation_mask: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
    num_tokens, num_experts = routing_weights.shape

    if expert_weights.dim() == 3:
        d1, d2 = expert_weights.shape[1], expert_weights.shape[2]
        shape_list = [num_tokens, d1, d2]

        token_indices, expert_indices = torch.where(activation_mask > 0)

        if token_indices.numel() == 0:
            return torch.zeros(shape_list, device=expert_weights.device, dtype=expert_weights.dtype)

        weights_for_active = routing_weights[token_indices, expert_indices]
        selected_expert_weights = expert_weights[expert_indices]

        weighted_expert_params = weights_for_active.view(-1, 1, 1) * selected_expert_weights

        combined_weights = torch.zeros(shape_list, device=expert_weights.device, dtype=expert_weights.dtype)
        combined_weights.index_add_(0, token_indices, weighted_expert_params)
        return combined_weights
    else:
        raise ValueError("Unsupported expert_weights dimension. Supported dimensions: 3.")


class DynSMHALayer(nn.Module):
    def __init__(self, config: UnifiedConfig, layer_index: int):
        super().__init__()
        self.config = config
        self.max_experts = config.max_attention_experts
        self.layer_index = layer_index
        self.gating_network = GatingNetwork(config, self.max_experts, config.min_attention_experts)
        self.q_proj = nn.Parameter(torch.empty(self.max_experts, config.hidden_size, config.head_dim, dtype=DTYPE))
        self.k_proj = nn.Parameter(torch.empty(self.max_experts, config.hidden_size, config.head_dim, dtype=DTYPE))
        self.v_proj = nn.Parameter(torch.empty(self.max_experts, config.hidden_size, config.head_dim, dtype=DTYPE))
        self.o_proj = nn.Parameter(torch.empty(self.max_experts, config.head_dim, config.hidden_size, dtype=DTYPE))
        for i in range(self.max_experts):
            nn.init.xavier_uniform_(self.q_proj[i])
            nn.init.xavier_uniform_(self.k_proj[i])
            nn.init.xavier_uniform_(self.v_proj[i])
            nn.init.xavier_uniform_(self.o_proj[i])

    def forward_gating(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.gating_network(hidden_states)

    def forward_main(self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, gate_cache: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        B, T, C = hidden_states.shape
        flat_hidden_states = hidden_states.view(B * T, C)
        activation_mask = gate_cache["activation_mask"]

        q_projs = sparse_weighted_sum(routing_weights, activation_mask, self.q_proj)
        k_projs = sparse_weighted_sum(routing_weights, activation_mask, self.k_proj)
        v_projs = sparse_weighted_sum(routing_weights, activation_mask, self.v_proj)

        q = torch.einsum('ac,ach->ah', flat_hidden_states, q_projs).view(B, T, -1)
        k = torch.einsum('ac,ach->ah', flat_hidden_states, k_projs).view(B, T, -1)
        v = torch.einsum('ac,ach->ah', flat_hidden_states, v_projs).view(B, T, -1)

        attn_output = F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), is_causal=True).squeeze(1)

        o_projs = sparse_weighted_sum(routing_weights, activation_mask, self.o_proj)
        final_output_flat = torch.einsum('ah,ahc->ac', attn_output.view(B * T, -1), o_projs)
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


class DynamicMoELayer(nn.Module):
    def __init__(self, config: UnifiedConfig, layer_index: int):
        super().__init__()
        self.config = config
        self.max_experts = config.max_moe_experts
        self.layer_index = layer_index
        self.gating_network = GatingNetwork(config, self.max_experts, config.min_moe_experts)
        self.w1 = nn.Parameter(torch.empty(self.max_experts, config.hidden_size, config.moe_intermediate_size, dtype=DTYPE))
        self.w2 = nn.Parameter(torch.empty(self.max_experts, config.moe_intermediate_size, config.hidden_size, dtype=DTYPE))
        for i in range(self.max_experts):
            nn.init.kaiming_uniform_(self.w1[i], a=5**0.5)
            nn.init.kaiming_uniform_(self.w2[i], a=5**0.5)

    def forward_gating(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.gating_network(hidden_states)

    def forward_main(self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, gate_cache: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        B, T, C = hidden_states.shape
        flat_hidden_states = hidden_states.view(B * T, C)
        activation_mask = gate_cache["activation_mask"]

        w1_projs = sparse_weighted_sum(routing_weights, activation_mask, self.w1)
        w2_projs = sparse_weighted_sum(routing_weights, activation_mask, self.w2)

        intermediate_activated = F.gelu(torch.einsum('ac,aci->ai', flat_hidden_states, w1_projs))
        final_output_flat = torch.einsum('ai,aic->ac', intermediate_activated, w2_projs)
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
    def __init__(self, config: UnifiedConfig, layer_index: int):
        super().__init__()
        self.layer_index = layer_index
        self.ln1 = nn.LayerNorm(config.hidden_size, dtype=DTYPE)
        self.smha_layer = DynSMHALayer(config, layer_index)
        self.ln2 = nn.LayerNorm(config.hidden_size, dtype=DTYPE)
        self.moe_layer = DynamicMoELayer(config, layer_index)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[ExpertID, Any]]:
        # --- Gating Phase (non-checkpointed) ---
        residual = hidden_states
        normed_hs_smha = self.ln1(hidden_states)
        smha_routing_weights, smha_gate_cache = self.smha_layer.forward_gating(normed_hs_smha)

        # --- Execution Phase (checkpointed) ---
        def smha_checkpointed_fn(hs_norm: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
            return self.smha_layer.forward_main(hs_norm, smha_routing_weights, smha_gate_cache)

        attn_output, smha_cache = checkpoint(smha_checkpointed_fn, normed_hs_smha, use_reentrant=False)
        smha_cache["normed_hs"] = normed_hs_smha
        hidden_states = residual + attn_output

        # --- Gating Phase (non-checkpointed) ---
        residual = hidden_states
        normed_hs_moe = self.ln2(hidden_states)
        moe_routing_weights, moe_gate_cache = self.moe_layer.forward_gating(normed_hs_moe)

        # --- Execution Phase (checkpointed) ---
        def moe_checkpointed_fn(hs_norm: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
            return self.moe_layer.forward_main(hs_norm, moe_routing_weights, moe_gate_cache)

        moe_output, moe_cache = checkpoint(moe_checkpointed_fn, normed_hs_moe, use_reentrant=False)
        moe_cache["normed_hs"] = normed_hs_moe
        hidden_states = residual + moe_output

        block_cache = {
            (1, 0, self.layer_index): smha_cache,
            (2, 0, self.layer_index): moe_cache,
        }
        return hidden_states, block_cache


class TinyOnnModel(nn.Module):
    def __init__(self, config: UnifiedConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, dtype=DTYPE)
        self.layers = nn.ModuleList([Block(config, i) for i in range(config.num_hidden_layers)])
        self.final_ln = nn.LayerNorm(config.hidden_size, dtype=DTYPE)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=DTYPE)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict[ExpertID, Any]]:
        hidden_states = self.embedding(input_ids)

        flat_forward_cache: dict[ExpertID, Any] = {}
        for layer in self.layers:
            hidden_states, block_cache = layer(hidden_states)
            flat_forward_cache.update(block_cache)
        hidden_states = self.final_ln(hidden_states)
        final_logits = self.lm_head(hidden_states)
        return final_logits, flat_forward_cache
