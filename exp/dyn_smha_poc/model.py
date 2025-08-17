from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .config import DTYPE, DynSMHAConfig


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        return (scores > 0).to(scores.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class SingleHeadAttention(nn.Module):
    def __init__(self, config: DynSMHAConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.head_dim, bias=config.bias, dtype=DTYPE)
        self.k_proj = nn.Linear(config.hidden_size, config.head_dim, bias=config.bias, dtype=DTYPE)
        self.v_proj = nn.Linear(config.hidden_size, config.head_dim, bias=config.bias, dtype=DTYPE)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        attn_output = F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)).squeeze(1)
        return attn_output, q, k

class GatingNetwork(nn.Module):
    def __init__(self, config: DynSMHAConfig):
        super().__init__()
        self.config = config
        self.sim_matrix = nn.Parameter(torch.randn(config.hidden_size, config.num_experts, dtype=DTYPE))
        self.gates = nn.Parameter(torch.zeros(config.num_experts, dtype=DTYPE))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        B, T, C = hidden_states.shape
        flat_hidden_states = hidden_states.view(-1, C)

        token_affinity = torch.matmul(flat_hidden_states, self.sim_matrix)
        logits = token_affinity - torch.sigmoid(self.gates)
        
        gated_logits = F.relu(logits)
        activation_mask = STEFunction.apply(gated_logits).view(B, T, -1)

        num_active = torch.sum(activation_mask, dim=2)
        inactive_mask = num_active == 0
        
        fallback_count = torch.tensor(0.0, device=activation_mask.device)
        if inactive_mask.any():
            fallback_count = inactive_mask.sum().float()
            k_fallback = self.config.num_experts // 2
            
            fallback_logits = logits.view(B, T, -1)[inactive_mask]
            fallback_indices = torch.topk(fallback_logits, k_fallback, dim=1).indices

            inactive_b_indices, inactive_t_indices = inactive_mask.nonzero(as_tuple=True)
            
            expanded_b = inactive_b_indices.unsqueeze(1).expand(-1, k_fallback)
            expanded_t = inactive_t_indices.unsqueeze(1).expand(-1, k_fallback)
            
            activation_mask[expanded_b, expanded_t, fallback_indices] = 1.0

        gated_logits_masked = torch.where(activation_mask > 0, gated_logits.view(B, T, -1), -torch.finfo(DTYPE).max)
        active_expert_probs = F.softmax(gated_logits_masked, dim=-1)

        cache = {"logits": logits.view(B, T, -1), "activation_mask": activation_mask}
        return active_expert_probs, fallback_count, cache


class DynSMHALayer(nn.Module):
    def __init__(self, config: DynSMHAConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([SingleHeadAttention(config) for _ in range(config.num_experts)])
        self.gating_network = GatingNetwork(config)
        self.o_weights = nn.Parameter(torch.randn(config.num_experts, config.head_dim, config.hidden_size, dtype=DTYPE))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        expert_outputs = torch.stack([expert(hidden_states)[0] for expert in self.experts], dim=2)
        
        routing_weights, fallback_count, gate_cache = self.gating_network(hidden_states)

        num_active = torch.sum(routing_weights > 0, dim=2).float().mean()

        combined_heads = torch.einsum('bteh,bte->bth', expert_outputs, routing_weights)
        dynamic_o_proj = torch.einsum('ehd,bte->bthd', self.o_weights, routing_weights)
        
        final_output = torch.einsum('bth,bthd->btd', combined_heads, dynamic_o_proj)

        forward_cache: Dict[str, Any] = {
            "gate_cache": gate_cache,
            "all_sha_outputs": expert_outputs,
            "num_active_experts": num_active,
        }
        return final_output, fallback_count, forward_cache

class TinyOnnModel(nn.Module):
    def __init__(self, config: DynSMHAConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, dtype=DTYPE)
        self.ln1 = nn.LayerNorm(config.hidden_size, dtype=DTYPE)
        self.smha_layer = DynSMHALayer(config)
        self.ln2 = nn.LayerNorm(config.hidden_size, dtype=DTYPE)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=DTYPE)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        hidden_states = self.embedding(input_ids)
        residual = hidden_states

        normalized_states = self.ln1(hidden_states)
        attn_output, fallback_count, forward_cache = self.smha_layer(normalized_states)

        hidden_states = residual + attn_output
        hidden_states = self.ln2(hidden_states)

        final_logits = self.lm_head(hidden_states)

        return final_logits, fallback_count, forward_cache
