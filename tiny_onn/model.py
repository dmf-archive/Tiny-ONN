from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3MLP,
)

from .config import TinyOnnConfig


class HierarchicalMoE(nn.Module):
    def __init__(self, config: TinyOnnConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        expert_config = deepcopy(config)
        expert_config.intermediate_size = config.moe_intermediate_size
        self.experts = nn.ModuleList(
            [Qwen3MLP(expert_config) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)

        router_logits = self.gate(hidden_states_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros_like(hidden_states_flat)
        flat_selected_experts = selected_experts.flatten()
        flat_token_indices = (
            torch.arange(
                0, hidden_states_flat.shape[0], device=hidden_states.device
            ).repeat_interleave(self.top_k)
        )

        for expert_idx in range(self.num_experts):
            mask = flat_selected_experts == expert_idx
            if not torch.any(mask):
                continue

            token_indices_for_expert = flat_token_indices[mask]
            routing_weights_for_expert = routing_weights.view(-1)[mask]

            expert_output = self.experts[expert_idx](
                hidden_states_flat[token_indices_for_expert]
            )
            weighted_output = (
                expert_output * routing_weights_for_expert.unsqueeze(-1)
            )
            final_hidden_states.index_add_(
                0, token_indices_for_expert, weighted_output
            )

        return final_hidden_states.view(
            batch_size, sequence_length, hidden_size
        )


class TinyOnnDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: TinyOnnConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = HierarchicalMoE(config)


class TinyOnnForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config: TinyOnnConfig):
        super().__init__(config)
