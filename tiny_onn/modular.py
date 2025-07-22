import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple

from transformers.activations import ACT2FN
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeDecoderLayer

from .config import TinyOnnConfig

class TinyOnnExpert(nn.Module):
    def __init__(self, config: TinyOnnConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.w2 = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states

class TinyOnnMoE(nn.Module):
    def __init__(self, config: TinyOnnConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts_per_layer
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([TinyOnnExpert(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        routing_weights, selected_experts = torch.topk(routing_weights, k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros_like(hidden_states)

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        return final_hidden_states.view(batch_size, sequence_length, hidden_dim), router_logits

class TinyOnnDecoderLayer(Qwen2MoeDecoderLayer):
    def __init__(self, config: TinyOnnConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = TinyOnnMoE(config)
