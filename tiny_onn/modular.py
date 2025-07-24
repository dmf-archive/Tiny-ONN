from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3MLP,
    Qwen3Model,
)

from .config import TinyOnnConfig

if TYPE_CHECKING:
    pass


class TinyOnnExpert(nn.Module):
    def __init__(self, config: TinyOnnConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.w2 = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(
            hidden_states
        )
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class TinyOnnMoE(Qwen3MLP):
    def __init__(self, config: TinyOnnConfig):
        super().__init__(config)
        self.num_experts = config.num_experts_per_layer
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [TinyOnnExpert(config) for _ in range(self.num_experts)]
        )
        self.last_router_logits: torch.Tensor | None = None
        self.last_expert_token_indices: dict[int, torch.Tensor] | None = None
        self.last_expert_inputs: dict[int, torch.Tensor] | None = None
        self.last_selected_experts: torch.Tensor | None = None
        self.last_k_per_token: torch.Tensor | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        surprise_budget: float = 0.5,
        surprise_context: dict | None = None,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        from training.autograd import CaptureSurprise

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden_states = hidden_states.view(-1, hidden_dim)
        num_tokens = flat_hidden_states.shape[0]

        router_logits = self.gate(flat_hidden_states)

        if self.top_k == -1:
            selection_probs = torch.softmax(-router_logits, dim=-1, dtype=torch.float)
            sorted_probs, _ = torch.sort(selection_probs, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            k_per_token = torch.sum(cumulative_probs < surprise_budget, dim=-1) + 1
        else:
            k_per_token = torch.full((num_tokens,), self.top_k, dtype=torch.long, device=hidden_states.device)
        self.last_k_per_token = k_per_token

        max_k = k_per_token.max().item()
        max_k = min(max_k, self.num_experts)

        routing_weights, selected_experts = torch.topk(router_logits, max_k, dim=-1, largest=False)
        routing_weights = F.softmax(-routing_weights, dim=-1, dtype=torch.float)

        final_hidden_states = torch.zeros_like(flat_hidden_states)

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        self.last_router_logits = router_logits
        self.last_expert_token_indices = {}
        self.last_expert_inputs = {}

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.numel() == 0:
                continue

            self.last_expert_token_indices[expert_idx] = top_x

            current_state = flat_hidden_states[top_x].clone()
            self.last_expert_inputs[expert_idx] = current_state

            if self.training and surprise_context is not None:
                current_state = CaptureSurprise.apply(
                    current_state, layer_idx, expert_idx, top_x, surprise_context
                )

            current_hidden_states = self.experts[expert_idx](current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        self.last_selected_experts = selected_experts
        return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


class TinyOnnDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: TinyOnnConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = TinyOnnMoE(config)
        self.layer_idx = layer_idx

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_outputs = self.self_attn(hidden_states, *args, **kwargs)
        hidden_states = attn_outputs[0]

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        mlp_kwargs = {
            "surprise_budget": kwargs.get("surprise_budget"),
            "surprise_context": kwargs.get("surprise_context"),
            "layer_idx": self.layer_idx,
        }
        hidden_states = self.mlp(hidden_states, **mlp_kwargs)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,) + attn_outputs[1:]

        return outputs


class TinyOnnModel(Qwen3Model):
    def __init__(self, config: TinyOnnConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                TinyOnnDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.post_init()

    def forward(self, *args, **kwargs):
        # Allow passing kwargs down to the layers
        # This is the only way to pass kwargs down to the layers
        # in the `transformers` framework.
        return super().forward(*args, **kwargs)


class TinyOnnForCausalLM(Qwen3ForCausalLM):
    config_class = TinyOnnConfig  # type: ignore

    def __init__(self, config: TinyOnnConfig):
        super().__init__(config)
        self.model = TinyOnnModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, *args, **kwargs):
        # This is the only way to pass kwargs down to the model
        return super().forward(*args, **kwargs)
