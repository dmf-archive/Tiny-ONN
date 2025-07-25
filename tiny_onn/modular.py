from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
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


class TinyOnnGate(nn.Module):
    def __init__(self, config: TinyOnnConfig):
        super().__init__()
        self.sim_matrix = nn.Parameter(torch.randn(config.hidden_size, config.num_experts_per_layer))
        self.temperature = nn.Parameter(torch.log(torch.full([1], 1.0 / 0.1)), requires_grad=False)
        self.gates = nn.Parameter(torch.zeros(config.num_experts_per_layer))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        norm_hidden_states = F.normalize(hidden_states, dim=-1)
        norm_sim_matrix = F.normalize(self.sim_matrix, dim=0)
        logits = torch.matmul(norm_hidden_states, norm_sim_matrix)
        logits = logits * self.temperature.exp()
        return logits


def _create_full_hook(
    cache: list[dict[str, Any]],
    routing_mask: torch.Tensor,
    layer_idx: int,
) -> Callable:
    def hook(grad: torch.Tensor) -> None:
        masked_grad = grad * routing_mask.unsqueeze(-1)
        norms = torch.linalg.norm(masked_grad.float(), dim=-1)

        token_indices, expert_indices = torch.where(routing_mask)

        for token_idx, expert_idx in zip(token_indices, expert_indices, strict=True):
            cache.append({
                "token_idx": token_idx.item(),
                "layer_idx": layer_idx,
                "expert_idx": expert_idx.item(),
                "norm": norms[token_idx, expert_idx].item(),
            })

    return hook


class TinyOnnMoE(Qwen3MLP):
    def __init__(self, config: TinyOnnConfig):
        super().__init__(config)
        self.num_experts = config.num_experts_per_layer
        self.gate = TinyOnnGate(config)
        self.experts = nn.ModuleList(
            [TinyOnnExpert(config) for _ in range(self.num_experts)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        surprise_cache: list[dict] | None = None,
        logits_cache: list[torch.Tensor] | None = None,
        layer_idx: int = -1,
    ):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(flat_hidden_states)
        if logits_cache is not None:
            logits_cache.append(router_logits.detach())
        activated_logits = F.relu(router_logits - self.gate.gates)
        routing_mask = (activated_logits > 0).bool()

        k_per_token = routing_mask.sum(dim=-1)
        zero_k_mask = k_per_token == 0
        if torch.any(zero_k_mask):
            top_expert_indices = router_logits[zero_k_mask].argmax(dim=-1)
            fallback_mask = F.one_hot(top_expert_indices, num_classes=self.num_experts).bool()
            full_fallback_mask = torch.zeros_like(routing_mask)
            full_fallback_mask[zero_k_mask] = fallback_mask
            routing_mask = routing_mask | full_fallback_mask

        masked_logits = torch.where(routing_mask, router_logits, torch.tensor(float('-inf'), device=router_logits.device, dtype=router_logits.dtype))
        routing_weights = F.softmax(masked_logits, dim=-1, dtype=torch.float).to(hidden_states.dtype)

        all_expert_outputs = torch.stack(
            [expert(flat_hidden_states) for expert in self.experts], dim=1
        )

        if surprise_cache is not None and self.training:
            hook = _create_full_hook(surprise_cache, routing_mask, layer_idx)
            all_expert_outputs.register_hook(hook)

        final_output = torch.einsum('be,bed->bd', routing_weights, all_expert_outputs)

        return final_output.view(batch_size, sequence_length, hidden_dim), router_logits, k_per_token, routing_mask


class TinyOnnDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: TinyOnnConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = TinyOnnMoE(config)
        self.layer_idx = layer_idx

    def forward(self, hidden_states, position_embeddings, *args, **kwargs):
        surprise_cache = kwargs.pop("surprise_cache", None)
        logits_cache = kwargs.pop("logits_cache", None)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_outputs = self.self_attn(hidden_states, *args, position_embeddings=position_embeddings, **kwargs)
        hidden_states = attn_outputs[0]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states_pre_mlp = self.post_attention_layernorm(hidden_states)

        mlp_output, router_logits, k_per_token, routing_mask = self.mlp(
            hidden_states_pre_mlp,
            surprise_cache=surprise_cache,
            logits_cache=logits_cache,
            layer_idx=self.layer_idx,
        )
        hidden_states = residual + mlp_output

        outputs = (hidden_states,) + attn_outputs[1:]

        aux_outputs = {
            "router_logits": router_logits,
            "k_per_token": k_per_token,
            "routing_mask": routing_mask,
        }
        return outputs, aux_outputs


class TinyOnnModel(Qwen3Model):
    def __init__(self, config: TinyOnnConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [TinyOnnDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()

    def forward(self, input_ids, attention_mask=None, position_embeddings=None, **kwargs):
        surprise_cache = kwargs.pop("surprise_cache", None)
        logits_cache = kwargs.pop("logits_cache", None)
        hidden_states = self.embed_tokens(input_ids)


        all_aux_outputs = {
            "router_logits": [],
            "k_per_token": [],
            "routing_mask": [],
        }

        for layer in self.layers:
            kwargs_for_layer = kwargs.copy()
            if surprise_cache is not None:
                kwargs_for_layer['surprise_cache'] = surprise_cache
                if logits_cache is not None:
                    kwargs_for_layer['logits_cache'] = logits_cache

            layer_outputs, aux_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                **kwargs_for_layer,
            )
            hidden_states = layer_outputs[0]
            for key, value in aux_outputs.items():
                all_aux_outputs[key].append(value)

        hidden_states = self.norm(hidden_states)
        return hidden_states, all_aux_outputs

    def forward_gate_only(self, input_ids, attention_mask=None, position_embeddings=None, **kwargs):
        hidden_states = self.embed_tokens(input_ids)
        all_router_logits = []
        for layer in self.layers:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            attn_outputs = layer.self_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = residual + attn_outputs[0]
            hidden_states_pre_mlp = layer.post_attention_layernorm(hidden_states)
            router_logits = layer.mlp.gate(hidden_states_pre_mlp.view(-1, hidden_states_pre_mlp.shape[-1]))
            all_router_logits.append(router_logits)
        return torch.cat(all_router_logits, dim=0)


class TinyOnnForCausalLM(Qwen3ForCausalLM):
    config_class = TinyOnnConfig  # type: ignore[assignment]

    def __init__(self, config: TinyOnnConfig):
        super().__init__(config)
        self.model = TinyOnnModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, position_embeddings=None, **kwargs):
        surprise_cache = kwargs.pop("surprise_cache", None)
        logits_cache = kwargs.pop("logits_cache", None)

        hidden_states, aux_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            surprise_cache=surprise_cache,
            logits_cache=logits_cache,
            **kwargs,
        )
        logits = self.lm_head(hidden_states)

        kwargs.pop("output_router_logits", None)

        output = CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=None,
            hidden_states=hidden_states,
            attentions=None
        )
        output.aux_outputs = aux_outputs
        return output

    def forward_gate_only(self, *args, **kwargs):
        return self.model.forward_gate_only(*args, **kwargs)
