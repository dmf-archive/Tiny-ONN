from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple, List, Dict

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


class GateSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


@dataclass
class CausalLMOutputWithAux(CausalLMOutputWithPast):
    aux_outputs: Optional[List[Dict[str, torch.Tensor]]] = None


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
        self.gates = nn.Parameter(torch.zeros(config.num_experts_per_layer))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        norm_hidden_states = F.normalize(hidden_states, dim=-1)
        norm_sim_matrix = F.normalize(self.sim_matrix, dim=0)
        raw_logits = torch.matmul(norm_hidden_states, norm_sim_matrix)
        
        scaled_logits = torch.sigmoid(raw_logits)
        scaled_gates = torch.sigmoid(self.gates)
        activated_scores = F.relu(scaled_logits - scaled_gates)
        return raw_logits, scaled_logits, activated_scores


def _create_sparse_hook(
    cache: List[Dict[str, Any]],
    layer_idx: int,
    expert_idx: int,
    token_indices: torch.Tensor,
) -> Callable:
    def hook(grad: torch.Tensor) -> None:
        if grad is not None:
            norms = torch.linalg.norm(grad.float(), dim=-1)
            for i, token_idx in enumerate(token_indices):
                cache.append({
                    "layer_idx": layer_idx,
                    "token_idx": token_idx.item(),
                    "expert_idx": expert_idx,
                    "norm": norms[i].item(),
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
        self.register_buffer("routing_records", torch.zeros(self.num_experts, dtype=torch.long))

    def reset_routing_records(self):
        self.routing_records.zero_()

    def forward(
        self,
        hidden_states: torch.Tensor,
        surprise_cache: Optional[List[Dict]] = None,
        layer_idx: int = -1,
        ignored_experts: Optional[set[int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden_states = hidden_states.view(-1, hidden_dim)

        raw_logits, scaled_logits, activated_scores = self.gate(flat_hidden_states)
        routing_mask = (activated_scores > 0).bool()

        if self.training:
            self.routing_records.add_(routing_mask.sum(dim=0).to(self.routing_records.device))

        if ignored_experts:
            for expert_idx in ignored_experts:
                routing_mask[:, expert_idx] = False

        k_per_token = routing_mask.sum(dim=-1)
        if torch.any(k_per_token == 0):
            fallback_mask = F.one_hot(raw_logits[k_per_token == 0].argmax(dim=-1), num_classes=self.num_experts).bool()
            routing_mask[k_per_token == 0] = fallback_mask
        
        ste_scores = GateSTE.apply(activated_scores)
        masked_scores = torch.where(routing_mask, ste_scores, torch.tensor(float('-inf'), device=ste_scores.device, dtype=ste_scores.dtype))
        routing_weights = F.softmax(masked_scores, dim=-1, dtype=torch.float).to(hidden_states.dtype)

        aux_output = {
            "raw_logits": raw_logits.detach(),
            "activated_scores": activated_scores.detach(),
            "routing_weights": routing_weights.detach(),
        }

        final_hidden_states = torch.zeros_like(flat_hidden_states)
        token_indices, expert_indices = torch.where(routing_mask)
        
        for expert_idx in torch.unique(expert_indices).tolist():
            expert = self.experts[expert_idx]
            token_mask_for_expert = (expert_indices == expert_idx)
            tokens_for_expert = token_indices[token_mask_for_expert]

            if tokens_for_expert.numel() > 0:
                current_hidden_states = flat_hidden_states[tokens_for_expert]
                if self.training and surprise_cache is not None and current_hidden_states.requires_grad:
                    current_hidden_states.register_hook(_create_sparse_hook(surprise_cache, layer_idx, expert_idx, tokens_for_expert))
                
                expert_output = expert(current_hidden_states)
                weighted_output = expert_output * routing_weights[tokens_for_expert, expert_idx].unsqueeze(1)
                final_hidden_states.index_add_(0, tokens_for_expert, weighted_output)

        return final_hidden_states.view(batch_size, sequence_length, hidden_dim), aux_output


class TinyOnnDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: TinyOnnConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = TinyOnnMoE(config)
        self.layer_idx = layer_idx

    def forward(self, *args, **kwargs):
        aux_outputs_list = kwargs.get("aux_outputs_list")
        
        residual = args[0]
        hidden_states = self.input_layernorm(args[0])
        
        attn_outputs = self.self_attn(hidden_states, *args[1:], **kwargs)
        hidden_states = attn_outputs[0]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states_pre_mlp = self.post_attention_layernorm(hidden_states)
        
        ignored_experts = kwargs.get("ignored_experts")
        layer_ignored_experts = ignored_experts.get(self.layer_idx) if ignored_experts else None
        
        surprise_cache = kwargs.get("surprise_cache")
        mlp_output, aux_output = self.mlp(
            hidden_states_pre_mlp,
            surprise_cache=surprise_cache,
            layer_idx=self.layer_idx,
            ignored_experts=layer_ignored_experts,
        )

        if aux_outputs_list is not None:
            aux_outputs_list.append(aux_output)

        hidden_states = residual + mlp_output
        outputs = (hidden_states,) + attn_outputs[1:]
        return outputs


class TinyOnnModel(Qwen3Model):
    def __init__(self, config: TinyOnnConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [TinyOnnDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        aux_outputs_list: List[Dict[str, torch.Tensor]] = []
        kwargs["aux_outputs_list"] = aux_outputs_list
        
        hidden_states = super().forward(*args, **kwargs)[0]
        
        return hidden_states, aux_outputs_list


class TinyOnnForCausalLM(Qwen3ForCausalLM):
    config_class = TinyOnnConfig

    def __init__(self, config: TinyOnnConfig):
        super().__init__(config)
        self.model = TinyOnnModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, *args, **kwargs) -> CausalLMOutputWithAux:
        hidden_states, aux_outputs = self.model(*args, **kwargs)
        logits = self.lm_head(hidden_states)
        
        return CausalLMOutputWithAux(
            logits=logits,
            aux_outputs=aux_outputs,
        )
