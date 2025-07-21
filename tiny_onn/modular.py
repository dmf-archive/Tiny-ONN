from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from .config import TinyOnnConfig


@dataclass
class ModelOutputWithRouterLogits(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[Tuple[torch.Tensor, ...]] = None
    selected_experts: Optional[Tuple[torch.Tensor, ...]] = None
    per_expert_surprise: Optional[Tuple[torch.Tensor, ...]] = None
    token_hidden_state: Optional[Tuple[torch.Tensor, ...]] = None


@dataclass
class CausalLMOutputWithRouterLogits(CausalLMOutputWithPast):
    router_logits: Optional[Tuple[torch.Tensor, ...]] = None
    selected_experts: Optional[Tuple[torch.Tensor, ...]] = None
    aux_loss: Optional[torch.FloatTensor] = None
    per_expert_surprise: Optional[Tuple[torch.Tensor, ...]] = None
    token_hidden_state: Optional[Tuple[torch.Tensor, ...]] = None


class SiLUHardThreshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_scores: torch.Tensor, threshold_param: torch.Tensor):
        activated_values = F.silu(input_scores - threshold_param)
        mask = (activated_values > 0).float()
        ctx.save_for_backward(input_scores, threshold_param, activated_values)
        return mask

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_scores, threshold_param, activated_values = ctx.saved_tensors
        
        silu_input = input_scores - threshold_param
        sigmoid_silu_input = torch.sigmoid(silu_input)
        grad_silu = sigmoid_silu_input * (1 + silu_input * (1 - sigmoid_silu_input))
        
        grad_input_scores = grad_output * grad_silu
        grad_threshold_param = -grad_output * grad_silu
        
        return grad_input_scores, grad_threshold_param


class DynMoE(nn.Module):
    def __init__(self, config: TinyOnnConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        if self.top_k == -1:
            self.expert_embeddings = nn.Parameter(
                torch.randn(self.num_experts, self.hidden_size)
            )
            self.gate_threshold = nn.Parameter(torch.zeros(self.num_experts))
        else:
            self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        expert_config = deepcopy(config)
        expert_config.intermediate_size = config.moe_intermediate_size
        self.experts = nn.ModuleList(
            [Qwen3MLP(expert_config) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        token_idx = None
        expert_idx = None

        if self.top_k == -1:
            scores = F.cosine_similarity(
                hidden_states_flat.unsqueeze(1),
                self.expert_embeddings.unsqueeze(0),
                dim=-1,
            )

            routing_mask = SiLUHardThreshold.apply(scores, self.gate_threshold)
            
            routing_mask_for_grad = routing_mask.clone()
            
            num_selected = routing_mask.sum(dim=-1)
            at_least_one_mask = num_selected == 0
            if torch.any(at_least_one_mask):
                top_indices = torch.topk(scores[at_least_one_mask], 1, dim=-1).indices
                routing_mask = routing_mask.scatter(
                    -1, top_indices.unsqueeze(-1).expand(-1, 1), 1.0
                )
            
            token_idx, expert_idx = torch.where(routing_mask.bool())
            
            router_logits = scores
            
            masked_scores = scores.masked_fill(~routing_mask_for_grad.bool(), float('-inf'))
            routing_weights = F.softmax(masked_scores, dim=-1, dtype=torch.float).to(hidden_states.dtype)
            
            selected_experts_info = (token_idx, expert_idx)

            final_hidden_states = torch.zeros_like(hidden_states_flat)
            unique_expert_ids = torch.unique(expert_idx)

            for current_expert_id in unique_expert_ids:
                expert_id = current_expert_id.item()
                token_indices_for_expert = token_idx[expert_idx == expert_id]
                
                weights_for_expert = routing_weights[token_indices_for_expert, expert_id].unsqueeze(-1)
                
                expert_input = hidden_states_flat[token_indices_for_expert]
                expert_output = self.experts[expert_id](expert_input)
                
                final_hidden_states.index_add_(
                    0,
                    token_indices_for_expert,
                    expert_output * weights_for_expert,
                )

        else:
            router_logits = self.gate(hidden_states_flat)
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            
            routing_weights, selected_experts = torch.topk(
                routing_weights, self.top_k, dim=-1
            )
            
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = routing_weights.to(hidden_states.dtype)

            final_hidden_states = torch.zeros_like(hidden_states_flat)
            
            expert_mask = F.one_hot(
                selected_experts, num_classes=self.num_experts
            ).permute(2, 1, 0) 
            
            for expert_id in range(self.num_experts):
                if expert_mask[expert_id].sum() == 0:
                    continue
                
                idx, top_x = torch.where(expert_mask[expert_id].squeeze(0))
                
                current_state = hidden_states_flat[top_x] 
                
                weights_for_expert = routing_weights[top_x, idx].unsqueeze(-1) 
                
                current_hidden_states = (
                    self.experts[expert_id](current_state)
                    * weights_for_expert
                )
                
                final_hidden_states.index_add_(
                    0, top_x, current_hidden_states.to(hidden_states.dtype)
                )
            selected_experts_info = selected_experts

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return_info = {
            'token_idx': token_idx,
            'expert_idx': expert_idx
        } if self.top_k == -1 else {
            'indices': selected_experts
        }
        return final_hidden_states, router_logits, return_info

class TinyOnnDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, config: TinyOnnConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = DynMoE(config)
        else:
            self.mlp = Qwen3MLP(config, intermediate_size=config.intermediate_size)
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=None,
        output_router_logits=False,
        **kwargs
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        if isinstance(self.mlp, DynMoE):
            hidden_states, router_logits, _ = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
            router_logits = None
            
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
            
        if output_router_logits and isinstance(self.mlp, DynMoE) and router_logits is not None:
            outputs += (router_logits,)

        return outputs
        

class TinyOnnModel(Qwen3MoeModel):
    def __init__(self, config: TinyOnnConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [TinyOnnDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.post_init()


class TinyOnnForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, config: TinyOnnConfig):
        super().__init__(config)
        self.model = TinyOnnModel(config)
        self.post_init()

    def forward(
        self,
        *args,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithRouterLogits]:
        
        outputs = super().forward(*args, **kwargs)
        
        return CausalLMOutputWithRouterLogits(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=getattr(outputs, "router_logits", None), 
            selected_experts=getattr(outputs, "selected_experts", None), 
            aux_loss=getattr(outputs, "aux_loss", None), 
        )
