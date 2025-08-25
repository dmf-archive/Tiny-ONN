from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3Model
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, BaseModelOutputWithPast
import torch.nn.functional as F

class GateSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output

class GatingNetwork(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.moe_config = config.moe
        self.hidden_size = config.hidden_size
        self.sim_matrix = nn.Parameter(torch.randn(self.hidden_size, self.moe_config.num_experts))
        self.gates = nn.Parameter(torch.zeros(self.moe_config.num_experts))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, c = hidden_states.shape
        flat_hidden_states = hidden_states.view(b * t, c)
        logit_scale = torch.sigmoid(self.temperature)
        logits = torch.matmul(F.normalize(flat_hidden_states, dim=-1), F.normalize(self.sim_matrix, dim=0))
        
        scaled_logits = logits * logit_scale
        scaled_gates = self.gates * logit_scale
        gated_logits = F.relu(scaled_logits - scaled_gates)
        activation_mask = GateSTE.apply(gated_logits)
        
        inactive_mask = torch.sum(activation_mask, dim=1) == 0
        if torch.any(inactive_mask):
            inactive_logits = logits[inactive_mask]
            fallback_indices = torch.topk(inactive_logits, self.moe_config.min_experts_per_tok, dim=-1).indices
            inactive_b_indices = torch.where(inactive_mask)[0]
            if inactive_b_indices.numel() > 0:
                activation_mask.index_put_(
                    (inactive_b_indices.unsqueeze(1).expand(-1, self.moe_config.min_experts_per_tok), fallback_indices),
                    torch.tensor(1.0, device=hidden_states.device, dtype=activation_mask.dtype),
                )
        return activation_mask, logits

class DynMoE(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.moe_config = config.moe
        self.gating = GatingNetwork(config)
        self.w1 = nn.Parameter(torch.empty(self.moe_config.num_experts, config.hidden_size, self.moe_config.intermediate_size))
        self.w2 = nn.Parameter(torch.empty(self.moe_config.num_experts, self.moe_config.intermediate_size, config.hidden_size))
        for i in range(self.moe_config.num_experts):
            nn.init.kaiming_uniform_(self.w1[i], a=5**0.5)
            nn.init.kaiming_uniform_(self.w2[i], a=5**0.5)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, c = hidden_states.shape
        activation_mask, router_logits = self.gating(hidden_states)
        activation_mask = activation_mask.view(b, t, self.moe_config.num_experts)
        
        intermediate_states = torch.einsum('btc,eci->btei', hidden_states, self.w1)
        intermediate_states = F.gelu(intermediate_states)
        expert_outputs = torch.einsum('btei,eic->btec', intermediate_states, self.w2)
        
        masked_outputs = expert_outputs * activation_mask.unsqueeze(-1)
        summed_outputs = masked_outputs.sum(dim=-2)
        num_activated = activation_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        final_output = summed_outputs / num_activated
        return final_output, router_logits

class DynONNDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = DynMoE(config)
    
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> Tuple:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        attn_outputs = self.self_attn(hidden_states, *args, **kwargs)
        hidden_states = attn_outputs[0]
        
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,) + attn_outputs[1:] + (router_logits,)
        return outputs

class ArcModel(Qwen3Model):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.layers = nn.ModuleList([DynONNDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.post_init()

class ArcTransformer(Qwen3ForCausalLM):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.model = ArcModel(config)
        self.post_init()

    def get_sdl_loss(self, router_logits: torch.Tensor, sim_matrix: torch.Tensor) -> torch.Tensor:
        if sim_matrix.shape[1] < 2: return torch.tensor(0.0, device=sim_matrix.device)
        sim_matrix_normalized = F.normalize(sim_matrix, dim=0)
        sims = torch.matmul(sim_matrix_normalized.T, sim_matrix_normalized)
        targets = torch.eye(sims.shape[0]).to(sims.device)
        sim_loss = torch.norm(sims - targets)
        simple_loss = torch.mean(torch.norm(sim_matrix, dim=0))
        return sim_loss + simple_loss

    def forward(self, *args, **kwargs) -> MoeCausalLMOutputWithPast:
        labels = kwargs.get("labels", None)
        
        outputs: BaseModelOutputWithPast = self.model(*args, **kwargs, output_hidden_states=True)
        
        logits = self.lm_head(outputs.last_hidden_state)
        
        loss = None
        sdl_loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1), ignore_index=-100)
            
            all_router_logits = [output[-1] for output in outputs.hidden_states]
            all_sim_matrices = [layer.mlp.gating.sim_matrix for layer in self.model.layers]
            
            total_sdl_loss = 0
            for router_logit, sim_matrix in zip(all_router_logits, all_sim_matrices):
                total_sdl_loss += self.get_sdl_loss(router_logit, sim_matrix)
            
            sdl_loss = total_sdl_loss / len(all_router_logits)
            loss += sdl_loss * 0.01

        return MoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=None,
            aux_loss=sdl_loss,
        )
