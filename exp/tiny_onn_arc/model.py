from typing import Any, Optional, Tuple

from transformers.cache_utils import Cache
import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3Model
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, BaseModelOutputWithPast, MoeModelOutputWithPast
from transformers.masking_utils import create_causal_mask
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
        self.num_experts = self.moe_config.num_experts
        
        self.sim_matrix = nn.Parameter(torch.randn(self.hidden_size, self.num_experts))
        self.gates = nn.Parameter(torch.zeros(self.num_experts))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.experts_mask = nn.Parameter(torch.ones(self.num_experts), requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, c = hidden_states.shape
        flat_hidden_states = hidden_states.view(b * t, c)
        
        logit_scale = torch.sigmoid(self.temperature)
        
        logits = torch.matmul(F.normalize(flat_hidden_states, dim=-1), F.normalize(self.sim_matrix, dim=0))
        logits = logits * self.experts_mask
        
        scaled_gates = self.gates * logit_scale
        
        gated_logits = F.relu(logits - scaled_gates)
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

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, c = hidden_states.shape
        activation_mask, router_logits = self.gating(hidden_states)
        
        intermediate_states = torch.einsum('btc,eci->btei', hidden_states, self.w1)
        intermediate_states = F.gelu(intermediate_states)
        expert_outputs = torch.einsum('btei,eic->btec', intermediate_states, self.w2)
        
        masked_outputs = expert_outputs * activation_mask.view(b, t, self.moe_config.num_experts).unsqueeze(-1)
        summed_outputs = masked_outputs.sum(dim=-2)
        num_activated = activation_mask.view(b, t, -1).sum(dim=-1, keepdim=True).clamp(min=1)
        final_output = summed_outputs / num_activated
        
        return final_output, router_logits, activation_mask

class DynONNDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = DynMoE(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        output_router_logits: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = attn_outputs[0]
        
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits, activation_mask = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,) + attn_outputs[1:]
        
        if output_router_logits:
            outputs += (router_logits, activation_mask)
            
        return outputs

class ArcModel(Qwen3Model):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.layers = nn.ModuleList([DynONNDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> MoeModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        all_activation_masks = () if output_router_logits else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                output_router_logits=output_router_logits,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            if output_router_logits:
                all_router_logits += (layer_outputs[-2],)
                all_activation_masks += (layer_outputs[-1],)


        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=(all_router_logits, all_activation_masks),
        )

class ArcTransformer(Qwen3ForCausalLM):
    _keys_to_ignore_on_load_missing = [r"model\.layers\.\d+\.mlp\.gating\.experts_mask"]

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.model = ArcModel(config)
        self.post_init()

    def get_diversity_loss(self, gating_network: GatingNetwork) -> torch.Tensor:
        if gating_network.num_experts < 2: return torch.tensor(0.0, device=gating_network.gates.device)
        
        expert_mask = gating_network.experts_mask
        gates = gating_network.gates
        
        sims = torch.matmul(F.normalize(gates.unsqueeze(0), dim=1).T, F.normalize(gates.unsqueeze(0), dim=1))
        targets = torch.eye(sims.shape[0]).to(sims.device)
        sim_mask = torch.matmul(expert_mask.unsqueeze(0).T, expert_mask.unsqueeze(0))
        sim_loss = torch.norm(sims * sim_mask - targets * sim_mask)
        
        simple_loss = torch.mean(torch.norm(gates, dim=0))
        return sim_loss + simple_loss

    def get_sparsity_loss(self, activation_masks: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        total_activated_experts = 0
        num_tokens = 0
        for mask in activation_masks:
            total_activated_experts += mask.sum()
            num_tokens += mask.shape[0]
        
        if num_tokens == 0: return torch.tensor(0.0, device=activation_masks[0].device)
        
        avg_activated_experts = total_activated_experts / num_tokens
        return F.l1_loss(avg_activated_experts, torch.tensor(self.config.moe.min_experts_per_tok, device=avg_activated_experts.device))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> MoeCausalLMOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=True,
            cache_position=cache_position,
        )
        
        logits = self.lm_head(outputs.last_hidden_state)
        
        loss = None
        aux_loss = torch.tensor(0.0, device=logits.device)
        
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1), ignore_index=-100)
            
            all_router_logits, all_activation_masks = outputs.router_logits
            
            total_diversity_loss = torch.tensor(0.0, device=logits.device)
            total_sparsity_loss = torch.tensor(0.0, device=logits.device)

            if all_router_logits:
                all_gating_networks = [layer.mlp.gating for layer in self.model.layers]
                for gating_network in all_gating_networks:
                    total_diversity_loss += self.get_diversity_loss(gating_network)
                
                total_sparsity_loss = self.get_sparsity_loss(all_activation_masks)
                
                aux_loss = (total_diversity_loss / len(all_gating_networks)) + (total_sparsity_loss * 0.01)
                loss += aux_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            aux_loss=aux_loss,
        )
