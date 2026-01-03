import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    apply_rotary_pos_emb,
)

from .config import FlatDynSIHAConfig


class CAPRRouter(nn.Module):
    def __init__(self, config: FlatDynSIHAConfig, num_prototypes: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_prototypes = num_prototypes
        self.routing_gain = config.routing_gain

        self.q_norm = Qwen3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.k_proto = nn.Parameter(torch.empty(num_prototypes, self.hidden_size))
        self.v_proto = nn.Parameter(torch.empty(num_prototypes, num_prototypes))

        self.dropout = nn.Dropout(config.routing_dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.k_proto, std=0.02)
        nn.init.eye_(self.v_proto)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = self.q_norm(hidden_states)
        attn_weights = torch.matmul(query, self.k_proto.t()) / math.sqrt(self.hidden_size)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        routing_logits = torch.matmul(attn_weights, self.v_proto)
        return routing_logits * self.routing_gain


class DynamicInfiniteHeadAttention(nn.Module):
    def __init__(self, config: FlatDynSIHAConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_physical_heads = config.num_physical_heads

        self.router = CAPRRouter(config, self.num_physical_heads)

        self.q_basis = nn.Parameter(torch.empty(self.num_physical_heads, self.hidden_size, self.head_dim))
        self.k_basis = nn.Parameter(torch.empty(self.num_physical_heads, self.hidden_size, self.head_dim))
        self.v_basis = nn.Parameter(torch.empty(self.num_physical_heads, self.hidden_size, self.head_dim))

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.scaling = self.head_dim**-0.5

        self.reset_parameters()

    def reset_parameters(self):
        std = self.config.initializer_range
        nn.init.normal_(self.q_basis, std=std)
        nn.init.normal_(self.k_basis, std=std)
        nn.init.normal_(self.v_basis, std=std)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        B, T, D = hidden_states.shape

        routing_weights = F.softmax(self.router(hidden_states), dim=-1)

        q_eff = torch.einsum("btp,pdh->btdh", routing_weights, self.q_basis)
        k_eff = torch.einsum("btp,pdh->btdh", routing_weights, self.k_basis)
        v_eff = torch.einsum("btp,pdh->btdh", routing_weights, self.v_basis)

        query_states = (
            torch.einsum("btd,btdh->bth", hidden_states, q_eff)
            .view(B, T, 1, self.head_dim)
            .expand(-1, -1, self.num_heads, -1)
        )
        key_states = (
            torch.einsum("btd,btdh->bth", hidden_states, k_eff)
            .view(B, T, 1, self.head_dim)
            .expand(-1, -1, self.num_heads, -1)
        )
        value_states = (
            torch.einsum("btd,btdh->bth", hidden_states, v_eff)
            .view(B, T, 1, self.head_dim)
            .expand(-1, -1, self.num_heads, -1)
        )

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            is_causal=past_key_values is None,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, {"routing_weights": routing_weights}


class DynamicMLP(nn.Module):
    def __init__(self, config: FlatDynSIHAConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_physical_experts
        self.top_k = config.top_k_experts

        self.router = CAPRRouter(config, self.num_experts)

        self.gate_projs = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.intermediate_size))
        self.up_projs = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.intermediate_size))
        self.down_projs = nn.Parameter(torch.empty(self.num_experts, self.intermediate_size, self.hidden_size))

        self.act_fn = F.silu
        self.reset_parameters()

    def reset_parameters(self):
        std = self.config.initializer_range
        nn.init.normal_(self.gate_projs, std=std)
        nn.init.normal_(self.up_projs, std=std)
        nn.init.normal_(self.down_projs, std=std)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        routing_logits = self.router(x)
        routing_weights = F.softmax(routing_logits, dim=-1)

        if self.top_k < self.num_experts:
            top_weights, top_indices = torch.topk(routing_weights, self.top_k, dim=-1)
            routing_weights = torch.zeros_like(routing_weights).scatter_(-1, top_indices, top_weights)
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-9)

        gate_eff = torch.einsum("btp,phi->bthi", routing_weights, self.gate_projs)
        up_eff = torch.einsum("btp,phi->bthi", routing_weights, self.up_projs)
        down_eff = torch.einsum("btp,pih->btih", routing_weights, self.down_projs)

        gate_out = torch.einsum("bth,bthi->bti", x, gate_eff)
        up_out = torch.einsum("bth,bthi->bti", x, up_eff)

        intermediate = self.act_fn(gate_out) * up_out
        output = torch.einsum("bti,btih->bth", intermediate, down_eff)

        return output, {"routing_weights": routing_weights}


class FlatDynSIHADecoderLayer(nn.Module):
    def __init__(self, config: FlatDynSIHAConfig, layer_idx: int):
        super().__init__()
        self.self_attn = DynamicInfiniteHeadAttention(config, layer_idx)
        self.mlp = DynamicMLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _, attn_info = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, mlp_info = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, None, {"attn": attn_info, "mlp": mlp_info}


class FlatDynSIHAModel(Qwen3PreTrainedModel):
    def __init__(self, config: FlatDynSIHAConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([FlatDynSIHADecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

        self.rotary_emb = Qwen3RotaryEmbedding(config)

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        if inputs_embeds is None:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            ).to(torch.long)

        if position_ids is None and cache_position is not None:
            position_ids = cache_position.unsqueeze(0).to(torch.long)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_routing_info = []
        for layer in self.layers:
            hidden_states, _, routing_info = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
            all_routing_info.append(routing_info)

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=None,
            attentions=None,
        )


class FlatDynSIHAForCausalLM(Qwen3PreTrainedModel):
    def __init__(self, config: FlatDynSIHAConfig):
        super().__init__(config)
        self.model = FlatDynSIHAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1), ignore_index=-100)

        return CausalLMOutputWithPast(
            loss=loss.to(torch.float32) if loss is not None else None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
