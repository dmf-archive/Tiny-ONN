# coding=utf-8
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
)

PRUNE_SELF_ATTN_THRESHOLD = 3
PRUNE_MLP_THRESHOLD_DIVISOR = 2


class FuncPrunedQwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.prune_self_attn = self.layer_idx < PRUNE_SELF_ATTN_THRESHOLD
        self.prune_mlp = (
            self.layer_idx >= config.num_hidden_layers // PRUNE_MLP_THRESHOLD_DIVISOR
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        self_attn_weights = None

        # Self Attention Block
        residual = hidden_states
        hidden_states_ln = self.input_layernorm(hidden_states)

        if self.prune_self_attn:
            # If pruned, skip computation. hidden_states remains residual.
            hidden_states = residual
        else:
            attn_outputs, self_attn_weights = self.self_attn(
                hidden_states=hidden_states_ln,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = residual + attn_outputs

        # MLP Block
        residual = hidden_states
        hidden_states_ln = self.post_attention_layernorm(hidden_states)

        if self.prune_mlp:
            # If pruned, skip computation. hidden_states remains residual.
            hidden_states = residual
        else:
            mlp_output = self.mlp(hidden_states_ln)
            hidden_states = residual + mlp_output

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (past_key_value,)
        return outputs


class FuncPrunedQwen3Model(Qwen3Model):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                FuncPrunedQwen3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.post_init()


class FuncPrunedQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.model = FuncPrunedQwen3Model(config)
        self.post_init()
