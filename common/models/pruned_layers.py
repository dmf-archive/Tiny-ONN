from typing import Any

import torch
from torch import FloatTensor, nn
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
)

# Constants for pruning logic
PRUNE_SELF_ATTN_THRESHOLD = 3
PRUNE_MLP_THRESHOLD_DIVISOR = 2

class PrunedQwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.prune_self_attn = self.layer_idx < PRUNE_SELF_ATTN_THRESHOLD
        self.prune_mlp = self.layer_idx >= (config.num_hidden_layers // PRUNE_MLP_THRESHOLD_DIVISOR)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[FloatTensor, tuple[FloatTensor, FloatTensor] | None]:
        residual = hidden_states

        # Self Attention Block
        present_key_value_output: tuple[FloatTensor, FloatTensor] | None = None

        if self.prune_self_attn:
            if use_cache:
                if past_key_value is not None:
                    # Explicitly cast past_key_value to the expected type
                    present_key_value_output = past_key_value # type: ignore[assignment]
                else:
                    batch_size, _, _ = hidden_states.shape
                    num_heads = self.self_attn.num_key_value_heads
                    head_dim = self.self_attn.head_dim

                    dtype = hidden_states.dtype
                    device = hidden_states.device

                    empty_shape: tuple[int, ...] = (batch_size, num_heads, 0, head_dim) # type: ignore[assignment]
                    dummy_keys = torch.empty(empty_shape, dtype=dtype, device=device)
                    dummy_values = torch.empty(empty_shape, dtype=dtype, device=device)
                    present_key_value_output = (dummy_keys.to(dtype), dummy_values.to(dtype)) # type: ignore[assignment]
        else:
            hidden_states_ln = self.input_layernorm(hidden_states)
            # Declare attn_outputs as Any to bypass Mypy's complex type inference for transformers
            attn_outputs: Any = self.self_attn( # type: ignore[assignment]
                hidden_states=hidden_states_ln,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = residual + attn_outputs[0]
            if use_cache and attn_outputs[2] is not None:
                present_key_value_output = attn_outputs[2] # type: ignore[assignment]
            else:
                present_key_value_output = None

        # MLP Block
        residual = hidden_states
        if not self.prune_mlp:
            hidden_states_ln = self.post_attention_layernorm(hidden_states)
            mlp_output = self.mlp(hidden_states_ln)
            hidden_states = residual + mlp_output

        # Qwen3DecoderLayer returns (hidden_states, present_key_value)
        return hidden_states.to(torch.float32), present_key_value_output # type: ignore[return-value] # Explicitly cast to torch.float32


class PrunedQwen3Model(Qwen3Model):
    """A Qwen3Model that uses PrunedQwen3DecoderLayer."""
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [PrunedQwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()

class PrunedQwen3ForCausalLM(Qwen3ForCausalLM):
    """A Qwen3ForCausalLM that uses the PrunedQwen3Model."""
    def __init__(self, config):
        super().__init__(config)
        self.model = PrunedQwen3Model(config)
