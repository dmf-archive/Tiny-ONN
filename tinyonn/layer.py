
import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
)

from tinyonn.moe import MoELayer


class TinyONNLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx):
        # Bypassing the parent's __init__ to prevent it from creating a dense MLP.
        # We call nn.Module's __init__ and then manually reconstruct the layer.
        nn.Module.__init__(self)

        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        # Directly instantiate our MoELayer, preventing the dense Qwen2MLP
        # from ever being created. This is the core fix.
        self.mlp = MoELayer(config)

        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = attn_outputs[0]
        self_attn_weights: torch.FloatTensor | None = attn_outputs[1]
        present_key_value: tuple[torch.FloatTensor, torch.FloatTensor] | None = attn_outputs[2]

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # The flag is now retrieved from the model's config
        force_dense_equivalent_test = getattr(self.mlp.config, "force_dense_equivalent_test", False)
        hidden_states = self.mlp(hidden_states, force_dense_equivalent_test=force_dense_equivalent_test)

        hidden_states = residual + hidden_states

        outputs: tuple[torch.FloatTensor, ...] = (hidden_states,)

        if output_attentions and self_attn_weights is not None:
            outputs += (self_attn_weights,)

        if use_cache and present_key_value is not None:
            outputs += (present_key_value,)

        return outputs
