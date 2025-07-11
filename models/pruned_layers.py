from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (  # Import necessary components
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3MLP,
    Qwen3RMSNorm,
)


class PrunedQwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx):
        # Do NOT call super().__init__(config, layer_idx) directly for the whole layer.
        # Instead, manually initialize the components that are NOT pruned.
        # This bypasses the default Qwen3DecoderLayer instantiation of all submodules.
        super(Qwen3DecoderLayer, self).__init__() # Call nn.Module's __init__

        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Determine whether to prune based on layer index
        prune_self_attn = self.layer_idx < 3  # Only prune first 3 self_attn layers (0, 1, 2)
        prune_mlp = self.layer_idx >= (config.num_hidden_layers // 2)  # Prune last 1/2 of MLP layers

        # Conditionally instantiate self_attn
        if prune_self_attn:
            self.self_attn = nn.Identity()
        else:
            self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        
        # Conditionally instantiate mlp
        if prune_mlp:
            self.mlp = nn.Identity()
        else:
            self.mlp = Qwen3MLP(config)

        # Instantiate other components that are always present in Qwen3DecoderLayer
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx] # This might need adjustment if config is not updated

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        # Self Attention Block
        residual = hidden_states
        self_attn_weights = None
        
        # If self_attn is an Identity module, it means the layer is pruned.
        if isinstance(self.self_attn, nn.Identity):
            # The residual connection is implicitly handled by doing nothing.
            present_key_value = past_key_value
        else:
            hidden_states_ln = self.input_layernorm(hidden_states)
            attn_outputs, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states_ln,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = residual + attn_outputs

        # MLP Block
        residual = hidden_states
        # If mlp is an Identity module, it means the layer is pruned.
        if isinstance(self.mlp, nn.Identity):
            pass  # The residual connection is implicitly handled.
        else:
            hidden_states_ln = self.post_attention_layernorm(hidden_states)
            mlp_output = self.mlp(hidden_states_ln)
            hidden_states = residual + mlp_output

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
