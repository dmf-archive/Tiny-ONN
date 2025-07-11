from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

class PrunedQwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.total_layers = config.num_hidden_layers
        self.prune_self_attn = self.layer_idx < 3 # Only prune first 3 self_attn layers (0, 1, 2)
        self.prune_mlp = self.layer_idx >= (self.total_layers // 2) # Prune last 1/2 of MLP layers

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
        present_key_value = past_key_value

        if self.prune_self_attn:
            # If pruned, the attention block is skipped. The output is the input.
            # hidden_states remains residual
            pass
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
        if self.prune_mlp:
            # If pruned, the MLP block is skipped. The output is the input.
            pass
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
