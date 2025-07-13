from typing import ClassVar, cast

import torch
import torch.nn as nn
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from .configuration_tinyonn import TinyONNConfig
from .layer import TinyONNLayer


class TinyONNPreTrainedModel(PreTrainedModel):
    """
    The base class for our custom TinyONN model, linking it to the TinyONNConfig
    and handling weight initialization.
    """

    config_class = TinyONNConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules: ClassVar[list[str]] = ["TinyONNLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class TinyONNModel(TinyONNPreTrainedModel):
    """
    The bare TinyONN Model transformer outputting raw hidden-states without any specific head on top.
    """
    def __init__(self, config: TinyONNConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([TinyONNLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: tuple[tuple[torch.FloatTensor, torch.FloatTensor], ...] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        all_hidden_states: tuple[torch.FloatTensor, ...] | None = () if output_hidden_states else None
        all_self_attns: tuple[torch.FloatTensor, ...] | None = () if output_attentions else None
        next_decoder_cache: list[tuple[torch.FloatTensor, torch.FloatTensor]] | None = [] if use_cache else None

        for decoder_layer in self.layers:
            if output_hidden_states and all_hidden_states is not None:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,  # Not supporting KV cache for now
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache and next_decoder_cache is not None:
                # layer_outputs[2 if output_attentions else 1] is a tuple of (key_value, key_value)
                next_decoder_cache.append(cast(tuple[torch.FloatTensor, torch.FloatTensor], layer_outputs[2 if output_attentions else 1]))

            if output_attentions and all_self_attns is not None:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states and all_hidden_states is not None:
            all_hidden_states += (hidden_states,)

        next_cache: tuple[tuple[torch.FloatTensor, torch.FloatTensor], ...] | None = tuple(next_decoder_cache) if use_cache and next_decoder_cache is not None else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class TinyONNForCausalLM(TinyONNPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = TinyONNModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: tuple[tuple[torch.FloatTensor, torch.FloatTensor], ...] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | CausalLMOutputWithPast:
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
            return_dict=True, # Force dict for easier processing
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits, *outputs.values()[1:])
            return (loss, *output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
