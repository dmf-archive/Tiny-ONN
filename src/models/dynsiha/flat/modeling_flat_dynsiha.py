import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput
from transformers.cache_utils import Cache, DynamicCache
from typing import Optional, Union, Dict, Any, List, Tuple
from .configuration_flat_dynsiha import FlatDynSIHAConfig
from ..shared.layers import DynSIHABlock, DynSIHARotaryEmbedding, DynSIHARMSNorm

@dataclass
class FlatDynSIHAOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    routing_info: Optional[List[Dict[str, torch.Tensor]]] = None

class FlatDynSIHAPreTrainedModel(PreTrainedModel):
    config_class = FlatDynSIHAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DynSIHABlock"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class FlatDynSIHAForCausalLM(FlatDynSIHAPreTrainedModel, GenerationMixin):
    def __init__(self, config: FlatDynSIHAConfig):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = DynSIHARotaryEmbedding(
            config.hidden_size // config.num_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.layers = nn.ModuleList([
            DynSIHABlock(
                config.hidden_size,
                config.num_heads,
                config.num_experts,
                config.top_k,
                layer_idx=i,
                ffn_scale=config.ffn_scale,
                rms_norm_eps=config.rms_norm_eps
            )
            for i in range(config.num_hidden_layers)
        ])
        self.ln_f = DynSIHARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self): return self.embedding
    def set_input_embeddings(self, value): self.embedding = value
    def get_output_embeddings(self): return self.lm_head
    def set_output_embeddings(self, value): self.lm_head = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        diff_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[FlatDynSIHAOutput, Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (kwargs.get("inputs_embeds") is not None):
            raise ValueError("You must specify either input_ids or inputs_embeds")

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        else:
            batch_size, seq_length = kwargs.get("inputs_embeds").shape[:2]

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_length, device=self.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if input_ids is not None:
            x = self.embedding(input_ids)
        else:
            x = kwargs.get("inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        # RoPE
        position_embeddings = self.rotary_emb(x, seq_len=cache_position.max().item() + 1)

        all_routing_info = []
        all_hidden_states = () if output_hidden_states else None
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (x,)
                
            x, routing_info, past_key_values = layer(
                x, 
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings
            )
            all_routing_info.append(routing_info)
            
        x = self.ln_f(x)
        
        if output_hidden_states:
            all_hidden_states += (x,)
            
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            main_loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))
            
            if diff_mask is not None:
                shifted_logits = logits[..., :-1, :].contiguous()
                shifted_labels = labels[..., 1:].contiguous()
                shifted_diff_mask = diff_mask[..., 1:].contiguous()
                
                diff_labels = torch.where(shifted_diff_mask.bool(), shifted_labels, -100)
                diff_loss = F.cross_entropy(shifted_logits.view(-1, self.config.vocab_size), diff_labels.view(-1))
                
                if torch.isnan(diff_loss):
                    diff_loss = torch.tensor(0.0, device=main_loss.device)
                
                num_diff = shifted_diff_mask.sum()
                num_total = (shifted_labels != -100).sum()
                num_identity = num_total - num_diff
                
                lambda_adaptive = (num_identity / (num_diff + 1e-6)).clamp(max=100.0)
                loss = main_loss + lambda_adaptive * diff_loss
            else:
                loss = main_loss
            
        if not return_dict:
            output = (logits,) + (all_routing_info,)
            return ((loss,) + output) if loss is not None else output

        return FlatDynSIHAOutput(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=None,
            routing_info=all_routing_info,
        )

    def prepare_inputs_for_generation(
        self, 
        input_ids, 
        past_key_values=None, 
        attention_mask=None, 
        inputs_embeds=None, 
        cache_position=None, 
        **kwargs
    ):
        # If we have past_key_values, we only need the last token
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = cache_length
                max_cache_length = None # DynamicCache doesn't always have get_max_length
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the last token if past_key_values is used
            if input_ids.shape[1] > cache_length:
                input_ids = input_ids[:, cache_length:]
            else:
                input_ids = input_ids[:, -1:]

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + input_ids.shape[1], device=input_ids.device
            )

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "cache_position": cache_position,
            }
        )
        return model_inputs
