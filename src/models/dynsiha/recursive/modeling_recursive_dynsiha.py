import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from typing import Optional, Union, Dict, Any, List, Tuple
from .configuration_recursive_dynsiha import RecursiveDynSIHAConfig
from ..shared.layers import DynSIHABlock, DynSIHARotaryEmbedding, DynSIHARMSNorm

class RecursiveDynSIHAPreTrainedModel(PreTrainedModel):
    config_class = RecursiveDynSIHAConfig
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

class RecursiveDynSIHAForCausalLM(RecursiveDynSIHAPreTrainedModel, GenerationMixin):
    def __init__(self, config: RecursiveDynSIHAConfig):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = DynSIHARotaryEmbedding(
            config.hidden_size // config.num_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.block = DynSIHABlock(
            config.hidden_size,
            config.num_heads,
            config.num_experts,
            config.top_k,
            layer_idx=0, # Base index, will be overridden in loop
            ffn_scale=config.ffn_scale,
            rms_norm_eps=config.rms_norm_eps
        )
        self.halt_head = nn.Linear(config.hidden_size, 1)
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
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[CausalLMOutputWithPast, Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
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

        position_embeddings = self.rotary_emb(x, seq_len=cache_position.max().item() + 1)

        all_routing_info = []
        step_losses = []
        all_halt_logits = []
        
        # PLSD: Per-Layer Speculative Decode
        # We share the same block but treat each step as a virtual layer for KV Cache
        for step in range(self.config.max_refinement_steps):
            x, routing_info, past_key_values = self.block(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                layer_idx=step # Override layer_idx for recursive cache
            )
            all_routing_info.append(routing_info)
            
            # ACT: Halt prediction
            halt_step_logits = self.halt_head(x) # [B, T, 1]
            all_halt_logits.append(halt_step_logits)
            
            if labels is not None:
                # Calculate intermediate loss for PLSD
                inter_x = self.ln_f(x)
                inter_logits = self.lm_head(inter_x)
                # Use reduction='none' to get per-token loss for more precise ACT
                step_loss = F.cross_entropy(inter_logits.view(-1, self.config.vocab_size), labels.view(-1), reduction='none')
                step_losses.append(step_loss.view(batch_size, seq_length).mean(dim=-1)) # [B]
            
            # Inference early exit logic
            if not self.training and step > 0:
                if torch.sigmoid(halt_step_logits).mean() > 0.9:
                    break

        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            # PLSD Loss: Minimum of all step losses per sample
            step_losses_tensor = torch.stack(step_losses) # [steps, B]
            best_losses, best_steps = step_losses_tensor.min(dim=0)
            main_loss = best_losses.mean()
            
            # ACT Loss (Ponder Loss variant):
            # 1. Alignment: halt_head should predict 1.0 for steps >= best_step
            halt_logits_tensor = torch.stack(all_halt_logits).squeeze(-1) # [steps, B, T]
            # Average over T for simplicity in this version
            halt_probs = torch.sigmoid(halt_logits_tensor.mean(dim=-1)) # [steps, B]
            
            target_halt = torch.zeros_like(halt_probs)
            for b in range(batch_size):
                target_halt[best_steps[b]:, b] = 1.0
            
            act_loss = F.binary_cross_entropy(halt_probs, target_halt)
            
            # 2. Ponder Penalty: encourage fewer steps
            ponder_penalty = (best_steps.float() / self.config.max_refinement_steps).mean()
            
            loss = main_loss + 0.1 * act_loss + 0.01 * ponder_penalty

        if not return_dict:
            return (loss, logits, all_routing_info) if loss is not None else (logits, all_routing_info)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=None,
            attentions=None,
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
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
            else:
                cache_length = past_key_values[0][0].shape[2]

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
