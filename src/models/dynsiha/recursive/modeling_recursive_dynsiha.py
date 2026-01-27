from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..shared.layers import DynSIHABlock, DynSIHARMSNorm, DynSIHARotaryEmbedding
from .configuration_recursive_dynsiha import RecursiveDynSIHAConfig
import torch.utils.checkpoint as cp


@dataclass
class RecursiveDynSIHAOutput(CausalLMOutputWithPast):
    routing_info: list[dict[str, torch.Tensor]] | None = None
    exit_steps: torch.LongTensor | None = None
    all_step_losses: torch.Tensor | None = None
    best_step_mask: torch.Tensor | None = None  # [B, T] 仅神谕步为 True

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
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self): return self.embedding
    def set_input_embeddings(self, value): self.embedding = value
    def get_output_embeddings(self): return self.lm_head
    def set_output_embeddings(self, value): self.lm_head = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool = True,
        cache_position: torch.LongTensor | None = None,
        **kwargs
    ) -> CausalLMOutputWithPast | tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        # 训练期间强制禁用 KV Cache 以节省显存并简化 Checkpointing 逻辑
        if self.training:
            use_cache = False
        else:
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
        actual_steps = 0

        # PLSD: Per-Layer Speculative Decode
        # We share the same block but treat each step as a virtual layer for KV Cache
        max_steps = self.config.max_refinement_steps if self.training else self.config.max_inference_steps

        def step_fn(step_input, step_idx, pkv):
            h, r_info, new_pkv = self.block(
                step_input,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=pkv,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                layer_idx=step_idx
            )
            # Checkpoint 必须返回 Tensor 元组。我们将 dict 展开。
            return h, r_info["q_logits"], r_info["k_logits"], r_info["v_logits"], r_info["mlp_logits"]

        for step in range(max_steps):
            actual_steps = step + 1
            if self.training and self.gradient_checkpointing:
                x, ql, kl, vl, mlpl = cp.checkpoint(
                    step_fn, x, step, None, use_reentrant=False
                )
                routing_info = {"q_logits": ql, "k_logits": kl, "v_logits": vl, "mlp_logits": mlpl}
                past_key_values = None # 训练期禁用 KV Cache
            else:
                x, ql, kl, vl, mlpl = step_fn(x, step, past_key_values)
                routing_info = {"q_logits": ql, "k_logits": kl, "v_logits": vl, "mlp_logits": mlpl}
            
            all_routing_info.append(routing_info)

            # ACT: Halt prediction
            halt_step_logits = self.halt_head(x)  # [B, T, 1]
            all_halt_logits.append(halt_step_logits)

            if labels is not None:
                # Calculate intermediate loss for PLSD
                inter_x = self.ln_f(x)
                inter_logits = self.lm_head(inter_x)

                shifted_logits = inter_logits[..., :-1, :].contiguous()
                shifted_labels = labels[..., 1:].contiguous()
                step_loss = F.cross_entropy(
                    shifted_logits.view(-1, self.config.vocab_size),
                    shifted_labels.view(-1),
                    reduction='none'
                )
                step_losses.append(step_loss.view(batch_size, seq_length - 1).mean(dim=-1))  # [B]

            # Inference early exit logic
            if not self.training and step > 0:
                halt_prob = torch.sigmoid(halt_step_logits).mean()
                if halt_prob > 0.8:
                    break

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        all_step_losses_tensor = None
        best_step_mask = None
        best_steps = torch.full((batch_size,), actual_steps - 1, device=self.device, dtype=torch.long)

        if labels is not None:
            all_step_losses_tensor = torch.stack(step_losses) # [steps, B]
            best_losses, best_steps = all_step_losses_tensor.min(dim=0)
            main_loss = best_losses.mean()
            
            # 生成路径掩码 [steps, B]，用于 FARS 路径对齐
            # 只有 t <= t_best 的步才会被计入 FARS 损失
            steps_range = torch.arange(all_step_losses_tensor.shape[0], device=self.device).unsqueeze(1)
            best_step_mask = (steps_range <= best_steps.unsqueeze(0)) # [steps, B]
            
            halt_logits_tensor = torch.stack(all_halt_logits).squeeze(-1) # [steps, B, T]
            halt_probs = torch.sigmoid(halt_logits_tensor.mean(dim=-1)) # [steps, B]
            
            target_halt = torch.zeros_like(halt_probs)
            for b in range(batch_size):
                target_halt[best_steps[b]:, b] = 1.0
            
            act_loss = F.binary_cross_entropy(halt_probs, target_halt)
            loss = main_loss + 0.1 * act_loss

        if not return_dict:
            return (loss, logits, all_routing_info) if loss is not None else (logits, all_routing_info)

        return RecursiveDynSIHAOutput(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=None,
            attentions=None,
            routing_info=all_routing_info,
            # 返回 1-based 的步数，方便观察
            exit_steps=best_steps + 1,
            all_step_losses=all_step_losses_tensor,
            best_step_mask=best_step_mask
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

        position_ids = kwargs.get("position_ids")
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
