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
    exit_steps: torch.LongTensor | None = None
    all_step_losses: torch.Tensor | None = None
    best_step_mask: torch.Tensor | None = None
    halt_logits: torch.Tensor | None = None
    all_step_logits: torch.Tensor | None = None
    eff_k: torch.Tensor | None = None
    routing_weights: torch.Tensor | None = None

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
            layer_idx=0,
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
        if self.training:
            use_cache = self.config.use_cache_in_train
        else:
            use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        else:
            batch_size, seq_length = kwargs.get("inputs_embeds").shape[:2]

        if self.training and self.config.max_refinement_steps > 1:
            use_cache = False

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

        step_losses = []
        step_logits_list = []
        all_halt_logits = []
        actual_steps = 0
        active_counts = []
        routing_weights = []

        max_steps = self.config.max_refinement_steps if self.training else self.config.max_inference_steps

        def step_fn(step_input, step_idx, pkv, route_input):
            h, routing_info, new_pkv = self.block(
                step_input,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=pkv,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                layer_idx=step_idx,
                route_x=route_input
            )
            return h, new_pkv, routing_info

        prev_x = None
        for step in range(max_steps):
            actual_steps = step + 1

            if self.config.use_sia and self.training and step > 0:
                x = x.detach()

            if self.training and self.gradient_checkpointing:
                x, past_key_values, routing_info = cp.checkpoint(
                    step_fn, x, step, past_key_values, prev_x, use_reentrant=False
                )
            else:
                x, past_key_values, routing_info = step_fn(x, step, past_key_values, prev_x)

            q_active = routing_info.get("q_active")
            k_active = routing_info.get("k_active")
            v_active = routing_info.get("v_active")
            mlp_active = routing_info.get("mlp_active")
            if q_active is not None and k_active is not None and v_active is not None and mlp_active is not None:
                q_mean = q_active.float().mean(dim=-1)
                k_mean = k_active.float().mean(dim=-1)
                v_mean = v_active.float().mean(dim=-1)
                active_counts.append((q_mean + k_mean + v_mean + mlp_active.float()).mean())
            q_weights = routing_info.get("q_weights")
            if q_weights is not None:
                routing_weights.append(q_weights.mean(dim=(1, 2)).detach())

            prev_x = x

            halt_step_logits = self.halt_head(x)
            all_halt_logits.append(halt_step_logits)

            if labels is not None:
                inter_x = self.ln_f(x)
                inter_logits = self.lm_head(inter_x)
                step_logits_list.append(inter_logits)

                shifted_logits = inter_logits[..., :-1, :].contiguous()
                shifted_labels = labels[..., 1:].contiguous()
                step_loss = F.cross_entropy(
                    shifted_logits.view(-1, self.config.vocab_size),
                    shifted_labels.view(-1),
                    reduction='none',
                    ignore_index=-100
                )
                step_loss = step_loss.view(batch_size, seq_length - 1)
                step_mask = shifted_labels.ne(-100).float()
                step_losses.append(
                    (step_loss * step_mask).sum(dim=-1) / (step_mask.sum(dim=-1) + 1e-9)
                )
            else:
                inter_x = self.ln_f(x)
                inter_logits = self.lm_head(inter_x)
                step_logits_list.append(inter_logits)

            if not self.training and self.config.use_act_inference and step > 0:
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
            all_step_losses_tensor = torch.stack(step_losses)
            best_losses, best_steps = all_step_losses_tensor.min(dim=0)
            main_loss = best_losses.mean()

            steps_range = torch.arange(all_step_losses_tensor.shape[0], device=self.device).unsqueeze(1)
            best_step_mask = (steps_range <= best_steps.unsqueeze(0))

            main_loss = (all_step_losses_tensor * best_step_mask.float()).sum() / (best_step_mask.float().sum() + 1e-9)

            halt_logits_tensor = torch.stack(all_halt_logits).squeeze(-1)
            halt_probs = torch.sigmoid(halt_logits_tensor.mean(dim=-1))

            target_halt = torch.zeros_like(halt_probs)
            for b in range(batch_size):
                target_halt[best_steps[b]:, b] = 1.0

            act_loss = F.binary_cross_entropy(halt_probs, target_halt)
            loss = main_loss + 0.1 * act_loss

            # === FARS 成本：标准熵最大化负载均衡 + normed FARS 权重 ===
            fars_loss = self._aggregate_fars_cost(routing_info)
            if fars_loss is not None:
                loss = loss + 0.1 * fars_loss

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return RecursiveDynSIHAOutput(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=None,
            attentions=None,
            exit_steps=best_steps + 1,
            all_step_losses=all_step_losses_tensor,
            best_step_mask=best_step_mask,
            halt_logits=torch.stack(all_halt_logits).squeeze(-1) if all_halt_logits else None,
            all_step_logits=torch.stack(step_logits_list) if step_logits_list else None,
            eff_k=(torch.stack(active_counts).mean() if active_counts else None),
            routing_weights=(torch.stack(routing_weights) if routing_weights else None)
        )

    def _aggregate_fars_cost(self, routing_info: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """
        实现基于 FARS 逻辑的负载均衡正则化：
        1. 熵最大化 (Entropy Maximization): 鼓励 P_avg 趋向均匀分布。
        2. FARS Cost: 使用归一化的二阶矩 (v_running) 作为惩罚权重。
        """
        total_fars_loss = 0.0
        count = 0
        
        # 访问 block 内部的 router 以获取 v_running
        routers = {
            "q": self.block.attn.q_router,
            "k": self.block.attn.k_router,
            "v": self.block.attn.v_router,
            "mlp": self.block.mlp.router
        }

        for head, router in routers.items():
            w = routing_info.get(f"{head}_weights")
            if w is not None:
                # 1. 计算平均路由分布 P_avg
                # w shape: [B, T, E] (attn) or [B, E] (mlp)
                p_avg = w.mean(dim=list(range(w.dim() - 1)))
                
                # 2. 熵最大化损失 (最小化 -H(P_avg))
                entropy_loss = (p_avg * torch.log(p_avg + 1e-9)).sum()
                
                # 3. FARS Cost (Fisher 曲率感知)
                # 使用 router 中维护的 v_running (二阶矩代理)
                v_norm = router.v_running / (router.v_running.max() + 1e-9)
                fars_penalty = (p_avg * v_norm).sum()
                
                total_fars_loss = total_fars_loss + (entropy_loss + fars_penalty)
                count += 1
                
        return total_fars_loss / count if count > 0 else None

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
