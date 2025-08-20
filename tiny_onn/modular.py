from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import TinyOnnConfig

ExpertID = tuple[str, int, int]


@dataclass
class CausalLMOutputWithAux(CausalLMOutputWithPast):
    aux_outputs: dict[ExpertID, Any] | None = None


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, scores: torch.Tensor) -> torch.Tensor:
        return (scores > 0).to(scores.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


@torch.jit.script
def _gating_logic(
    hidden_states: torch.Tensor, sim_matrix: torch.Tensor, gates: torch.Tensor, max_experts: int, min_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b, t, c = hidden_states.shape
    flat_hidden_states = hidden_states.view(b * t, c)
    logits = torch.matmul(F.normalize(flat_hidden_states, dim=-1), F.normalize(sim_matrix, dim=0)) - torch.sigmoid(
        gates
    )
    gated_logits = F.relu(logits)
    activation_mask = STEFunction.apply(gated_logits)
    inactive_mask = torch.sum(activation_mask, dim=1) == 0
    if inactive_mask.any():
        inactive_logits = logits[inactive_mask]
        fallback_indices = torch.topk(inactive_logits, min_experts, dim=-1).indices
        inactive_b_indices = torch.where(inactive_mask)[0]
        activation_mask.index_put_(
            (inactive_b_indices.unsqueeze(1).expand(-1, min_experts), fallback_indices),
            torch.tensor(1.0, device=hidden_states.device, dtype=activation_mask.dtype),
        )
    gated_logits_masked = torch.where(
        activation_mask > 0,
        gated_logits,
        torch.tensor(-1e9, dtype=gated_logits.dtype, device=gated_logits.device),
    )
    return F.softmax(gated_logits_masked, dim=-1), logits, activation_mask, gated_logits


class GatingNetwork(nn.Module):
    def __init__(self, config: TinyOnnConfig, max_experts: int, min_experts: int):
        super().__init__()
        self.config = config
        self.max_experts = max_experts
        self.min_experts = min_experts
        self.sim_matrix = nn.Parameter(torch.randn(config.hidden_size, max_experts))
        self.gates = nn.Parameter(torch.zeros(max_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        active_expert_probs, logits, activation_mask, gated_logits = _gating_logic(
            hidden_states, self.sim_matrix, self.gates, self.max_experts, self.min_experts
        )
        return active_expert_probs, {"logits": logits, "activation_mask": activation_mask, "gated_logits": gated_logits}


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000, device: Any = None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len: int, device: Any, dtype: Any) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: Any, seq_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len is not None and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[position_ids]
    sin = sin[position_ids]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DynSMHALayer(nn.Module):
    def __init__(self, config: TinyOnnConfig):
        super().__init__()
        self.config = config
        self.max_experts = config.max_attention_experts

        self.compress_mlp = nn.Sequential(
            nn.Linear(config.hidden_size * config.block_size, config.hidden_size),
            nn.ReLU(),
        )

        self.global_router = GatingNetwork(config, self.max_experts, config.min_attention_experts)
        self.importance_threshold = nn.Parameter(torch.zeros(1))
        self.fine_grained_router = GatingNetwork(config, self.max_experts, config.min_attention_experts)

        self.q_proj = nn.Parameter(torch.empty(self.max_experts, config.hidden_size, config.head_dim))
        self.k_proj = nn.Parameter(torch.empty(self.max_experts, config.hidden_size, config.head_dim))
        self.v_proj = nn.Parameter(torch.empty(self.max_experts, config.hidden_size, config.head_dim))
        self.o_proj = nn.Parameter(torch.empty(self.max_experts, config.head_dim, config.hidden_size))
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.max_position_embeddings, base=config.rope_theta)

        for i in range(self.max_experts):
            nn.init.xavier_uniform_(self.q_proj[i])
            nn.init.xavier_uniform_(self.k_proj[i])
            nn.init.xavier_uniform_(self.v_proj[i])
            nn.init.xavier_uniform_(self.o_proj[i])

    def forward_gating(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, tuple, int]:
        B, T, C = hidden_states.shape
        W = self.config.block_size

        original_T = T
        if T % W != 0:
            pad_len = W - (T % W)
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))

        padded_T = hidden_states.shape[1]
        N = padded_T // W

        blocks = rearrange(hidden_states, "b (n w) c -> b n (w c)", w=W)
        compressed_blocks = self.compress_mlp(blocks)

        global_routing_weights, _ = self.global_router(compressed_blocks)
        importance_scores = global_routing_weights.max(dim=-1).values.view(B, N)
        block_activation_mask = STEFunction.apply(importance_scores - self.importance_threshold).bool()

        gate_caches, routing_weights_list = [], []
        for i in range(N):
            if not block_activation_mask[:, i].any():
                gate_caches.append(None)
                routing_weights_list.append(None)
                continue

            active_batch_indices = torch.where(block_activation_mask[:, i])[0]
            block_hs = hidden_states[active_batch_indices, i * W : (i + 1) * W, :]

            routing_weights, gate_cache = self.fine_grained_router(block_hs)
            gate_caches.append(gate_cache)
            routing_weights_list.append(routing_weights)

        return block_activation_mask, (gate_caches, routing_weights_list), original_T

    def forward_main(self, hidden_states: torch.Tensor, position_ids: torch.Tensor, block_activation_mask: torch.Tensor, gating_results: tuple, original_T: int) -> tuple[torch.Tensor, dict[str, Any]]:
        B, T, C = hidden_states.shape
        W = self.config.block_size

        padded_T = T
        if T % W != 0:
            pad_len = W - (T % W)
            padded_T = T + pad_len
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
            position_ids = F.pad(position_ids, (0, pad_len), value=0)

        N = padded_T // W
        gate_caches, routing_weights_list = gating_results

        final_output = torch.zeros_like(hidden_states)

        # Buffers to store per-token routing info for the entire sequence
        full_logits = torch.zeros(B, padded_T, self.max_experts, device=hidden_states.device, dtype=hidden_states.dtype)
        full_routing_weights = torch.zeros(B, padded_T, self.max_experts, device=hidden_states.device, dtype=hidden_states.dtype)
        full_activation_mask = torch.zeros(B, padded_T, self.max_experts, device=hidden_states.device, dtype=torch.bool)

        num_active_tokens = 0

        for i in range(N):
            if not block_activation_mask[:, i].any():
                continue

            active_batch_indices = torch.where(block_activation_mask[:, i])[0]

            # Since we pad hidden_states at the start of forward_main, we select all batch indices here
            block_hs = hidden_states[:, i * W : (i + 1) * W, :]
            block_pos_ids = position_ids[:, i * W : (i + 1) * W]

            routing_weights = routing_weights_list[i]
            gate_cache = gate_caches[i]

            if routing_weights is None or gate_cache is None: continue

            # We need to map the partial batch back to the full batch for scatter operations
            # This is complex. Let's rethink. The gating was on a partial batch.
            # The main computation must be on a full tensor to be checkpoint-safe.

            # The issue is `gating_results` are for partial batches. Let's reconstruct full batch info.
            temp_logits = torch.zeros(B, W, self.max_experts, device=hidden_states.device, dtype=hidden_states.dtype)
            temp_routing_weights = torch.zeros(B, W, self.max_experts, device=hidden_states.device, dtype=hidden_states.dtype)
            temp_activation_mask = torch.zeros(B, W, self.max_experts, device=hidden_states.device, dtype=torch.bool)

            if gate_cache is not None and routing_weights is not None:
                temp_logits[active_batch_indices] = gate_cache["logits"].view(-1, W, self.max_experts)
                temp_routing_weights[active_batch_indices] = routing_weights.view(-1, W, self.max_experts)
                temp_activation_mask[active_batch_indices] = (gate_cache["activation_mask"] > 0).view(-1, W, self.max_experts)


            block_output, fine_cache = self.forward_fine(block_hs, temp_routing_weights.view(-1, self.max_experts), {"activation_mask": temp_activation_mask.view(-1, self.max_experts)}, block_pos_ids)
            final_output[:, i * W : (i + 1) * W, :] = block_output

            full_logits[:, i * W : (i + 1) * W, :] = temp_logits
            full_routing_weights[:, i * W : (i + 1) * W, :] = temp_routing_weights
            full_activation_mask[:, i * W : (i + 1) * W, :] = temp_activation_mask
            num_active_tokens += torch.sum(temp_activation_mask)


        if final_output.shape[1] != original_T:
            final_output = final_output[:, :original_T, :]
            full_logits = full_logits[:,:original_T,:]
            full_routing_weights = full_routing_weights[:,:original_T,:]
            full_activation_mask = full_activation_mask[:,:original_T,:]

        cache = {
            "final_output": final_output,
            "gate_cache": {"logits": full_logits.view(-1, self.max_experts), "activation_mask": full_activation_mask.view(-1, self.max_experts)},
            "routing_weights": full_routing_weights.view(-1, self.max_experts),
            "num_active_tokens": num_active_tokens,
            "B": B, "T": original_T,
        }
        return final_output, cache

    def forward_fine(
        self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, gate_cache: dict[str, torch.Tensor], position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        B, T, C = hidden_states.shape
        flat_hidden_states = hidden_states.view(B * T, C)
        activation_mask = gate_cache["activation_mask"]

        token_indices, expert_indices = torch.where(activation_mask > 0)

        if token_indices.numel() == 0:
            return torch.zeros_like(hidden_states), {}

        selected_hidden_states = flat_hidden_states[token_indices]
        weights_for_active = routing_weights[token_indices, expert_indices].unsqueeze(-1)

        q_parts = torch.bmm(selected_hidden_states.unsqueeze(1), self.q_proj[expert_indices]).squeeze(1)
        k_parts = torch.bmm(selected_hidden_states.unsqueeze(1), self.k_proj[expert_indices]).squeeze(1)
        v_parts = torch.bmm(selected_hidden_states.unsqueeze(1), self.v_proj[expert_indices]).squeeze(1)

        weighted_q_parts = q_parts * weights_for_active
        weighted_k_parts = k_parts * weights_for_active
        weighted_v_parts = v_parts * weights_for_active

        q_flat = torch.zeros(B * T, self.config.head_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        k_flat = torch.zeros(B * T, self.config.head_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        v_flat = torch.zeros(B * T, self.config.head_dim, device=hidden_states.device, dtype=hidden_states.dtype)

        q_flat.scatter_add_(0, token_indices.unsqueeze(-1).expand_as(weighted_q_parts), weighted_q_parts)
        k_flat.scatter_add_(0, token_indices.unsqueeze(-1).expand_as(weighted_k_parts), weighted_k_parts)
        v_flat.scatter_add_(0, token_indices.unsqueeze(-1).expand_as(weighted_v_parts), weighted_v_parts)

        q = q_flat.view(B, T, -1)
        k = k_flat.view(B, T, -1)
        v = v_flat.view(B, T, -1)

        cos, sin = self.rotary_emb.cos_cached, self.rotary_emb.sin_cached
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids.view(-1))

        attn_output = F.scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), is_causal=True
        ).squeeze(1)

        attn_output_flat = attn_output.view(B * T, -1)
        selected_attn_output = attn_output_flat[token_indices]

        o_parts = torch.bmm(selected_attn_output.unsqueeze(1), self.o_proj[expert_indices]).squeeze(1)
        weighted_o_parts = o_parts * weights_for_active

        final_output_flat = torch.zeros_like(flat_hidden_states)
        final_output_flat.scatter_add_(0, token_indices.unsqueeze(-1).expand_as(weighted_o_parts), weighted_o_parts)
        final_output = final_output_flat.view(B, T, C)

        return final_output, {}


class DynamicMoELayer(nn.Module):
    def __init__(self, config: TinyOnnConfig):
        super().__init__()
        self.config = config
        self.max_experts = config.max_moe_experts
        self.gating_network = GatingNetwork(config, self.max_experts, config.min_moe_experts)
        self.w1 = nn.Parameter(torch.empty(self.max_experts, config.hidden_size, config.intermediate_size))
        self.w2 = nn.Parameter(torch.empty(self.max_experts, config.intermediate_size, config.hidden_size))
        for i in range(self.max_experts):
            nn.init.kaiming_uniform_(self.w1[i], a=5**0.5)
            nn.init.kaiming_uniform_(self.w2[i], a=5**0.5)

    def forward_gating(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.gating_network(hidden_states)

    def forward_main(
        self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, gate_cache: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        B, T, C = hidden_states.shape
        flat_hidden_states = hidden_states.view(B * T, C)
        activation_mask = gate_cache["activation_mask"]

        token_indices, expert_indices = torch.where(activation_mask > 0)

        if token_indices.numel() == 0:
            final_output = torch.zeros_like(hidden_states)
        else:
            selected_hidden_states = flat_hidden_states[token_indices]

            selected_w1 = self.w1[expert_indices]
            selected_w2 = self.w2[expert_indices]

            intermediate_activated = F.gelu(torch.bmm(selected_hidden_states.unsqueeze(1), selected_w1))
            output_flat = torch.bmm(intermediate_activated, selected_w2).squeeze(1)

            weights_for_active = routing_weights[token_indices, expert_indices]
            weighted_output = output_flat * weights_for_active.unsqueeze(-1)

            final_output_flat = torch.zeros_like(flat_hidden_states)
            final_output_flat.scatter_add_(0, token_indices.unsqueeze(-1).expand_as(weighted_output), weighted_output)
            final_output = final_output_flat.view(B, T, C)

        cache = {
            "gate_cache": gate_cache,
            "routing_weights": routing_weights,
            "final_output": final_output,
            "num_active_tokens": torch.sum(activation_mask).item(),
            "B": B,
            "T": T,
        }
        return final_output, cache

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        routing_weights, gate_cache = self.forward_gating(hidden_states)
        return self.forward_main(hidden_states, routing_weights, gate_cache)


class Block(nn.Module):
    def __init__(self, config: TinyOnnConfig, layer_index: int):
        super().__init__()
        self.layer_index = layer_index
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.smha_layer = DynSMHALayer(config)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.moe_layer = DynamicMoELayer(config)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, dict[ExpertID, Any]]:
        residual = hidden_states
        normed_hs_smha = self.ln1(hidden_states)

        block_activation_mask, gating_results, original_T = self.smha_layer.forward_gating(normed_hs_smha)

        def smha_checkpointed_fn(hs_norm: torch.Tensor, pos_ids: torch.Tensor, mask: torch.Tensor, results: tuple, og_T: int) -> tuple[torch.Tensor, dict[str, Any]]:
            return self.smha_layer.forward_main(hs_norm, pos_ids, mask, results, og_T)

        attn_output, smha_cache = checkpoint(smha_checkpointed_fn, normed_hs_smha, position_ids, block_activation_mask, gating_results, original_T, use_reentrant=False)
        smha_cache["normed_hs"] = normed_hs_smha
        hidden_states = residual + attn_output

        residual = hidden_states
        normed_hs_moe = self.ln2(hidden_states)
        moe_routing_weights, moe_gate_cache = self.moe_layer.forward_gating(normed_hs_moe)

        def moe_checkpointed_fn(hs_norm: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
            return self.moe_layer.forward_main(hs_norm, moe_routing_weights, moe_gate_cache)

        moe_output, moe_cache = checkpoint(moe_checkpointed_fn, normed_hs_moe, use_reentrant=False)
        moe_cache["normed_hs"] = normed_hs_moe
        hidden_states = residual + moe_output

        block_cache = {
            ("smha", self.layer_index, 0): smha_cache,
            ("moe", self.layer_index, 0): moe_cache,
        }
        return hidden_states, block_cache


class TinyOnnPreTrainedModel(PreTrainedModel):
    config_class = TinyOnnConfig
    base_model_prefix = "model"


class TinyOnnModel(TinyOnnPreTrainedModel):
    def __init__(self, config: TinyOnnConfig):
        super().__init__(config)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.embedding_size)
        self.embedding_proj_in = nn.Linear(config.embedding_size, config.hidden_size, bias=config.bias)
        self.layers = nn.ModuleList([Block(config, i) for i in range(config.num_hidden_layers)])
        self.final_ln = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, dict[ExpertID, Any]]:
        hidden_states = self.tok_embeddings(input_ids)
        hidden_states = self.embedding_proj_in(hidden_states)

        B, T, C = hidden_states.shape
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)

        flat_forward_cache: dict[ExpertID, Any] = {}
        for layer in self.layers:
            hidden_states, block_cache = layer(hidden_states, position_ids)
            flat_forward_cache.update(block_cache)
        hidden_states = self.final_ln(hidden_states)
        return hidden_states, flat_forward_cache


class TinyOnnForCausalLM(TinyOnnPreTrainedModel):
    def __init__(self, config: TinyOnnConfig):
        super().__init__(config)
        self.model = TinyOnnModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.tok_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Module) -> None:
        self.model.tok_embeddings = new_embeddings

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> CausalLMOutputWithAux:
        hidden_states, flat_forward_cache = self.model(input_ids, **kwargs)
        final_logits = self.lm_head(hidden_states)
        return CausalLMOutputWithAux(logits=final_logits, aux_outputs=flat_forward_cache)

    @torch.no_grad()
    def generate(
        self, input_ids: torch.Tensor, max_new_tokens: int = 30, eos_token_id: int | None = None
    ) -> torch.Tensor:
        generated_ids = input_ids
        for _ in range(max_new_tokens):
            outputs = self.forward(input_ids=generated_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        return generated_ids
