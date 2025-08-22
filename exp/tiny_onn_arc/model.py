import math
from typing import Any, Optional, Tuple, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .config import TinyOnnArcConfig

ExpertID = tuple[str, int, int]
KVCache: TypeAlias = Tuple[torch.Tensor, torch.Tensor]
PastKVCache: TypeAlias = Tuple[KVCache, ...]


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
        torch.tensor(-torch.inf, dtype=gated_logits.dtype, device=gated_logits.device),
    )
    return F.softmax(gated_logits_masked, dim=-1), logits, activation_mask, gated_logits


class GatingNetwork(nn.Module):
    def __init__(self, config: TinyOnnArcConfig, max_experts: int, min_experts: int):
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


class DynSMHALayer(nn.Module):
    def __init__(self, config: TinyOnnArcConfig, is_causal: bool = False):
        super().__init__()
        self.config = config
        self.max_experts = config.max_attention_experts
        self.is_causal = is_causal
        self.gating_network = GatingNetwork(config, self.max_experts, config.min_attention_experts)
        self.q_proj = nn.Parameter(torch.empty(self.max_experts, config.hidden_size, config.head_dim))
        self.k_proj = nn.Parameter(torch.empty(self.max_experts, config.hidden_size, config.head_dim))
        self.v_proj = nn.Parameter(torch.empty(self.max_experts, config.hidden_size, config.head_dim))
        self.o_proj = nn.Parameter(torch.empty(self.max_experts, config.head_dim, config.hidden_size))

        for i in range(self.max_experts):
            nn.init.xavier_uniform_(self.q_proj[i])
            nn.init.xavier_uniform_(self.k_proj[i])
            nn.init.xavier_uniform_(self.v_proj[i])
            nn.init.xavier_uniform_(self.o_proj[i])

    def forward_gating(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.gating_network(hidden_states)

    def forward_main(
        self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, gate_cache: dict[str, torch.Tensor], past_key_values: Optional[KVCache] = None, use_cache: bool = False
    ) -> Tuple[torch.Tensor, dict[str, Any], Optional[KVCache]]:
        B, T, C = hidden_states.shape
        flat_hidden_states = hidden_states.view(B * T, C)
        activation_mask = gate_cache["activation_mask"]

        q_acc = torch.zeros(B * T, self.config.head_dim, dtype=hidden_states.dtype, device=hidden_states.device)
        k_acc = torch.zeros_like(q_acc)
        v_acc = torch.zeros_like(q_acc)

        for i in range(self.max_experts):
            token_indices = torch.where(activation_mask[:, i] > 0)[0]
            if token_indices.shape[0] == 0:
                continue

            tokens_for_expert = flat_hidden_states[token_indices]
            weights_for_expert = routing_weights[token_indices, i].unsqueeze(1)

            q_acc.index_add_(0, token_indices, torch.matmul(tokens_for_expert, self.q_proj[i]) * weights_for_expert)
            k_acc.index_add_(0, token_indices, torch.matmul(tokens_for_expert, self.k_proj[i]) * weights_for_expert)
            v_acc.index_add_(0, token_indices, torch.matmul(tokens_for_expert, self.v_proj[i]) * weights_for_expert)

        q_agg = q_acc.view(B, T, -1)
        k_agg = k_acc.view(B, T, -1)
        v_agg = v_acc.view(B, T, -1)

        if past_key_values is not None:
            past_k, past_v = past_key_values
            k_agg = torch.cat([past_k, k_agg], dim=1)
            v_agg = torch.cat([past_v, v_agg], dim=1)
        
        present_key_values = (k_agg, v_agg) if use_cache else None

        q = rearrange(q_agg, "b t h -> b 1 t h")
        k = rearrange(k_agg, "b t h -> b 1 t h")
        v = rearrange(v_agg, "b t h -> b 1 t h")

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal and T > 1)
        attn_output = rearrange(attn_output, "b 1 t h -> b t h").view(B * T, -1)
        
        final_output = torch.zeros_like(flat_hidden_states)
        for i in range(self.max_experts):
            token_indices = torch.where(activation_mask[:, i] > 0)[0]
            if token_indices.shape[0] == 0:
                continue
            
            attn_output_for_expert = attn_output[token_indices]
            weights_for_expert = routing_weights[token_indices, i].unsqueeze(1)
            
            final_output.index_add_(0, token_indices, torch.matmul(attn_output_for_expert, self.o_proj[i]) * weights_for_expert)
        
        final_output = final_output.view(B, T, C)
        cache = {"final_output": final_output, "routing_weights": routing_weights, "B": B, "T": T}
        return final_output, cache, present_key_values

class DynamicMoELayer(nn.Module):
    def __init__(self, config: TinyOnnArcConfig):
        super().__init__()
        self.config = config
        self.max_experts = config.max_moe_experts
        self.gating_network = GatingNetwork(config, self.max_experts, config.min_moe_experts)
        self.w1 = nn.Parameter(torch.empty(self.max_experts, config.hidden_size, config.intermediate_size))
        self.w2 = nn.Parameter(torch.empty(self.max_experts, config.intermediate_size, config.hidden_size))
        for i in range(self.max_experts):
            nn.init.kaiming_uniform_(self.w1[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w2[i], a=math.sqrt(5))

    def forward_gating(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.gating_network(hidden_states)

    def forward_main(
        self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, gate_cache: dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, dict[str, Any], Optional[KVCache]]:
        B, T, C = hidden_states.shape
        flat_hidden_states = hidden_states.view(B * T, C)
        final_output = torch.zeros_like(flat_hidden_states)
        
        activation_mask = gate_cache["activation_mask"]

        for i in range(self.max_experts):
            token_indices = torch.where(activation_mask[:, i] > 0)[0]
            if token_indices.shape[0] == 0:
                continue
            
            tokens_for_expert = flat_hidden_states[token_indices]
            weights_for_expert = routing_weights[token_indices, i].unsqueeze(1)
            
            intermediate = F.gelu(torch.matmul(tokens_for_expert, self.w1[i]))
            output = torch.matmul(intermediate, self.w2[i])
            
            final_output.index_add_(0, token_indices, output * weights_for_expert)
            
        final_output = final_output.view(B, T, C)
        cache = {"final_output": final_output, "gate_cache": gate_cache, "routing_weights": routing_weights, "B": B, "T": T}
        return final_output, cache, None

class Block(nn.Module):
    def __init__(self, config: TinyOnnArcConfig, layer_index: int):
        super().__init__()
        self.layer_index = layer_index
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.smha_layer = DynSMHALayer(config, is_causal=True)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.moe_layer = DynamicMoELayer(config)

    def forward(
        self, hidden_states: torch.Tensor, past_key_values: Optional[KVCache] = None, use_cache: bool = False
    ) -> Tuple[torch.Tensor, dict[ExpertID, Any], Optional[KVCache]]:
        residual = hidden_states
        normed_hs_smha = self.ln1(hidden_states)
        smha_routing_weights, smha_gate_cache = self.smha_layer.forward_gating(normed_hs_smha)
        
        attn_output, smha_cache, present_key_values = self.smha_layer.forward_main(
            normed_hs_smha, smha_routing_weights, smha_gate_cache, past_key_values, use_cache=use_cache
        )
        smha_cache["gate_cache"] = smha_gate_cache
        hidden_states = residual + attn_output

        residual = hidden_states
        normed_hs_moe = self.ln2(hidden_states)
        moe_routing_weights, moe_gate_cache = self.moe_layer.forward_gating(normed_hs_moe)
        
        moe_output, moe_cache, _ = self.moe_layer.forward_main(normed_hs_moe, moe_routing_weights, moe_gate_cache)
        hidden_states = residual + moe_output

        block_cache = {("smha", self.layer_index, 0): smha_cache, ("moe", self.layer_index, 0): moe_cache}
        return hidden_states, block_cache, present_key_values


class AutoregressiveEmbedding(nn.Module):
    def __init__(self, config: TinyOnnArcConfig):
        super().__init__()
        self.tok_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings))

    def forward(self, input_ids: torch.Tensor, past_seq_len: int = 0) -> torch.Tensor:
        seq_len = input_ids.size(1)
        pos_ids = self.position_ids[past_seq_len : past_seq_len + seq_len]
        return self.tok_embed(input_ids) + self.pos_embed(pos_ids)


class TinyOnnModel(nn.Module):
    def __init__(self, config: TinyOnnArcConfig):
        super().__init__()
        self.config = config
        self.embeddings = AutoregressiveEmbedding(config)
        self.layers = nn.ModuleList([Block(config, i) for i in range(config.num_hidden_layers)])
        self.final_ln = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids: torch.Tensor, past_key_values: Optional[PastKVCache] = None, use_cache: bool = False) -> Tuple[torch.Tensor, dict[ExpertID, Any], Optional[PastKVCache]]:
        past_seq_len = past_key_values[0][0].shape[1] if past_key_values is not None else 0
        hidden_states = self.embeddings(input_ids, past_seq_len=past_seq_len)

        flat_forward_cache: dict[ExpertID, Any] = {}
        present_key_values_all = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            hidden_states, block_cache, present_key_values = layer(hidden_states, past_key_values=layer_past, use_cache=use_cache)
            flat_forward_cache.update(block_cache)
            if use_cache:
                present_key_values_all.append(present_key_values)
            
        hidden_states = self.final_ln(hidden_states)
        return hidden_states, flat_forward_cache, tuple(present_key_values_all) if use_cache else None


class TinyOnnForArcReconstruction(nn.Module):
    def __init__(self, config: TinyOnnArcConfig):
        super().__init__()
        self.config = config
        self.model = TinyOnnModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, past_key_values: Optional[PastKVCache] = None, use_cache: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, dict[ExpertID, Any], Optional[PastKVCache]]:
        hidden_states, flat_forward_cache, present_key_values = self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache)
        return self.lm_head(hidden_states), flat_forward_cache, present_key_values

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        self.eval()
        generated_ids = input_ids
        past_key_values = None
        for _ in range(max_new_tokens):
            input_token = generated_ids[:, -1].unsqueeze(-1) if past_key_values else generated_ids
            logits, _, past_key_values = self.forward(input_ids=input_token, past_key_values=past_key_values, use_cache=True)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
        self.train()
        return generated_ids
