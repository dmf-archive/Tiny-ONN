import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config

ExpertID = tuple[str, int, int]
type KVCache = tuple[torch.Tensor, torch.Tensor]
type PastKVCache = tuple[KVCache, ...]

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
    logits = torch.matmul(F.normalize(flat_hidden_states, dim=-1), F.normalize(sim_matrix, dim=0)) - torch.sigmoid(gates)
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
        activation_mask > 0, gated_logits, torch.tensor(-torch.inf, dtype=gated_logits.dtype, device=gated_logits.device)
    )
    return F.softmax(gated_logits_masked, dim=-1), logits, activation_mask, gated_logits

class GatingNetwork(nn.Module):
    def __init__(self, config: Config, max_experts: int, min_experts: int):
        super().__init__()
        self.config = config
        self.max_experts = max_experts
        self.min_experts = min_experts
        self.sim_matrix = nn.Parameter(torch.randn(config.hidden_size, max_experts))
        self.gates = nn.Parameter(torch.zeros(max_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        routing_weights, logits, activation_mask, gated_logits = _gating_logic(
            hidden_states, self.sim_matrix, self.gates, self.max_experts, self.min_experts
        )
        return routing_weights, {"logits": logits, "activation_mask": activation_mask, "gated_logits": gated_logits, "gating_net": self}

class DynSMHALayer(nn.Module):
    def __init__(self, config: Config, is_causal: bool = False):
        super().__init__()
        self.config = config
        self.is_causal = is_causal
        self.max_experts = config.max_attention_experts
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
        self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, attention_mask: torch.Tensor | None = None, past_key_values: KVCache | None = None, use_cache: bool = False
    ) -> tuple[torch.Tensor, dict[str, Any], KVCache | None]:
        B, T, C = hidden_states.shape
        E, D_head = self.max_experts, self.config.head_dim

        q_proj_w = torch.einsum("bte,ecd->btdc", routing_weights.view(B, T, E), self.q_proj)
        k_proj_w = torch.einsum("bte,ecd->btdc", routing_weights.view(B, T, E), self.k_proj)
        v_proj_w = torch.einsum("bte,ecd->btdc", routing_weights.view(B, T, E), self.v_proj)
        o_proj_w = torch.einsum("bte,edc->btdc", routing_weights.view(B, T, E), self.o_proj)

        q = torch.einsum("btc,btdc->btd", hidden_states, q_proj_w)
        k = torch.einsum("btc,btdc->btd", hidden_states, k_proj_w)
        v = torch.einsum("btc,btdc->btd", hidden_states, v_proj_w)

        if past_key_values is not None:
            k = torch.cat([past_key_values[0], k], dim=1)
            v = torch.cat([past_key_values[1], v], dim=1)

        present_key_values = (k, v) if use_cache else None

        attn_output = F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), attn_mask=attention_mask, is_causal=self.is_causal and T > 1 and attention_mask is None).squeeze(1)

        final_output = torch.einsum("btd,btdc->btc", attn_output, o_proj_w)

        cache = {"final_output": final_output, "routing_weights": routing_weights, "B": B, "T": T}
        return final_output, cache, present_key_values

class DynamicMoELayer(nn.Module):
    def __init__(self, config: Config):
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

    def forward_main(self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, gate_cache: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        B, T, C = hidden_states.shape

        w1_w = torch.einsum("bte,eci->btci", routing_weights.view(B, T, self.max_experts), self.w1)
        w2_w = torch.einsum("bte,eic->btic", routing_weights.view(B, T, self.max_experts), self.w2)

        intermediate = F.gelu(torch.einsum("btc,btci->bti", hidden_states, w1_w))
        final_output = torch.einsum("bti,btic->btc", intermediate, w2_w)

        cache = {"final_output": final_output, "gate_cache": gate_cache, "routing_weights": routing_weights, "B": B, "T": T}
        return final_output, cache

class Block(nn.Module):
    def __init__(self, config: Config, layer_index: int):
        super().__init__()
        self.layer_index = layer_index
        self.ln = nn.LayerNorm(config.hidden_size)
        self.moe_layer = DynamicMoELayer(config)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[ExpertID, Any]]:
        residual = hidden_states
        normed_hs = self.ln(hidden_states)
        
        moe_routing_weights, moe_gate_cache = self.moe_layer.forward_gating(normed_hs)
        moe_output, moe_cache = self.moe_layer.forward_main(normed_hs, moe_routing_weights, moe_gate_cache)
        
        hidden_states = residual + moe_output
        
        # No more SMHA in the block, only MoE
        block_cache = {("moe", self.layer_index, 0): moe_cache}
        return hidden_states, block_cache

class AutoregressiveEmbedding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.tok_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings))

    def forward(self, input_ids: torch.Tensor, past_seq_len: int = 0) -> torch.Tensor:
        seq_len = input_ids.size(1)
        pos_ids = self.position_ids[past_seq_len : past_seq_len + seq_len]
        return self.tok_embed(input_ids) + self.pos_embed(pos_ids)

class TinyOnnModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embeddings = AutoregressiveEmbedding(config)
        
        if self.config.use_object_finder:
            self.obj_finder_ln = nn.LayerNorm(config.hidden_size)
            self.object_finder_layer = DynSMHALayer(config, is_causal=False)

        self.layers = nn.ModuleList([Block(config, i) for i in range(config.num_hidden_layers)])
        self.final_ln = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, use_cache: bool = False) -> tuple[torch.Tensor, dict[ExpertID, Any]]:
        hidden_states = self.embeddings(input_ids, past_seq_len=0)
        flat_forward_cache: dict[ExpertID, Any] = {}

        if self.config.use_object_finder:
            B, T, C = hidden_states.shape
            normed_hs = self.obj_finder_ln(hidden_states)
            routing_weights, gate_cache = self.object_finder_layer.forward_gating(normed_hs)

            E, D_head = self.object_finder_layer.max_experts, self.config.head_dim
            q_proj_w = torch.einsum("bte,ecd->btdc", routing_weights.view(B, T, E), self.object_finder_layer.q_proj)
            k_proj_w = torch.einsum("bte,ecd->btdc", routing_weights.view(B, T, E), self.object_finder_layer.k_proj)

            Q = torch.einsum("btc,btdc->btd", hidden_states, q_proj_w)
            K = torch.einsum("btc,btdc->btd", hidden_states, k_proj_w)

            affinity_scores = torch.einsum("btd,bsd->bts", Q, K) / math.sqrt(D_head)
            affinity_matrix = F.softmax(affinity_scores, dim=-1)

            object_ids = torch.argmax(affinity_matrix, dim=-1)
            object_prototypes = torch.gather(hidden_states, 1, object_ids.unsqueeze(-1).expand_as(hidden_states))
            hidden_states = hidden_states + object_prototypes
            
            # Use layer_idx = -1 to denote the Object Finder Layer for logging
            flat_forward_cache[("smha", -1, 0)] = {"gate_cache": gate_cache, "B": B, "T": T}

        for i, layer in enumerate(self.layers):
            hidden_states, block_cache = layer(hidden_states)
            flat_forward_cache.update(block_cache)

        hidden_states = self.final_ln(hidden_states)
        return hidden_states, flat_forward_cache

class TinyOnnForArcReconstruction(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model = TinyOnnModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: Any) -> tuple[torch.Tensor, dict[ExpertID, Any]]:
        hidden_states, flat_forward_cache = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return self.lm_head(hidden_states), flat_forward_cache

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        self.eval()
        # For simplicity, generation in the new model does not support KV caching
        # as blocks are no longer causal.
        generated_ids = input_ids
        for _ in range(max_new_tokens):
            logits, _ = self.forward(
                input_ids=generated_ids, attention_mask=attention_mask
            )
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            attention_mask = F.pad(attention_mask, (0, 1), "constant", 1)
        self.train()
        return generated_ids
