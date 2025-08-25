import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config

ExpertCache = dict[str, Any]

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, scores: torch.Tensor) -> torch.Tensor:
        return (scores > 0).to(scores.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output

@torch.jit.script
def _gating_logic(
    hidden_states: torch.Tensor, sim_matrix: torch.Tensor, gates: torch.Tensor, fallback_k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, t, c = hidden_states.shape
    flat_hidden_states = hidden_states.view(b * t, c)
    logits = torch.matmul(F.normalize(flat_hidden_states, dim=-1), F.normalize(sim_matrix, dim=0)) - gates
    gated_logits = F.relu(logits)
    activation_mask = STEFunction.apply(gated_logits)

    inactive_mask = torch.sum(activation_mask, dim=1) == 0
    if inactive_mask.any():
        num_inactive = inactive_mask.sum()
        fallback_values, fallback_indices = torch.topk(logits[inactive_mask], k=fallback_k, dim=1)
        
        fallback_mask = torch.zeros(num_inactive, logits.shape[1], device=logits.device, dtype=activation_mask.dtype)
        fallback_mask.scatter_(1, fallback_indices, 1)
        
        activation_mask[inactive_mask] = fallback_mask

    gated_logits_masked = torch.where(
        activation_mask > 0, gated_logits, torch.tensor(-torch.inf, dtype=gated_logits.dtype, device=gated_logits.device)
    )
    return F.softmax(gated_logits_masked, dim=-1), logits, activation_mask

class GatingNetwork(nn.Module):
    def __init__(self, config: Config, max_experts: int):
        super().__init__()
        self.max_experts = max_experts
        self.sim_matrix = nn.Parameter(torch.randn(config.hidden_size, max_experts))
        self.gates = nn.Parameter(torch.zeros(max_experts))
        self.register_buffer("activation_counts", torch.zeros(max_experts, dtype=torch.long))

    def forward(self, hidden_states: torch.Tensor, fallback_k: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        routing_weights, logits, activation_mask = _gating_logic(hidden_states, self.sim_matrix, self.gates, fallback_k)
        if self.training:
            self.activation_counts += activation_mask.sum(dim=0).long()
        return routing_weights, {"logits": logits, "activation_mask": activation_mask, "gating_net_ref": self}

    def reset_counts(self):
        self.activation_counts.zero_()

class DynSMHALayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.max_experts = config.max_attention_experts
        self.gating_network = GatingNetwork(config, self.max_experts)
        self.q_proj = nn.Parameter(torch.empty(self.max_experts, config.hidden_size, config.head_dim))
        self.k_proj = nn.Parameter(torch.empty(self.max_experts, config.hidden_size, config.head_dim))
        self.v_proj = nn.Parameter(torch.empty(self.max_experts, config.hidden_size, config.head_dim))
        self.o_proj = nn.Parameter(torch.empty(self.max_experts, config.head_dim, config.hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.max_experts):
            nn.init.xavier_uniform_(self.q_proj[i])
            nn.init.xavier_uniform_(self.k_proj[i])
            nn.init.xavier_uniform_(self.v_proj[i])
            nn.init.xavier_uniform_(self.o_proj[i])

    def regenerate_dead_experts(self) -> int:
        dead_experts = torch.where(self.gating_network.activation_counts == 0)[0]
        if len(dead_experts) > 0:
            with torch.no_grad():
                for i in dead_experts:
                    nn.init.xavier_uniform_(self.q_proj[i])
                    nn.init.xavier_uniform_(self.k_proj[i])
                    nn.init.xavier_uniform_(self.v_proj[i])
                    nn.init.xavier_uniform_(self.o_proj[i])
                    self.gating_network.sim_matrix.data[:, i].normal_(0, 0.02)
                    self.gating_network.gates.data[i].zero_()
        self.gating_network.reset_counts()
        return len(dead_experts)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, ExpertCache]:
        B, T, C = hidden_states.shape
        is_cross_attention = encoder_hidden_states is not None
        key_value_states = encoder_hidden_states if is_cross_attention else hidden_states
        
        q_routing_weights, q_gate_cache = self.gating_network(hidden_states, self.config.fallback_attention_k)
        
        if is_cross_attention:
            kv_routing_weights, kv_gate_cache = self.gating_network(key_value_states, self.config.fallback_attention_k)
            gate_cache = {**q_gate_cache, **kv_gate_cache}
        else:
            kv_routing_weights = q_routing_weights
            gate_cache = q_gate_cache

        q_proj_w = torch.einsum("bte,ecd->btdc", q_routing_weights.view(B, T, self.max_experts), self.q_proj)
        k_proj_w = torch.einsum("bte,ecd->btdc", kv_routing_weights.view(key_value_states.shape[0], key_value_states.shape[1], self.max_experts), self.k_proj)
        v_proj_w = torch.einsum("bte,ecd->btdc", kv_routing_weights.view(key_value_states.shape[0], key_value_states.shape[1], self.max_experts), self.v_proj)
        o_proj_w = torch.einsum("bte,edc->btdc", q_routing_weights.view(B, T, self.max_experts), self.o_proj)

        q = torch.einsum("btc,btdc->btd", hidden_states, q_proj_w)
        k = torch.einsum("btc,btdc->btd", key_value_states, k_proj_w)
        v = torch.einsum("btc,btdc->btd", key_value_states, v_proj_w)

        attn_output = F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), attn_mask=attention_mask).squeeze(1)
        final_output = torch.einsum("btd,btdc->btc", attn_output, o_proj_w)

        return final_output, {**gate_cache, "type": "smha", "expert_outputs": final_output}

class DynamicMoELayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.max_experts = config.max_moe_experts
        self.gating_network = GatingNetwork(config, self.max_experts)
        self.w1 = nn.Parameter(torch.empty(self.max_experts, config.hidden_size, config.intermediate_size))
        self.w2 = nn.Parameter(torch.empty(self.max_experts, config.intermediate_size, config.hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.max_experts):
            nn.init.kaiming_uniform_(self.w1[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w2[i], a=math.sqrt(5))

    def regenerate_dead_experts(self) -> int:
        dead_experts = torch.where(self.gating_network.activation_counts == 0)[0]
        if len(dead_experts) > 0:
            with torch.no_grad():
                for i in dead_experts:
                    nn.init.kaiming_uniform_(self.w1[i], a=math.sqrt(5))
                    nn.init.kaiming_uniform_(self.w2[i], a=math.sqrt(5))
                    self.gating_network.sim_matrix.data[:, i].normal_(0, 0.02)
                    self.gating_network.gates.data[i].zero_()
        self.gating_network.reset_counts()
        return len(dead_experts)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, ExpertCache]:
        B, T, C = hidden_states.shape
        routing_weights, gate_cache = self.gating_network(hidden_states, self.config.fallback_moe_k)

        intermediate_experts = F.gelu(torch.einsum("btc,eci->btei", hidden_states, self.w1))
        moe_experts_out = torch.einsum("btei,eic->btec", intermediate_experts, self.w2)
        final_output = torch.einsum("btec,bte->btc", moe_experts_out, routing_weights.view(B, T, self.max_experts))

        return final_output, {**gate_cache, "type": "moe", "expert_outputs": moe_experts_out}

class DynONNBlock(nn.Module):
    def __init__(self, config: Config, is_decoder: bool = False):
        super().__init__()
        self.is_decoder = is_decoder
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attn = DynSMHALayer(config)
        
        if self.is_decoder:
            self.ln_cross = nn.LayerNorm(config.hidden_size)
            self.cross_attention = DynSMHALayer(config)

        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.moe = DynamicMoELayer(config)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor | None = None) -> tuple[torch.Tensor, list[ExpertCache]]:
        attn_output, attn_cache = self.attn(self.ln1(hidden_states))
        hidden_states = hidden_states + attn_output
        
        cache = [attn_cache]
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attn_output, cross_attn_cache = self.cross_attention(self.ln_cross(hidden_states), encoder_hidden_states)
            hidden_states = hidden_states + cross_attn_output
            cache.append(cross_attn_cache)

        moe_output, moe_cache = self.moe(self.ln2(hidden_states))
        hidden_states = hidden_states + moe_output
        cache.append(moe_cache)
        
        return hidden_states, cache

class Encoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tok_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Parameter(torch.randn(1, config.MAX_GRID_SIZE * config.MAX_GRID_SIZE, config.hidden_size))

        if self.config.use_object_finder:
            self.obj_finder_ln = nn.LayerNorm(config.hidden_size)
            self.object_finder = DynSMHALayer(config)

        self.layers = nn.ModuleList([DynONNBlock(config) for _ in range(config.num_hidden_layers)])
        self.final_ln = nn.LayerNorm(config.hidden_size)

    def forward(self, input_grid: torch.Tensor) -> tuple[torch.Tensor, list[ExpertCache]]:
        B, H, W = input_grid.shape
        x = self.tok_embed(input_grid).view(B, H * W, -1)
        x = x + self.pos_embed[:, :(H * W), :]

        flat_cache = []
        if self.config.use_object_finder:
            obj_features, obj_cache = self.object_finder(self.obj_finder_ln(x))
            x = x + obj_features
            flat_cache.append(obj_cache)

        for layer in self.layers:
            x, block_cache = layer(x)
            flat_cache.extend(block_cache)

        return self.final_ln(x), flat_cache

class Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.output_query = nn.Parameter(torch.randn(1, config.MAX_GRID_SIZE * config.MAX_GRID_SIZE, config.hidden_size))
        self.layers = nn.ModuleList([DynONNBlock(config, is_decoder=True) for _ in range(config.num_hidden_layers // 2)])
        self.final_ln = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, encoder_output: torch.Tensor, target_h: int, target_w: int) -> tuple[torch.Tensor, list[ExpertCache]]:
        B = encoder_output.shape[0]
        query = self.output_query[:, :(target_h * target_w), :].expand(B, -1, -1)

        flat_cache = []
        x = query
        for layer in self.layers:
            x, block_cache = layer(x, encoder_output)
            flat_cache.extend(block_cache)

        x = self.final_ln(x)
        logits = self.lm_head(x)
        return logits.view(B, target_h, target_w, self.config.vocab_size), flat_cache

class DynONNForArc(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(
        self,
        input_grid: torch.Tensor,
        output_grid: torch.Tensor | None = None,
        target_h: int | None = None,
        target_w: int | None = None,
    ) -> tuple[torch.Tensor, list[ExpertCache]]:
        if output_grid is not None:
            target_h, target_w = output_grid.shape[1], output_grid.shape[2]
        elif target_h is None or target_w is None:
            raise ValueError("Must provide either output_grid or target_h/target_w")

        encoder_output, encoder_cache = self.encoder(input_grid)
        logits, decoder_cache = self.decoder(encoder_output, target_h, target_w)
        return logits, encoder_cache + decoder_cache
