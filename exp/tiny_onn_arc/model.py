import math
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config

ExpertID = Tuple[str, int, int]

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, scores: torch.Tensor) -> torch.Tensor:
        return (scores > 0).to(scores.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output

@torch.jit.script
def _gating_logic(
    hidden_states: torch.Tensor, sim_matrix: torch.Tensor, gates: torch.Tensor, min_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return F.softmax(gated_logits_masked, dim=-1), logits, activation_mask

class GatingNetwork(nn.Module):
    def __init__(self, config: Config, max_experts: int, min_experts: int):
        super().__init__()
        self.config = config
        self.max_experts = max_experts
        self.min_experts = min_experts
        self.sim_matrix = nn.Parameter(torch.randn(config.hidden_size, max_experts))
        self.gates = nn.Parameter(torch.zeros(max_experts))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        routing_weights, logits, activation_mask = _gating_logic(
            hidden_states, self.sim_matrix, self.gates, self.min_experts
        )
        return routing_weights, {"logits": logits, "activation_mask": activation_mask}

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

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, dict]:
        B, T, C = hidden_states.shape
        routing_weights, gate_cache = self.gating_network(hidden_states)
        E = self.max_experts
        
        q_proj_w = torch.einsum("bte,ecd->btdc", routing_weights.view(B, T, E), self.q_proj)
        k_proj_w = torch.einsum("bte,ecd->btdc", routing_weights.view(B, T, E), self.k_proj)
        v_proj_w = torch.einsum("bte,ecd->btdc", routing_weights.view(B, T, E), self.v_proj)
        o_proj_w = torch.einsum("bte,edc->btdc", routing_weights.view(B, T, E), self.o_proj)

        q = torch.einsum("btc,btdc->btd", hidden_states, q_proj_w)
        k = torch.einsum("btc,btdc->btd", hidden_states, k_proj_w)
        v = torch.einsum("btc,btdc->btd", hidden_states, v_proj_w)
        
        attn_output = F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), attn_mask=attention_mask, is_causal=self.is_causal and T > 1).squeeze(1)
        final_output = torch.einsum("btd,btdc->btc", attn_output, o_proj_w)
        
        cache = {"final_output": final_output, "gate_cache": gate_cache, "B": B, "T": T}
        return final_output, cache

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

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        B, T, C = hidden_states.shape
        routing_weights, gate_cache = self.gating_network(hidden_states)
        
        intermediate_experts = F.gelu(torch.einsum("btc,eci->btei", hidden_states, self.w1))
        moe_experts_out = torch.einsum("btei,eic->btec", intermediate_experts, self.w2)
        final_output = torch.einsum("btec,bte->btc", moe_experts_out, routing_weights.view(B, T, self.max_experts))
        
        cache = {"final_output": final_output, "moe_experts_out": moe_experts_out, "gate_cache": gate_cache, "B": B, "T": T}
        return final_output, cache

class DynONNBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attn = DynSMHALayer(config, is_causal=False)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.moe = DynamicMoELayer(config)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        attn_output, attn_cache = self.attn(self.ln1(hidden_states))
        hidden_states = hidden_states + attn_output
        moe_output, moe_cache = self.moe(self.ln2(hidden_states))
        hidden_states = hidden_states + moe_output
        return hidden_states, {"attn": attn_cache, "moe": moe_cache}

class Encoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tok_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Parameter(torch.randn(1, config.MAX_GRID_SIZE * config.MAX_GRID_SIZE, config.hidden_size))
        
        if self.config.use_object_finder:
            self.obj_finder_ln = nn.LayerNorm(config.hidden_size)
            self.object_finder = DynSMHALayer(config, is_causal=False)
        
        self.layers = nn.ModuleList([DynONNBlock(config) for _ in range(config.num_hidden_layers)])
        self.final_ln = nn.LayerNorm(config.hidden_size)

    def forward(self, input_grid: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        B, H, W = input_grid.shape
        x = self.tok_embed(input_grid).view(B, H * W, -1)
        x = x + self.pos_embed[:, :(H * W), :]
        
        all_cache = {}
        if self.config.use_object_finder:
            obj_features, obj_cache = self.object_finder(self.obj_finder_ln(x))
            x = x + obj_features
            all_cache["object_finder"] = obj_cache

        for i, layer in enumerate(self.layers):
            x, block_cache = layer(x)
            all_cache[f"layer_{i}"] = block_cache
            
        return self.final_ln(x), all_cache

class Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.output_query = nn.Parameter(torch.randn(1, config.MAX_GRID_SIZE * config.MAX_GRID_SIZE, config.hidden_size))
        self.layers = nn.ModuleList([DynONNBlock(config) for _ in range(config.num_hidden_layers // 2)])
        self.final_ln = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, encoder_output: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        B = encoder_output.shape[0]
        query = self.output_query[:, :(target_h * target_w), :].expand(B, -1, -1)
        
        # Simple decoder for now: just a few blocks on top of the query
        x = query
        for layer in self.layers:
            x, _ = layer(x) # We ignore cache in the decoder for now
        
        x = self.final_ln(x)
        logits = self.lm_head(x)
        return logits.view(B, target_h, target_w, self.config.vocab_size)

class DynONNForArc(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        target_h, target_w = output_grid.shape[1], output_grid.shape[2]
        encoder_output, cache = self.encoder(input_grid)
        logits = self.decoder(encoder_output, target_h, target_w)
        return logits, cache