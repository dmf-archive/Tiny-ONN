import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple

from .config import DynSMHAConfig, DTYPE

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        return (scores > 0).to(scores.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class SingleHeadAttention(nn.Module):
    def __init__(self, config: DynSMHAConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.head_dim, bias=config.bias, dtype=DTYPE)
        self.k_proj = nn.Linear(config.hidden_size, config.head_dim, bias=config.bias, dtype=DTYPE)
        self.v_proj = nn.Linear(config.hidden_size, config.head_dim, bias=config.bias, dtype=DTYPE)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        attn_output = F.scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
        ).squeeze(1)
        
        return attn_output, q, k

class GatingNetwork(nn.Module):
    def __init__(self, config: DynSMHAConfig):
        super().__init__()
        self.config = config
        self.gates = nn.Parameter(torch.zeros(config.num_experts, dtype=DTYPE))

    def forward(self, all_q: torch.Tensor, all_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_scores = torch.einsum('bmtd,bmjd->bmtj', all_q, all_k) / (self.config.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-9), dim=-1)
        mean_entropy = entropy.mean(dim=-1)
        
        affinity_score = -mean_entropy
        
        affinity_score = (affinity_score - affinity_score.mean(dim=-1, keepdim=True)) / (affinity_score.std(dim=-1, keepdim=True) + 1e-9)
        
        pre_activation_logits = affinity_score - torch.sigmoid(self.gates)
        
        activation_mask = STEFunction.apply(pre_activation_logits)
        
        num_active = torch.sum(activation_mask, dim=1)
        inactive_mask = num_active == 0
        
        fallback_count = 0
        if inactive_mask.any():
            fallback_count = inactive_mask.sum().item()
            fallback_indices = torch.topk(affinity_score[inactive_mask], self.config.k_fallback, dim=1).indices
            activation_mask[inactive_mask] = activation_mask[inactive_mask].scatter(1, fallback_indices, 1.0)
            
        return pre_activation_logits, activation_mask, torch.tensor(fallback_count, device=activation_mask.device)

class DynSMHALayer(nn.Module):
    def __init__(self, config: DynSMHAConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([SingleHeadAttention(config) for _ in range(config.num_experts)])
        self.gating_network = GatingNetwork(config)
        self.o_weights = nn.Parameter(torch.randn(config.num_experts, config.head_dim, config.hidden_size, dtype=DTYPE))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        expert_outputs = [expert(hidden_states) for expert in self.experts]
        all_sha_outputs = torch.stack([out[0] for out in expert_outputs], dim=2)
        all_q = torch.stack([out[1] for out in expert_outputs], dim=1)
        all_k = torch.stack([out[2] for out in expert_outputs], dim=1)

        pre_activation_logits, activation_mask, fallback_count = self.gating_network(all_q, all_k)
        
        num_active = torch.sum(activation_mask, dim=1, keepdim=True)
        normalized_mask = activation_mask / torch.clamp(num_active, min=1)
        
        combined_heads = torch.einsum('btmh,bm->bth', all_sha_outputs, normalized_mask)
        
        dynamic_o_proj = torch.einsum('mhd,bm->bhd', self.o_weights, normalized_mask)
        
        final_output = torch.einsum('bth,bhd->btd', combined_heads, dynamic_o_proj)

        return final_output, all_sha_outputs, pre_activation_logits, activation_mask, fallback_count

class TinyOnnModel(nn.Module):
    def __init__(self, config: DynSMHAConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, dtype=DTYPE)
        self.ln1 = nn.LayerNorm(config.hidden_size, dtype=DTYPE)
        self.smha_layer = DynSMHALayer(config)
        self.ln2 = nn.LayerNorm(config.hidden_size, dtype=DTYPE)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=DTYPE)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = self.embedding(input_ids)
        residual = hidden_states
        
        normalized_states = self.ln1(hidden_states)
        
        attn_output, all_sha_outputs, pre_act_logits, act_mask, fallback_count = self.smha_layer(normalized_states)
        
        hidden_states = residual + attn_output
        hidden_states = self.ln2(hidden_states)
        
        final_logits = self.lm_head(hidden_states)
        
        return final_logits, all_sha_outputs, pre_act_logits, act_mask, fallback_count