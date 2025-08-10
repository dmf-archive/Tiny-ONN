import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple, Dict

from .config import DenseConfig, DynNSAConfig, DEVICE, DTYPE

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        return (scores > 0).to(scores.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class DenseModel(nn.Module):
    def __init__(self, config: DenseConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.attn = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            batch_first=True,
            dtype=DTYPE,
            device=DEVICE,
            bias=False
        )
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, T = input_ids.shape
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=DEVICE)
        
        hidden_states = self.embedding(input_ids)
        attn_output, _ = self.attn(hidden_states, hidden_states, hidden_states, attn_mask=causal_mask, need_weights=False)
        hidden_states = self.ln1(hidden_states + attn_output)
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.ln2(hidden_states + ffn_output)
        logits = self.lm_head(hidden_states)
        return logits, {}

class DynNSAGate(nn.Module):
    def __init__(self, config: DynNSAConfig):
        super().__init__()
        self.config = config
        self.sim_matrix = nn.Parameter(torch.randn(config.hidden_size, config.num_heads))
        self.gates = nn.Parameter(torch.zeros(config.num_heads))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled_x = torch.mean(x, dim=1)
        
        logits = torch.matmul(F.normalize(pooled_x, dim=-1), F.normalize(self.sim_matrix, dim=0))
        gate_thresholds = torch.sigmoid(self.gates)
        pre_activation_logits = logits - gate_thresholds

        gated_logits = F.relu(pre_activation_logits)
        activation_mask = STEFunction.apply(gated_logits)

        num_active_experts = torch.sum(activation_mask, dim=1)
        inactive_seq_mask = num_active_experts == 0
        if inactive_seq_mask.any():
            k_fallback = self.config.num_heads // 2 if self.config.num_heads > 1 else 1
            topk_expert_indices = torch.topk(logits[inactive_seq_mask], k=k_fallback, dim=1).indices
            for i, idx in enumerate(torch.where(inactive_seq_mask)[0]):
                activation_mask[idx, topk_expert_indices[i]] = 1.0
        
        gated_logits_masked = torch.where(activation_mask > 0, gated_logits, -torch.finfo(DTYPE).max)
        active_expert_probs = F.softmax(gated_logits_masked, dim=-1)

        return active_expert_probs, pre_activation_logits, activation_mask

class DynNSALayer_v2(nn.Module):
    def __init__(self, config: DynNSAConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * self.head_dim, config.hidden_size, bias=False)
        
        self.gate = DynNSAGate(config)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C = hidden_states.shape
        
        routing_weights, pre_act_logits, activation_mask = self.gate(hidden_states)
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads)
        
        all_head_outputs = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        weighted_output = all_head_outputs * routing_weights.view(B, self.num_heads, 1, 1)

        combined_heads = rearrange(weighted_output, 'b h t d -> b t (h d)')
        
        final_output = self.o_proj(combined_heads)

        return final_output, all_head_outputs, pre_act_logits, activation_mask

class DynNSAv2Model(nn.Module):
    def __init__(self, config: DynNSAConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.attn = DynNSALayer_v2(config)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = self.embedding(input_ids)
        
        attn_output, all_head_outputs, pre_act_logits, activation_mask = self.attn(self.ln1(hidden_states))
        hidden_states = hidden_states + attn_output
        
        ffn_output = self.ffn(self.ln2(hidden_states))
        hidden_states = hidden_states + ffn_output
        
        logits = self.lm_head(hidden_states)
        
        return logits, all_head_outputs, pre_act_logits, activation_mask