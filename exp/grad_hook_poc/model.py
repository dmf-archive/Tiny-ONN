import torch
import torch.nn as nn
from typing import Tuple

class Expert(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SparseMoE(nn.Module):
    def __init__(self, d_model: int, n_experts: int, top_k: int):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d_model, d_model * 4) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        x_reshaped = x.view(-1, d_model)
        
        router_logits = self.gate(x_reshaped)
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = nn.functional.softmax(routing_weights, dim=-1, dtype=torch.float)
        routing_weights = routing_weights.to(x.dtype)

        final_output = torch.zeros_like(x_reshaped)
        
        flat_expert_indices = selected_experts.flatten()
        flat_routing_weights = routing_weights.flatten()
        
        token_indices = torch.arange(x_reshaped.size(0), device=x.device).repeat_interleave(self.top_k)
        
        for i in range(self.n_experts):
            expert_mask = (flat_expert_indices == i)
            if expert_mask.any():
                tokens_for_expert = x_reshaped[token_indices[expert_mask]]
                weights_for_expert = flat_routing_weights[expert_mask].unsqueeze(-1)
                
                expert_output = self.experts[i](tokens_for_expert)
                weighted_output = expert_output * weights_for_expert
                
                final_output.index_add_(0, token_indices[expert_mask], weighted_output)

        return final_output.view(batch_size, seq_len, d_model)

class PocMoEModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_experts: int, top_k: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.moe = SparseMoE(d_model, n_experts, top_k)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embedding(x)
        hidden_states = self.moe(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits