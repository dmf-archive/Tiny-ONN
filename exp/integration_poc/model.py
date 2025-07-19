import torch
import torch.nn as nn
from transformers.models.llama.configuration_llama import LlamaConfig


class Expert(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act_fn(self.w1(x)) * self.w3(x))


class SimpleMoE(nn.Module):
    def __init__(self, config: LlamaConfig, n_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.d_model = config.hidden_size
        self.n_experts = n_experts
        self.top_k = top_k

        self.gate = nn.Linear(self.d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(config) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        x_reshaped = x.view(-1, d_model)

        router_logits = self.gate(x_reshaped)
        routing_weights, selected_experts = torch.topk(
            router_logits, self.top_k, dim=-1
        )
        routing_weights = nn.functional.softmax(
            routing_weights, dim=-1, dtype=torch.float
        ).to(x.dtype)

        final_output = torch.zeros_like(x_reshaped)

        flat_expert_indices = selected_experts.flatten()
        token_indices = torch.arange(
            x_reshaped.size(0), device=x.device
        ).repeat_interleave(self.top_k)

        for i in range(self.n_experts):
            expert_mask = flat_expert_indices == i
            if not torch.any(expert_mask):
                continue

            tokens_for_expert_indices = token_indices[expert_mask]

            weights_for_expert = routing_weights.flatten()[expert_mask].unsqueeze(1)

            expert_output = self.experts[i](x_reshaped[tokens_for_expert_indices])

            weighted_output = expert_output * weights_for_expert
            final_output.index_add_(0, tokens_for_expert_indices, weighted_output)

        return final_output.view(batch_size, seq_len, d_model)
