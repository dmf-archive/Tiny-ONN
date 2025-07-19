import torch
import torch.nn as nn


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


class ExpertGroup(nn.Module):
    def __init__(self, d_model: int, n_experts_per_group: int, top_k_l2: int):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts_per_group
        self.top_k = top_k_l2
        self.gate = nn.Linear(d_model, self.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [Expert(d_model, d_model * 2) for _ in range(self.n_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) == 0:
            return torch.empty(0, self.d_model, device=x.device, dtype=x.dtype)

        router_logits = self.gate(x)
        routing_weights, selected_experts = torch.topk(
            router_logits, self.top_k, dim=-1
        )
        routing_weights = nn.functional.softmax(
            routing_weights, dim=-1, dtype=torch.float
        ).to(x.dtype)

        final_output = torch.zeros_like(x)

        flat_expert_indices = selected_experts.flatten()
        flat_routing_weights = routing_weights.flatten()
        token_indices = torch.arange(x.size(0), device=x.device).repeat_interleave(
            self.top_k
        )

        for i in range(self.n_experts):
            expert_mask = flat_expert_indices == i
            if expert_mask.any():
                tokens_for_expert = x[token_indices[expert_mask]]
                weights_for_expert = flat_routing_weights[expert_mask].unsqueeze(-1)
                expert_output = self.experts[i](tokens_for_expert)
                weighted_output = expert_output * weights_for_expert
                final_output.index_add_(0, token_indices[expert_mask], weighted_output)

        return final_output


class HierarchicalMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_groups: int,
        n_experts_per_group: int,
        top_k_l1: int,
        top_k_l2: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_groups = n_groups
        self.top_k = top_k_l1
        self.gate = nn.Linear(d_model, self.n_groups, bias=False)
        self.groups = nn.ModuleList(
            [
                ExpertGroup(d_model, n_experts_per_group, top_k_l2)
                for _ in range(self.n_groups)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        x_reshaped = x.view(-1, d_model)

        router_logits = self.gate(x_reshaped)
        routing_weights, selected_groups = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = nn.functional.softmax(
            routing_weights, dim=-1, dtype=torch.float
        ).to(x.dtype)

        final_output = torch.zeros_like(x_reshaped)

        flat_group_indices = selected_groups.flatten()
        flat_routing_weights = routing_weights.flatten()
        token_indices = torch.arange(
            x_reshaped.size(0), device=x.device
        ).repeat_interleave(self.top_k)

        for i in range(self.n_groups):
            group_mask = flat_group_indices == i
            if group_mask.any():
                tokens_for_group = x_reshaped[token_indices[group_mask]]
                weights_for_group = flat_routing_weights[group_mask].unsqueeze(-1)

                group_output = self.groups[i](tokens_for_group)
                weighted_output = group_output * weights_for_group

                final_output.index_add_(0, token_indices[group_mask], weighted_output)

        return final_output.view(batch_size, seq_len, d_model)


class PocHMoEModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_groups: int,
        n_experts_per_group: int,
        top_k_l1: int,
        top_k_l2: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.hmoe = HierarchicalMoE(
            d_model, n_groups, n_experts_per_group, top_k_l1, top_k_l2
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embedding(x)
        hidden_states = self.hmoe(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits
