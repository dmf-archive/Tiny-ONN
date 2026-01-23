import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden_size, num_experts, bias=False, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        routing_logits = self.net(x)
        
        if self.top_k < self.num_experts:
            routing_weights, selected_experts = torch.topk(routing_logits, self.top_k, dim=-1)
            routing_weights = F.softmax(routing_weights, dim=-1)
        else:
            routing_weights = F.softmax(routing_logits, dim=-1)
            selected_experts = torch.arange(self.num_experts, device=x.device).expand_as(routing_logits)
            
        return routing_weights, selected_experts, routing_logits

class VectorizedExpertMLP(nn.Module):
    def __init__(
        self,
        num_experts: int,
        input_size: int,
        intermediate_size: int,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.num_experts = num_experts
        self.input_size = input_size
        self.intermediate_size = intermediate_size

        self.w1 = nn.Parameter(torch.empty(num_experts, input_size, intermediate_size, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(num_experts, intermediate_size, input_size, dtype=dtype))
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

    def forward(self, x: torch.Tensor, routing_weights: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        D = shape[-1]
        K = routing_weights.shape[-1]
        
        x_flat = x.view(-1, D)
        rw_flat = routing_weights.view(-1, K)
        se_flat = selected_experts.view(-1, K)
        
        S = x_flat.shape[0]
        expert_inputs = x_flat.unsqueeze(1).expand(S, K, D)
        
        selected_w1 = self.w1[se_flat]
        selected_w2 = self.w2[se_flat]
        
        hidden = torch.matmul(expert_inputs.unsqueeze(-2), selected_w1)
        hidden = F.silu(hidden)
        
        output = torch.matmul(hidden, selected_w2)
        output = output.squeeze(-2)
        
        weighted_output = output * rw_flat.unsqueeze(-1)
        return weighted_output.sum(dim=1).view(shape)
