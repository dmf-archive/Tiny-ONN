import torch
import torch.nn as nn
import torch.nn.functional as F

class CPRRouter(nn.Module):
    def __init__(self, num_experts: int, hidden_size: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.proto = nn.Parameter(torch.empty(num_experts, hidden_size))
        nn.init.normal_(self.proto, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_hidden_states = F.normalize(hidden_states, p=2, dim=-1)
        normalized_proto = F.normalize(self.proto, p=2, dim=-1)

        router_logits = torch.matmul(normalized_hidden_states, normalized_proto.t())
        
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        return routing_weights, selected_experts
