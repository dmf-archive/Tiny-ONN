import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CAPR(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        self.proto_k = nn.Parameter(torch.empty(num_experts, hidden_size, dtype=dtype))
        self.gate = nn.Parameter(torch.empty(num_experts, dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.proto_k, std=0.02)
        nn.init.normal_(self.gate, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (..., hidden_size)
        # proto_k: (num_experts, hidden_size)

        # Scaled Dot-Product Attention as Routing
        raw_logits = torch.einsum('...d,pd->...p', x, self.proto_k) / math.sqrt(self.hidden_size)

        # Subtract gate (bias) to control activation threshold
        raw_logits = raw_logits - self.gate.unsqueeze(0)

        # Apply ReLU for sparse activation
        routing_logits = F.relu(raw_logits)

        # Select Top-K experts
        routing_weights, selected_experts = torch.topk(routing_logits, self.top_k, dim=-1)

        # Re-normalize weights if needed, or use as is for gated output
        # In DynSIHA, we often use the raw relu output for gating

        return routing_weights, selected_experts
