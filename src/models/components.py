import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.w1: nn.Linear = nn.Linear(input_size, output_size, bias=False)
        self.w2: nn.Linear = nn.Linear(output_size, input_size, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w1_out: torch.Tensor = self.w1(x)
        silu_out: torch.Tensor = F.silu(w1_out)
        w2_out: torch.Tensor = self.w2(silu_out)
        return w2_out, w1_out

class SparseProtoLinear(nn.Module):
    def __init__(self, d_model: int, num_experts: int, head_dim: int) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.num_experts: int = num_experts
        self.head_dim: int = head_dim

        self.proto: nn.Parameter = nn.Parameter(torch.empty(num_experts, head_dim))
        self.gate: nn.Parameter = nn.Parameter(torch.empty(num_experts))

        self.experts: nn.ModuleList = nn.ModuleList([ExpertMLP(head_dim, head_dim) for _ in range(num_experts)])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.proto, std=0.02)
        nn.init.normal_(self.gate, mean=0.0, std=0.02)

    def forward(self, x_proj: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B: int
        T: int
        H: int
        D_h: int
        B, T, H, D_h = x_proj.shape
        S: int = B * T * H
        P: int = self.num_experts

        x_flat: torch.Tensor = x_proj.reshape(S, D_h)

        logits: torch.Tensor = torch.einsum('bthd,pd->bthp', x_proj, self.proto) / math.sqrt(self.head_dim)
        logits = logits - self.gate.view(1, 1, 1, P)

        mask: torch.Tensor = F.relu(logits)
        mask_flat: torch.Tensor = mask.reshape(S, P)

        active_mask: torch.Tensor = mask_flat > 1e-6
        all_outputs: torch.Tensor = torch.zeros(S, P, D_h, dtype=x_proj.dtype, device=x_proj.device)

        for i in range(P):
            m: torch.Tensor = active_mask[:, i]
            if m.any():
                out: torch.Tensor
                out, _ = self.experts[i](x_flat[m])
                all_outputs[m, i] = out * mask_flat[m, i].unsqueeze(-1)

        output: torch.Tensor = all_outputs.sum(dim=1).view(B, T, H, D_h)

        routing_info: dict[str, torch.Tensor] = {
            "logits": logits,
            "mask": mask,
            "active_mask": active_mask
        }

        return output, routing_info
