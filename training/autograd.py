from typing import Any

import torch


class CaptureSurprise(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        expert_input: torch.Tensor,
        layer_idx: int,
        expert_idx: int,
        token_indices: torch.Tensor,
        surprise_context: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        ctx.layer_idx = layer_idx
        ctx.expert_idx = expert_idx
        ctx.token_indices = token_indices
        ctx.surprise_context = surprise_context
        return expert_input

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        surprise = torch.linalg.norm(grad_output.flatten(start_dim=1), dim=1).detach()
        surprise_fp8 = surprise.to(torch.float8_e5m2)

        ctx.surprise_context[(ctx.layer_idx, ctx.expert_idx)] = (ctx.token_indices, surprise_fp8)

        return grad_output, None, None, None, None
