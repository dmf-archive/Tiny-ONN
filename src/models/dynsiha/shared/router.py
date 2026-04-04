import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        temperature: float = 0.1,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        self.last_active_counts = None
        # Second-order moment tracking for curvature control (usage intensity proxy)
        self.register_buffer("v_running", torch.ones(num_experts, dtype=dtype))

        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype),
            nn.LayerNorm(hidden_size, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden_size, num_experts, bias=True, dtype=dtype)
        )
        self.expert_bias = nn.Parameter(torch.zeros(num_experts, dtype=dtype))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        routing_logits = (self.net(x) + self.expert_bias) / self.temperature
        routing_weights = F.softmax(routing_logits, dim=-1)
        selected_experts = torch.arange(self.num_experts, device=x.device).expand_as(routing_weights)

        if self.training:
            with torch.no_grad():
                # Update second-order moment proxy (usage intensity)
                curr_v = routing_weights.view(-1, self.num_experts).pow(2).mean(0)
                self.v_running.copy_(0.95 * self.v_running + 0.05 * curr_v)

        # Remove hard thresholding to ensure full gradient flow during early training
        # Use a soft threshold for active count statistics only
        self.last_active_counts = (routing_weights > 1e-3).sum(dim=-1)
        return routing_weights, selected_experts, routing_logits


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    original_size = logits.size()
    logits = logits.transpose(dim, -1)
    logits = logits.reshape(-1, logits.size(-1))
    z_sorted, _ = torch.sort(logits, descending=True, dim=-1)
    k = torch.arange(1, z_sorted.size(-1) + 1, device=logits.device, dtype=logits.dtype).unsqueeze(0)
    z_cumsum = z_sorted.cumsum(dim=-1)
    k_z = 1 + k * z_sorted > z_cumsum
    k_max = k_z.sum(dim=-1, keepdim=True).clamp(min=1)
    tau = (z_cumsum.gather(dim=-1, index=k_max - 1) - 1) / k_max
    output = torch.clamp(logits - tau, min=0)
    output = output.view(*original_size)
    return output

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
        x_flat = x.view(-1, D)
        rw_flat = routing_weights.view(-1, self.num_experts)

        hidden = F.silu(torch.einsum("nd,edh->neh", x_flat, self.w1))
        out = torch.einsum("neh,ehd->ned", hidden, self.w2)
        combined_output = (out * rw_flat.unsqueeze(-1)).sum(dim=1)
        return combined_output.view(shape)


def compute_fars_cost(weights: torch.Tensor, cost_vec: torch.Tensor) -> torch.Tensor:
    """
    weights: [B, E] 路由权重（softmax 后）
    cost_vec: [E]   per-expert 归一化二阶矩 v_t / max(v_t)
    返回：标量损失，鼓励低认知代价专家获得高权重
    """
    return (weights * cost_vec.unsqueeze(0)).sum()
