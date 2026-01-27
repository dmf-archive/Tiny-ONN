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
        
        # 贝叶斯竞争：全局 Softmax 归一化
        probs = F.softmax(routing_logits, dim=-1)
        routing_weights = probs

        if self.top_k < self.num_experts:
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        else:
            selected_experts = torch.arange(self.num_experts, device=x.device).expand_as(routing_weights)
            
            # 物理截断：优化 Kernel 调度，真正为 0 的部分不参与计算
            routing_weights = torch.where(routing_weights > 1e-5, routing_weights, torch.zeros_like(routing_weights))

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

        combined_output = torch.zeros(x_flat.shape[0], D, device=x.device, dtype=x.dtype)

        # Memory efficient MoE: Loop over experts instead of materializing all weights
        for i in range(self.num_experts):
            # 优化调度：只有当该专家在当前 Batch 中有非零贡献时才启动 Kernel
            # 在 Top-Any 模式下，这能显著减少无效计算
            expert_mask = (se_flat == i) & (rw_flat > 0)
            if not expert_mask.any():
                continue

            # Get token indices and which 'k' slot they used
            token_indices, k_slots = torch.where(expert_mask)

            # Extract inputs and weights
            expert_inputs = x_flat[token_indices] # [num_matched, D]
            weights = rw_flat[token_indices, k_slots].unsqueeze(-1) # [num_matched, 1]

            # Compute expert output
            # hidden = silu(x @ w1) @ w2
            hidden = F.silu(torch.matmul(expert_inputs, self.w1[i]))
            out = torch.matmul(hidden, self.w2[i])

            # Accumulate weighted output
            combined_output.index_add_(0, token_indices, out * weights)

        return combined_output.view(shape)
