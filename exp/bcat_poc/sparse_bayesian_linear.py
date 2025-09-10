import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List

from .config import CONFIG

@torch.jit.script
def bcat_cluster(raw_weights: torch.Tensor, grid_size: int, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = raw_weights > threshold
    
    if not mask.any():
        return torch.empty((0, 4), dtype=torch.int32, device=raw_weights.device), torch.empty((0,), dtype=raw_weights.dtype, device=raw_weights.device)

    rows = torch.tensor(raw_weights.shape[0], device=raw_weights.device)
    cols = torch.tensor(raw_weights.shape[1], device=raw_weights.device)
    active_coords = torch.nonzero(mask)

    if active_coords.shape[0] == 0:
        return torch.empty((0, 4), dtype=torch.int32, device=raw_weights.device), torch.empty((0,), dtype=raw_weights.dtype, device=raw_weights.device)

    row_bins = torch.div(active_coords[:, 0], grid_size, rounding_mode='floor')
    col_bins = torch.div(active_coords[:, 1], grid_size, rounding_mode='floor')
    
    num_row_grids = torch.div(rows - 1, grid_size, rounding_mode='floor') + 1
    bin_indices = row_bins * num_row_grids + col_bins
    
    unique_bins, inverse_indices = torch.unique(bin_indices, return_inverse=True)

    block_meta_list: List[torch.Tensor] = []
    block_values_list: List[torch.Tensor] = []

    for i in range(unique_bins.shape[0]):
        bin_mask = inverse_indices == i
        points_in_bin = active_coords[bin_mask]
        
        min_coords = torch.min(points_in_bin, dim=0).values
        max_coords = torch.max(points_in_bin, dim=0).values
        
        r_start, c_start = min_coords[0], min_coords[1]
        r_end, c_end = max_coords[0], max_coords[1]
        
        row_len = r_end - r_start + 1
        col_len = c_end - c_start + 1

        block_meta_list.append(torch.tensor([int(r_start), int(c_start), int(row_len), int(col_len)], dtype=torch.int32, device=raw_weights.device))
        block_values_list.append(raw_weights[r_start:r_end+1, c_start:c_end+1].flatten())

    if not block_meta_list:
        return torch.empty((0, 4), dtype=torch.int32, device=raw_weights.device), torch.empty((0,), dtype=raw_weights.dtype, device=raw_weights.device)

    return torch.stack(block_meta_list), torch.cat(block_values_list)


class SparseBayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bcat_grid_size: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bcat_grid_size = bcat_grid_size

        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.gate_param = nn.Parameter(torch.empty(out_features, dtype=dtype))
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.mu_weight, a=math.sqrt(5))
        nn.init.normal_(self.sigma_weight, mean=0.0, std=0.5)
        nn.init.constant_(self.gate_param, -0.1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        x_reshaped = x.view(-1, self.in_features)

        keys = self.mu_weight * F.softplus(self.sigma_weight)
        scores = torch.matmul(x_reshaped, keys.t()) / math.sqrt(self.in_features)
        raw_weights = F.relu(scores - self.gate_param.unsqueeze(0))

        activation_rate = (raw_weights > 0).float().mean()
        
        masked_output: torch.Tensor = torch.empty_like(raw_weights)
        # For this PoC, we validate that bcat_cluster is JIT-compatible and runnable,
        # but we fall back to the JIT-optimized dense computation for both forward and backward
        # to prevent autograd performance issues with the Python-based scatter implementation.
        # The true performance test requires a Triton/CUDA kernel.
        if self.training and activation_rate < 0.95 and activation_rate > 0.0:
            # Call bcat_cluster to ensure it compiles and runs without error under JIT.
            # We ignore the output for computation to avoid the autograd bottleneck.
            _ = bcat_cluster(raw_weights, self.bcat_grid_size, 0.01)

        computation_output = F.linear(x_reshaped, self.mu_weight)
        masked_output = computation_output * raw_weights
        
        new_shape = list(original_shape[:-1]) + [self.out_features]
        final_output = masked_output.view(new_shape)
        
        sbl_bias = torch.zeros(self.out_features, dtype=x.dtype, device=x.device)
        final_output += sbl_bias

        return final_output, scores, masked_output
