import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BitSBL(nn.Module):
    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu_weight_latent = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.gate_param = nn.Parameter(torch.empty(out_features, dtype=dtype))
        self.mu_bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.mu_weight_latent, a=math.sqrt(5))
        nn.init.normal_(self.sigma_weight, mean=0.0, std=0.5)
        nn.init.constant_(self.gate_param, -0.1)
        nn.init.zeros_(self.mu_bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        x_reshaped = x.view(-1, self.in_features)
        
        keys = self.mu_weight_latent * F.softplus(self.sigma_weight)
        scores = torch.matmul(x_reshaped, keys.t()) / math.sqrt(self.in_features)
        raw_weights = F.relu(scores - self.gate_param.unsqueeze(0))

        w_binary = torch.sign(self.mu_weight_latent)
        w_binary_ste = (w_binary - self.mu_weight_latent).detach() + self.mu_weight_latent
        
        computation_output = F.linear(x_reshaped, w_binary_ste, self.mu_bias)
        masked_output = computation_output * raw_weights
        
        new_shape = list(original_shape[:-1]) + [self.out_features]
        output = masked_output.view(new_shape)
        return output, scores, masked_output