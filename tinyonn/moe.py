import torch
import torch.nn as nn


class TinyONNExpert(nn.Module):
    """
    A self-contained "expert" which is a full, smaller MLP.
    It takes hidden_states and returns hidden_states, making the MoE layer
    a simple summation of expert outputs. This is numerically more stable.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.intermediate_size_per_expert = config.intermediate_size // config.num_local_experts

        self.gate_proj = nn.Linear(config.hidden_size, self.intermediate_size_per_expert, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, self.intermediate_size_per_expert, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size_per_expert, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states):
        """
        Performs the full MLP computation for this expert slice.
        """
        intermediate_states = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(intermediate_states)

class MoELayer(nn.Module):
    """
    A Mixture-of-Experts layer that, in dense mode, reproduces the exact
    computation of a standard MLP by summing the outputs of all experts.
    This avoids the numerically unstable `torch.cat` operation.
    """
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.config = config

        # self.router = ... # Router will be added later
        self.experts = nn.ModuleList([TinyONNExpert(config) for _ in range(self.num_experts)])
        # The down_proj is now part of each expert, not centralized.

    def forward(self, hidden_states, force_dense_equivalent_test=False):
        """
        In dense equivalent mode, it sums the outputs of all experts.
        This is mathematically equivalent to the original large MLP.
        """
        # The sparse logic will use a weighted sum based on a router.
        expert_outputs = [expert(hidden_states) for expert in self.experts]
        # Stack and sum. This is more robust than concatenating intermediate states.
        return torch.sum(torch.stack(expert_outputs, dim=0), dim=0)
