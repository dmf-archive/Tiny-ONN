import pytest
import torch

from tiny_onn.config import TinyOnnConfig
from tiny_onn.modular import TinyOnnMoE


@pytest.fixture
def config():
    return TinyOnnConfig(
        hidden_size=128,
        num_experts_per_layer=8,
        moe_intermediate_size=32,
    )


def test_tiny_onn_moe_forward(config):
    config.num_experts_per_tok = 2
    moe = TinyOnnMoE(config)
    hidden_states = torch.randn(4, 10, config.hidden_size)
    output = moe(hidden_states, surprise_budget=0.1)

    assert output.shape == hidden_states.shape
    assert moe.last_router_logits is not None
    assert moe.last_router_logits.shape == (4 * 10, config.num_experts_per_layer)


def test_tiny_onn_moe_dynamic_k_forward(config):
    config.num_experts_per_tok = -1  # Enable dynamic k
    moe = TinyOnnMoE(config)
    hidden_states = torch.randn(2, 5, config.hidden_size)

    # Test with a low budget, should select fewer experts
    low_budget_output = moe(hidden_states, surprise_budget=0.1)
    assert low_budget_output.shape == hidden_states.shape
    assert moe.last_selected_experts is not None
    num_selected_low = (moe.last_selected_experts != -1).sum(dim=-1)

    # Test with a high budget, should select more experts
    high_budget_output = moe(hidden_states, surprise_budget=0.9)
    assert high_budget_output.shape == hidden_states.shape
    assert moe.last_selected_experts is not None
    num_selected_high = (moe.last_selected_experts != -1).sum(dim=-1)

    # On average, a higher budget should lead to more experts being selected
    assert num_selected_high.float().mean() > num_selected_low.float().mean()
