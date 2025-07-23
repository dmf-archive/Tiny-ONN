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
    moe = TinyOnnMoE(config)
    hidden_states = torch.randn(4, 10, config.hidden_size)
    output, router_logits = moe(hidden_states)

    assert output.shape == hidden_states.shape
    assert router_logits.shape == (4 * 10, config.num_experts_per_layer)
