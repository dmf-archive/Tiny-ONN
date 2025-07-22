import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from tiny_onn.config import TinyOnnConfig
from tiny_onn.modular import DynMoE, STEGate, TinyOnnDecoderLayer


def test_ste_gate_forward():
    scores = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    threshold = torch.tensor(0.0)
    output = STEGate.apply(scores, threshold)
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
    assert torch.allclose(output, expected)


def test_dynmoe_forward_and_fallback():
    config = TinyOnnConfig(
        hidden_size=16,
        num_experts=4,
        num_experts_per_tok=-1,
        moe_intermediate_size=32,
    )
    dynmoe = DynMoE(config)
    hidden_states = torch.randn(2, 5, config.hidden_size)

    output, router_logits, selected_experts = dynmoe(hidden_states)
    assert output.shape == hidden_states.shape
    assert router_logits.shape == (10, config.num_experts)
    assert selected_experts.shape == (10, config.num_experts)
    assert not torch.isnan(output).any()

    dynmoe.gate_threshold.data.fill_(1000.0)
    output_fallback, _, _ = dynmoe(hidden_states)
    assert not torch.isnan(output_fallback).any()


def test_dynmoe_gradient_flow():
    config = TinyOnnConfig(
        hidden_size=16,
        num_experts=4,
        num_experts_per_tok=-1,
        moe_intermediate_size=32,
    )
    dynmoe = DynMoE(config)
    hidden_states = torch.randn(2, 5, config.hidden_size, requires_grad=True)

    output, router_logits, _ = dynmoe(hidden_states)
    output, router_logits, routing_mask = dynmoe(hidden_states)
    loss = output.sum() + router_logits.sum() + routing_mask.sum()
    loss.backward()

    assert hidden_states.grad is not None
    assert dynmoe.expert_embeddings.grad is not None
    assert dynmoe.gate_threshold.grad is not None

    for expert in dynmoe.experts:
        for param in expert.parameters():
            assert param.grad is not None

    assert dynmoe.expert_embeddings.grad.abs().sum().item() > 0
    assert dynmoe.gate_threshold.grad.abs().sum().item() > 0


def test_tinyonn_decoder_layer_init():
    config_moe = TinyOnnConfig(
        num_experts=4, num_experts_per_tok=-1, decoder_sparse_step=1
    )
    layer_moe = TinyOnnDecoderLayer(config_moe, 0)
    assert isinstance(layer_moe.mlp, DynMoE)

    config_mlp_no_experts = TinyOnnConfig(num_experts=0)
    layer_mlp_no_experts = TinyOnnDecoderLayer(config_mlp_no_experts, 0)
    assert isinstance(layer_mlp_no_experts.mlp, Qwen3MLP)

    config_mlp_sparse_step = TinyOnnConfig(num_experts=4, decoder_sparse_step=2)
    layer_mlp_sparse_step = TinyOnnDecoderLayer(config_mlp_sparse_step, 0)
    assert isinstance(layer_mlp_sparse_step.mlp, Qwen3MLP)
    layer_moe_sparse_step = TinyOnnDecoderLayer(config_mlp_sparse_step, 1)
    assert isinstance(layer_moe_sparse_step.mlp, DynMoE)
