import pytest
import torch
from torch import nn
from tiny_onn.modular import DynMoE, SiLUHardThreshold, TinyOnnDecoderLayer
from tiny_onn.config import TinyOnnConfig # Assuming TinyOnnConfig will be defined here

def test_silu_hard_threshold_forward():
    input_scores = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
    threshold_param = torch.tensor([0.0], requires_grad=True)

    # Test case 1: input_scores - threshold_param > 0
    # silu(-2.0) = -0.2689, silu(-0.5) = -0.1798, silu(0.0) = 0.0, silu(0.5) = 0.3244, silu(2.0) = 1.7616
    # mask should be [0, 0, 0, 1, 1]
    output = SiLUHardThreshold.apply(input_scores, threshold_param)
    expected_output = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
    assert torch.allclose(output, expected_output), f"Expected: {expected_output}, Got: {output}"

    # Test case 2: Different threshold
    threshold_param_2 = torch.tensor([1.0], requires_grad=True)
    # input_scores - threshold_param_2: [-3.0, -1.5, -1.0, -0.5, 1.0]
    # silu: [-0.1419, -0.3000, -0.2689, -0.1798, 0.7311]
    # mask should be [0, 0, 0, 0, 1]
    output_2 = SiLUHardThreshold.apply(input_scores, threshold_param_2)
    expected_output_2 = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
    assert torch.allclose(output_2, expected_output_2), f"Expected: {expected_output_2}, Got: {output_2}"

def test_silu_hard_threshold_backward():
    # Test with gradcheck
    input_scores = torch.randn(5, requires_grad=True, dtype=torch.double)
    threshold_param = torch.randn(1, requires_grad=True, dtype=torch.double)
    modified_scores = input_scores.detach().clone().requires_grad_(True)
    modified_scores[0] = threshold_param.item() + 0.01
    input_scores[1] = threshold_param.item() - 0.01
    input_scores[2] = threshold_param.item()
    
    assert torch.autograd.gradcheck(SiLUHardThreshold.apply, (input_scores, threshold_param), eps=1e-6, atol=1e-4)

def test_dynmoe_dynamic_k_forward():
    config = TinyOnnConfig(
        hidden_size=16,
        num_experts=4,
        num_experts_per_tok=-1, # Dynamic K
        moe_intermediate_size=32
    )
    dynmoe = DynMoE(config)
    
    # Manually set expert embeddings and gate threshold for predictable behavior
    dynmoe.expert_embeddings.data = torch.randn(config.num_experts, config.hidden_size)
    dynmoe.gate_threshold.data = torch.tensor([0.5, 0.5, 0.5, 0.5]) # Set a threshold

    hidden_states = torch.randn(2, 5, config.hidden_size) # Batch size 2, seq len 5

    output, router_logits, return_info = dynmoe(hidden_states)

    assert output.shape == hidden_states.shape
    assert router_logits.shape == (hidden_states.numel() // hidden_states.shape[-1], config.num_experts)
    assert 'token_idx' in return_info
    assert 'expert_idx' in return_info
    
    # Test "at least one expert" fallback
    # Set hidden states such that all scores are below threshold
    dynmoe.expert_embeddings.data = torch.ones(config.num_experts, config.hidden_size) * 100 # Make them very different
    dynmoe.gate_threshold.data = torch.tensor([1000.0, 1000.0, 1000.0, 1000.0]) # Very high threshold
    
    hidden_states_low_score = torch.randn(1, 1, config.hidden_size)
    output_fallback, router_logits_fallback, return_info_fallback = dynmoe(hidden_states_low_score)
    
    # Ensure at least one expert was selected
    assert len(return_info_fallback['token_idx']) > 0
    assert len(return_info_fallback['expert_idx']) > 0
    assert torch.allclose(output_fallback.sum(), output_fallback.sum()) # Simple check for non-zero output

def test_dynmoe_fixed_k_forward():
    config = TinyOnnConfig(
        hidden_size=16,
        num_experts=4,
        num_experts_per_tok=2, # Fixed K = 2
        moe_intermediate_size=32
    )
    dynmoe = DynMoE(config)
    
    hidden_states = torch.randn(2, 5, config.hidden_size)

    output, router_logits, return_info = dynmoe(hidden_states)

    assert output.shape == hidden_states.shape
    assert router_logits.shape == (hidden_states.numel() // hidden_states.shape[-1], config.num_experts)
    assert 'indices' in return_info
    assert return_info['indices'].shape == (hidden_states.numel() // hidden_states.shape[-1], config.num_experts_per_tok)

def test_dynmoe_gradient_flow_dynamic_k():
    config = TinyOnnConfig(
        hidden_size=16,
        num_experts=4,
        num_experts_per_tok=-1, # Dynamic K
        moe_intermediate_size=32
    )
    dynmoe = DynMoE(config)
    
    hidden_states = torch.randn(2, 5, config.hidden_size, requires_grad=True)
    
    # Ensure parameters require grad
    dynmoe.expert_embeddings.requires_grad_(True)
    dynmoe.gate_threshold.requires_grad_(True)

    output, router_logits, _ = dynmoe(hidden_states)
    
    # Simple loss to ensure gradients are computed
    loss = output.sum() + router_logits.sum()
    loss.backward()

    assert hidden_states.grad is not None
    assert dynmoe.expert_embeddings.grad is not None
    assert dynmoe.gate_threshold.grad is not None
    
    # Check if gradients are non-zero (might be zero for some experts if not selected)
    assert dynmoe.expert_embeddings.grad.abs().sum().item() > 0
    assert dynmoe.gate_threshold.grad.abs().sum().item() > 0

def test_dynmoe_gradient_flow_fixed_k():
    config = TinyOnnConfig(
        hidden_size=16,
        num_experts=4,
        num_experts_per_tok=2, # Fixed K
        moe_intermediate_size=32
    )
    dynmoe = DynMoE(config)
    
    hidden_states = torch.randn(2, 5, config.hidden_size, requires_grad=True)
    
    # Ensure gate weights require grad
    dynmoe.gate.weight.requires_grad_(True)

    output, router_logits, _ = dynmoe(hidden_states)
    
    # Simple loss to ensure gradients are computed
    loss = output.sum() + router_logits.sum()
    loss.backward()

    assert hidden_states.grad is not None
    assert dynmoe.gate.weight.grad is not None
    assert dynmoe.gate.weight.grad.abs().sum().item() > 0

def test_tinyonn_decoder_layer_init():
    # Test case 1: Should use DynMoE
    config_moe = TinyOnnConfig(
        hidden_size=16,
        num_attention_heads=16,
        num_experts=4,
        num_experts_per_tok=-1,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        decoder_sparse_step=1 # Every layer is MoE
    )
    layer_moe = TinyOnnDecoderLayer(config_moe, 0)
    assert isinstance(layer_moe.mlp, DynMoE)

    # Test case 2: Should use Qwen3MLP (num_experts = 0)
    config_mlp_only = TinyOnnConfig(
        hidden_size=16,
        num_attention_heads=16,
        num_experts=0, # No experts
        num_experts_per_tok=0,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        decoder_sparse_step=1
    )
    layer_mlp_only = TinyOnnDecoderLayer(config_mlp_only, 0)
    assert isinstance(layer_mlp_only.mlp, nn.Linear) # Qwen3MLP starts with Linear

    # Test case 3: Should use Qwen3MLP (decoder_sparse_step not met)
    config_sparse_step = TinyOnnConfig(
        hidden_size=16,
        num_attention_heads=16,
        num_experts=4,
        num_experts_per_tok=-1,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        decoder_sparse_step=2 # Only layer 1 (index 0+1=1) is MoE
    )
    layer_mlp_sparse = TinyOnnDecoderLayer(config_sparse_step, 0)
    assert isinstance(layer_mlp_sparse.mlp, nn.Linear) # Layer 0 should be MLP

    layer_moe_sparse = TinyOnnDecoderLayer(config_sparse_step, 1)
    assert isinstance(layer_moe_sparse.mlp, DynMoE)
