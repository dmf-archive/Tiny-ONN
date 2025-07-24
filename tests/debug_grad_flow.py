from typing import cast

import torch
from transformers import AutoTokenizer

from tiny_onn.modular import TinyOnnForCausalLM, TinyOnnMoE


def test_manual_surprise_calculation():
    model_path = "weights/Tiny-ONN-0.6B-Hyper-SMoE"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyOnnForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.to(device)
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch_size, seq_len = 4, 32
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss

    print("\n--- Manual Gradient Flow Analysis using torch.autograd.grad ---")

    total_surprise_norm = 0.0
    num_experts_with_grad = 0

    all_expert_inputs = []
    for layer in model.model.layers:
        moe_layer = cast(TinyOnnMoE, layer.mlp)
        if moe_layer.debug_expert_inputs:
            all_expert_inputs.extend(moe_layer.debug_expert_inputs)

    if not all_expert_inputs:
        print("No expert inputs were collected. Cannot perform analysis.")
        return

    # Allow non-leaf tensors to have grads
    grads = torch.autograd.grad(
        outputs=[loss],
        inputs=all_expert_inputs,
        grad_outputs=torch.ones_like(loss),
        allow_unused=True
    )

    for i, grad_tensor in enumerate(grads):
        if grad_tensor is not None:
            surprise = torch.linalg.norm(grad_tensor.flatten(start_dim=1), dim=1)
            total_surprise_norm += surprise.sum().item()
            num_experts_with_grad += 1
            print(f"  - Grad for expert input {i}: Shape={grad_tensor.shape}, Norm={torch.linalg.norm(grad_tensor).item():.4f}")
        else:
            print(f"  - Grad for expert input {i}: None")

    print("\nAnalysis complete.")
    print(f"Total experts with non-None gradients: {num_experts_with_grad} / {len(all_expert_inputs)}")
    print(f"Total surprise norm calculated: {total_surprise_norm:.4f}")

    assert num_experts_with_grad > 0, "No gradients were calculated for any expert inputs!"
    assert total_surprise_norm > 0, "Total surprise norm is zero!"
    print("\n✅ PoC successful. Gradients are flowing to expert inputs.")
