from typing import cast

import torch
from transformers import AutoTokenizer

from tiny_onn.modular import TinyOnnForCausalLM, TinyOnnMoE


def test_autograd_surprise_calculation():
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

    surprise_context: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    # Set a default surprise_budget for testing, even if not in dyn_k mode
    outputs = model(
        input_ids=input_ids,
        labels=labels,
        surprise_context=surprise_context,
        surprise_budget=0.5
    )
    loss = outputs.loss
    assert loss is not None

    loss.backward()

    total_routed_tokens_in_forward = 0
    for layer in model.model.layers:
        moe_layer = cast(TinyOnnMoE, layer.mlp)
        if moe_layer.last_expert_token_indices:
            for indices in moe_layer.last_expert_token_indices.values():
                total_routed_tokens_in_forward += len(indices)

    calculated_surprises = 0
    for _, (_, surprise) in surprise_context.items():
        calculated_surprises += surprise.numel()

    print("\n--- V12.1 Autograd-based Surprise Calculation ---")
    print(f"Total number of token-expert routes in forward pass: {total_routed_tokens_in_forward}")
    print(f"Total number of surprise values calculated via autograd: {calculated_surprises}")

    assert calculated_surprises > 0, "No surprise values were calculated!"
    assert calculated_surprises == total_routed_tokens_in_forward, \
        "Mismatch between calculated surprises and routed tokens!"

    print("\n✅ PoC successful. Per-token surprise calculated correctly via custom autograd.")
