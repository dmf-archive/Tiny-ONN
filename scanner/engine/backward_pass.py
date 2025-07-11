from functools import partial
from typing import Any

import torch
from bitsandbytes.nn import Linear4bit
from tqdm import tqdm

from common.models.pruned_layers import PrunedQwen3DecoderLayer


def _capture_backward_hook(
    capture_state: dict[str, Any], module: torch.nn.Module, grad_input, grad_output
):
    if (
        (data_dict := capture_state.get("capture_dict"))
        and (grad := grad_output[0]) is not None
        and (param_name := module.param_name) in data_dict
    ):
        num_features = grad.shape[-1]
        block_size = getattr(module, "blocksize", 64)
        num_blocks = (num_features + block_size - 1) // block_size

        padded_grad = torch.nn.functional.pad(
            grad, (0, num_blocks * block_size - num_features)
        )
        reshaped_grad = padded_grad.view(*grad.shape[:-1], num_blocks, block_size)
        block_norms = torch.norm(reshaped_grad.float(), p=2, dim=-1)

        data_dict[param_name]["gradient"] = block_norms.flatten().tolist()


def run_per_token_backward_pass(
    model: torch.nn.Module,
    full_sequence_ids: torch.Tensor,
    per_token_activation_data: list[dict[str, Any]],
    prompt_len: int,
) -> list[dict[str, Any]]:
    model.train()

    hook_modules = []
    for name, module in model.named_modules():
        if isinstance(module, Linear4bit):
            parent_name = ".".join(name.split(".")[:-2])
            parent_module = model.get_submodule(parent_name) if parent_name else None
            if isinstance(parent_module, PrunedQwen3DecoderLayer) and (
                (parent_module.prune_self_attn and "self_attn" in name)
                or (parent_module.prune_mlp and "mlp" in name)
            ):
                continue
            module.param_name = name
            hook_modules.append(module)

    num_generated_tokens = len(per_token_activation_data)
    if num_generated_tokens == 0:
        model.eval()
        return per_token_activation_data

    print(f"LOG: Starting per-token backward pass for {num_generated_tokens} tokens.")

    capture_state: dict[str, Any] = {"capture_dict": None}
    handles = [
        m.register_full_backward_hook(partial(_capture_backward_hook, capture_state))
        for m in hook_modules
    ]

    try:
        logits = model(input_ids=full_sequence_ids.unsqueeze(0)).logits
        for i in tqdm(range(num_generated_tokens), desc="Per-Token Backward Pass"):
            model.zero_grad()
            capture_state["capture_dict"] = per_token_activation_data[i]
            token_idx_in_seq = prompt_len + i
            pred_logits = logits[:, token_idx_in_seq - 1, :]
            true_label = full_sequence_ids[token_idx_in_seq].unsqueeze(0)
            loss = torch.nn.functional.cross_entropy(pred_logits, true_label)
            if loss.requires_grad:
                loss.backward(retain_graph=(i < num_generated_tokens - 1))
    finally:
        for handle in handles:
            handle.remove()
        for module in hook_modules:
            if hasattr(module, "param_name"):
                delattr(module, "param_name")
        model.eval()

    print("LOG: Per-token backward pass completed.")
    return per_token_activation_data
