from functools import partial

import torch
from bitsandbytes.nn import Linear4bit
from tqdm import tqdm
from transformers.cache_utils import DynamicCache

_global_grad_storage = {}


def _capture_backward_hook(module, grad_input, grad_output, param_name):
    if grad_output[0] is not None:
        grad = grad_output[0].float()
        num_features = grad.shape[-1]
        block_size = getattr(module, "blocksize", 64)

        padding_size = (block_size - (num_features % block_size)) % block_size
        if padding_size > 0:
            grad = torch.nn.functional.pad(grad, (0, padding_size))

        reshaped_grad = grad.view(-1, block_size)
        block_grads = torch.linalg.vector_norm(reshaped_grad, ord=2, dim=-1)

        num_blocks = (num_features + block_size - 1) // block_size
        final_grads = block_grads[:num_blocks]

        _global_grad_storage[param_name] = final_grads


def run_per_token_backward_pass(
    model, full_sequence_ids, per_token_activation_data, prompt_len
):
    model.train()

    hook_modules = []
    for name, module in model.named_modules():
        if isinstance(module, Linear4bit):
            module.param_name = name
            hook_modules.append(module)

    num_generated_tokens = len(per_token_activation_data)
    print(
        f"Starting per-token backward pass for {num_generated_tokens} generated tokens."
    )

    with torch.no_grad():
        prompt_ids = full_sequence_ids[:prompt_len].unsqueeze(0)
        outputs = model(prompt_ids, use_cache=True)
        past_key_values = DynamicCache.from_legacy_cache(outputs.past_key_values)

    handles = []
    for module in hook_modules:
        handle = module.register_full_backward_hook(
            partial(_capture_backward_hook, param_name=module.param_name)
        )
        handles.append(handle)

    try:
        for i in tqdm(range(num_generated_tokens), desc="Per-Token Backward Pass"):
            model.zero_grad()
            _global_grad_storage.clear()

            current_token_id = full_sequence_ids[prompt_len + i].unsqueeze(0).unsqueeze(0)
            labels = current_token_id.clone()

            outputs = model(
                input_ids=current_token_id,
                past_key_values=past_key_values,
                labels=labels,
                use_cache=True,
            )
            loss = outputs.loss

            if loss is not None and loss.requires_grad:
                loss.backward()

            for param_name, grads in _global_grad_storage.items():
                if param_name in per_token_activation_data[i]:
                    per_token_activation_data[i][param_name]["gradient"] = grads.tolist()

            if outputs.past_key_values is not None:
                for layer_idx in range(len(outputs.past_key_values.key_cache)):
                    if outputs.past_key_values.key_cache[layer_idx] is not None:
                        outputs.past_key_values.key_cache[layer_idx] = outputs.past_key_values.key_cache[layer_idx].detach()
                    if outputs.past_key_values.value_cache[layer_idx] is not None:
                        outputs.past_key_values.value_cache[layer_idx] = outputs.past_key_values.value_cache[layer_idx].detach()
            past_key_values = outputs.past_key_values

            del loss
            del outputs
            torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()

    print("Per-token backward pass completed.")
    model.eval()
    for module in hook_modules:
        if hasattr(module, "param_name"):
            del module.param_name

    return per_token_activation_data


def run_full_sequence_backward_pass(
    model, full_sequence_ids, per_token_activation_data, prompt_len
):
    model.train()

    hook_modules = []
    for name, module in model.named_modules():
        if isinstance(module, Linear4bit):
            module.param_name = name
            hook_modules.append(module)

    print("Starting full sequence backward pass.")

    handles = []
    for module in hook_modules:
        handle = module.register_full_backward_hook(
            partial(_capture_backward_hook, param_name=module.param_name)
        )
        handles.append(handle)

    try:
        model.zero_grad()
        _global_grad_storage.clear()

        inputs = full_sequence_ids.unsqueeze(0)
        labels = inputs.clone()

        outputs = model(
            input_ids=inputs,
            labels=labels,
            use_cache=False,
        )
        loss = outputs.loss

        if loss is not None and loss.requires_grad:
            loss.backward()

        if per_token_activation_data:
            aggregated_data = per_token_activation_data[0]
            for param_name, grads in _global_grad_storage.items():
                if param_name in aggregated_data:
                    aggregated_data[param_name]["gradient"] = grads.tolist()

        del loss
        del outputs
        torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()

    print("Full sequence backward pass completed.")
    model.eval()
    for module in hook_modules:
        if hasattr(module, "param_name"):
            del module.param_name

    return per_token_activation_data
