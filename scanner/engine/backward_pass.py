from functools import partial

import torch
from bitsandbytes.nn import Linear4bit
from tqdm import tqdm
from transformers.cache_utils import DynamicCache


def _capture_backward_hook(module, grad_input, grad_output, data_dict_for_token):
    param_name = module.param_name
    if grad_output[0] is not None and param_name in data_dict_for_token:
        grad = grad_output[0]
        num_features = grad.shape[-1]
        block_size = getattr(module, "blocksize", 64)
        num_blocks = (num_features + block_size - 1) // block_size

        block_grads = [
            torch.norm(
                grad[..., i * block_size : min((i + 1) * block_size, num_features)].float(),
                p=2,
            ).item()
            for i in range(num_blocks)
        ]

        if "gradient" not in data_dict_for_token[param_name]:
            data_dict_for_token[param_name]["gradient"] = [0.0] * num_blocks

        for i, grad_val in enumerate(block_grads):
            data_dict_for_token[param_name]["gradient"][i] = grad_val


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

    # Initial forward pass on the prompt
    with torch.no_grad():
        prompt_ids = full_sequence_ids[:prompt_len].unsqueeze(0)
        outputs = model(prompt_ids, use_cache=True)
        # Convert the tuple-based kv-cache to a DynamicCache object
        past_key_values = DynamicCache.from_legacy_cache(outputs.past_key_values)

    for i in tqdm(range(num_generated_tokens), desc="Per-Token Backward Pass"):
        model.zero_grad()

        handles = []
        current_token_data_dict = per_token_activation_data[i]
        for module in hook_modules:
            handle = module.register_full_backward_hook(
                partial(
                    _capture_backward_hook, data_dict_for_token=current_token_data_dict
                )
            )
            handles.append(handle)

        try:
            # We only need to process the single next token
            current_token_id = full_sequence_ids[prompt_len + i].unsqueeze(0).unsqueeze(0)
            labels = current_token_id.clone()

            outputs = model(
                input_ids=current_token_id,
                past_key_values=past_key_values,
                labels=labels,
                use_cache=True,
            )
            loss = outputs.loss
            # The past_key_values object is updated in-place. We need to detach its tensors from the graph.
            if outputs.past_key_values is not None:
                for i in range(len(outputs.past_key_values.key_cache)):
                    if outputs.past_key_values.key_cache[i] is not None:
                        outputs.past_key_values.key_cache[i] = outputs.past_key_values.key_cache[i].detach()
                    if outputs.past_key_values.value_cache[i] is not None:
                        outputs.past_key_values.value_cache[i] = outputs.past_key_values.value_cache[i].detach()
            past_key_values = outputs.past_key_values

            if loss is not None and loss.requires_grad:
                loss.backward()

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
