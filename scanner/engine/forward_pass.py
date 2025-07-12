from typing import Any

import torch
import torch.nn.functional as F
from bitsandbytes.nn import Linear4bit
from transformers import BatchEncoding, PreTrainedTokenizer


def _capture_block_activations_hook(name: str, activation_data: dict, x: torch.Tensor):
    sequence_length = x.shape[1]
    num_features = x.shape[-1]
    block_size = 64
    num_blocks = (num_features + block_size - 1) // block_size

    token_block_norms = []
    for i in range(sequence_length):
        token_slice = x[:, i, :]
        padded_x = F.pad(token_slice, (0, num_blocks * block_size - num_features))
        reshaped_x = padded_x.view(x.shape[0], num_blocks, block_size)
        block_norms = torch.norm(reshaped_x.float(), p=2, dim=-1)
        token_block_norms.append(block_norms.squeeze(0).tolist())

    activation_data[name] = {"activation": token_block_norms}


def run_forward_pass_and_capture_activations(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 256,
) -> tuple[str, list[dict[str, Any]], torch.Tensor, int]:
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    # Ensure inputs is a BatchEncoding and move to device
    if isinstance(inputs, BatchEncoding):
        inputs = inputs.to(model.device)
    else:
        # This case should ideally not happen with return_dict=True
        raise TypeError("Tokenizer did not return a BatchEncoding.")


    # Generate the full response first
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_sequence_ids = outputs[0]
    prompt_len = inputs["input_ids"].shape[1]
    final_response = tokenizer.decode(full_sequence_ids[prompt_len:], skip_special_tokens=True)

    # Now, run a single forward pass on the complete sequence to capture activations
    activation_data: dict[str, dict[str, list[list[float]]]] = {}
    hooks: list[Any] = []

    def get_hook(name):
        return lambda model, input, output: _capture_block_activations_hook(
            name, activation_data, input[0]
        )

    for name, module in model.named_modules():
        if isinstance(module, Linear4bit):
            hooks.append(module.register_forward_hook(get_hook(name)))

    with torch.no_grad():
        _ = model(full_sequence_ids.unsqueeze(0))

    for handle in hooks:
        handle.remove()

    # Reformat the captured data to be per-token
    num_generated_tokens = full_sequence_ids.shape[0] - prompt_len
    per_token_data: list[dict[str, Any]] = [{} for _ in range(num_generated_tokens)]

    for param_name, data in activation_data.items():
        block_activations_per_token = data["activation"]
        for i in range(num_generated_tokens):
            token_idx_in_sequence = prompt_len + i
            if param_name not in per_token_data[i]:
                per_token_data[i][param_name] = {}
            per_token_data[i][param_name]["activation"] = block_activations_per_token[
                token_idx_in_sequence
            ]

    return final_response, per_token_data, full_sequence_ids, prompt_len
