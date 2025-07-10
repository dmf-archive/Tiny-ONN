from functools import partial

import torch
from bitsandbytes.nn import Linear4bit
from tqdm import tqdm

from utils.logging_utils import log_message


def _capture_backward_hook(module, grad_input, grad_output, data_dict_for_token):
    param_name = module.param_name
    if grad_output[0] is not None and param_name in data_dict_for_token:
        grad = grad_output[0]
        num_features = grad.shape[-1]
        block_size = getattr(module, 'blocksize', 64)
        num_blocks = (num_features + block_size - 1) // block_size
        
        block_grads = [
            torch.norm(grad[..., i*block_size:min((i+1)*block_size, num_features)].float(), p=2).item()
            for i in range(num_blocks)
        ]
        
        if "gradient" not in data_dict_for_token[param_name]:
             data_dict_for_token[param_name]["gradient"] = [0.0] * num_blocks
        
        for i, grad_val in enumerate(block_grads):
            data_dict_for_token[param_name]["gradient"][i] = grad_val

def run_per_token_backward_pass(model, full_sequence_ids, per_token_activation_data, prompt_len):
    model.train()
    
    hook_modules = []
    for name, module in model.named_modules():
        if isinstance(module, Linear4bit):
            module.param_name = name
            hook_modules.append(module)

    num_generated_tokens = len(per_token_activation_data)
    log_message(f"Starting per-token backward pass for {num_generated_tokens} generated tokens.")

    for i in tqdm(range(num_generated_tokens), desc="Per-Token Backward Pass"):
        model.zero_grad()
        
        handles = []
        current_token_data_dict = per_token_activation_data[i]
        for module in hook_modules:
            handle = module.register_full_backward_hook(
                partial(_capture_backward_hook, data_dict_for_token=current_token_data_dict)
            )
            handles.append(handle)

        try:
            # The sequence for this step includes the full prompt + generated tokens up to current_token_idx
            current_sequence_len = prompt_len + i + 1
            input_ids_for_step = full_sequence_ids[:current_sequence_len].unsqueeze(0)
            
            # Labels are set for the last token in the sequence to calculate loss
            labels = input_ids_for_step.clone()
            labels[:, :-1] = -100 # Ignore all tokens except the last one for loss calculation

            # Only perform backward pass if there's a token to predict (i.e., after the prompt)
            if input_ids_for_step.shape[1] <= prompt_len:
                 for handle in handles: handle.remove()
                 continue

            outputs = model(input_ids=input_ids_for_step, labels=labels)
            loss = outputs.loss
            
            if loss is not None and loss.requires_grad:
                loss.backward()
        
        finally:
            for handle in handles:
                handle.remove()

    log_message("Per-token backward pass completed.")
    model.eval()
    for module in hook_modules:
        if hasattr(module, 'param_name'):
            del module.param_name
            
    return per_token_activation_data
