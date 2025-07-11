from functools import partial
import re # Import re module

import torch
from bitsandbytes.nn import Linear4bit
from tqdm import tqdm
from models.pruned_layers import PrunedQwen3DecoderLayer

from utils.logging_utils import log_message


def _capture_backward_hook(capture_state, module, grad_input, grad_output):
    data_dict_for_token = capture_state.get('capture_dict')
    if data_dict_for_token is None:
        return

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
            is_pruned_module = False
            match = re.search(r'model\.layers\.(\d+)\.(self_attn|mlp)', name)
            if match:
                layer_idx = int(match.group(1))
                parent_name = ".".join(name.split('.')[:-2])
                if parent_name:
                    parent_module = model.get_submodule(parent_name)
                    if isinstance(parent_module, PrunedQwen3DecoderLayer):
                        module_type = match.group(2)
                        if module_type == "self_attn" and parent_module.prune_self_attn:
                            is_pruned_module = True
                        elif module_type == "mlp" and parent_module.prune_mlp:
                            is_pruned_module = True
            if not is_pruned_module:
                module.param_name = name
                hook_modules.append(module)

    num_generated_tokens = len(per_token_activation_data)
    log_message(f"Starting per-token backward pass for {num_generated_tokens} generated tokens.")

    capture_state = {'capture_dict': None}
    handles = []
    try:
        for module in hook_modules:
            handle = module.register_full_backward_hook(
                partial(_capture_backward_hook, capture_state)
            )
            handles.append(handle)

        for i in tqdm(range(num_generated_tokens), desc="Per-Token Backward Pass (Optimized)"):
            model.zero_grad()
            
            current_token_data_dict = per_token_activation_data[i]
            capture_state['capture_dict'] = current_token_data_dict

            current_sequence_len = prompt_len + i + 1
            input_ids_for_step = full_sequence_ids[:current_sequence_len].unsqueeze(0)
            
            if input_ids_for_step.shape[1] <= prompt_len:
                 continue

            labels = input_ids_for_step.clone()
            labels[:, :-1] = -100

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
