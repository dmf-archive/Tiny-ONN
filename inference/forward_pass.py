from functools import partial
from functools import partial
from typing import Dict, List
import re # Import re module

import torch
from bitsandbytes.nn import Linear4bit
from tqdm import tqdm
from models.pruned_layers import PrunedQwen3DecoderLayer
from transformers import PreTrainedTokenizer

from utils.logging_utils import log_debug


def _patched_forward(module, capture_state, original_forward, *args, **kwargs):
    """
    Patched forward method to capture block-level activation metrics.
    It uses a 'capture_state' dictionary to get the correct data dictionary for the current token.
    """
    activation_data = capture_state.get('capture_dict')
    if activation_data is None:
        # If the capture dict is not set, just run the original forward pass
        return original_forward(*args, **kwargs)

    x = args[0]
    param_name = module.param_name
    
    num_features = x.shape[-1]
    block_size = 64
    num_blocks = (num_features + block_size - 1) // block_size

    block_activations = [
        torch.norm(x[..., i*block_size:min((i+1)*block_size, num_features)].float(), p=2).item()
        for i in range(num_blocks)
    ]
    
    activation_data[param_name] = {"activation": block_activations}
    if hasattr(module.weight, 'quant_state') and module.weight.quant_state is not None:
        absmax = module.weight.quant_state.absmax
        weight_values = absmax.tolist()
        if len(weight_values) == num_blocks:
            activation_data[param_name]["weight"] = weight_values
        else:
            # Handle cases where absmax might not be per-block
            activation_data[param_name]["weight"] = [weight_values[0]] * num_blocks if weight_values else [0.0] * num_blocks

    return original_forward(*args, **kwargs)

def run_forward_pass_and_capture_activations(model, tokenizer: PreTrainedTokenizer, user_message: str, history: list):
    """
    Runs a token-by-token forward pass to generate a response, showing a progress bar
    and capturing block-level activation data at each step.
    This version optimizes hook registration by doing it only once.
    """
    messages = [{"role": "user", "content": user_message}]
    if history:
        messages = history + messages

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_ids = model_inputs.input_ids

    # --- Identify modules to hook ---
    hook_modules = []
    for name, module in model.named_modules():
        if isinstance(module, Linear4bit):
            is_pruned_module = False
            match = re.search(r'model\.layers\.(\d+)\.(self_attn|mlp)', name)
            if match:
                layer_idx = int(match.group(1))
                # Correctly identify the parent PrunedQwen3DecoderLayer
                parent_name = ".".join(name.split('.')[:-2]) 
                if parent_name: # Ensure parent_name is not empty
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

    # --- Token-by-token Generation with Optimized Hooking ---
    generated_ids = []
    per_token_block_data = []
    past_key_values = None
    model.eval()

    # This object will be passed to the hook and modified in-place
    capture_state = {'capture_dict': None}

    # Register hooks only ONCE
    original_forwards = []
    try:
        for module in hook_modules:
            original_forward = module.forward
            original_forwards.append((module, original_forward))
            # The hook's signature will be changed to accept capture_state
            module.forward = partial(_patched_forward, module, capture_state, original_forward)

        with torch.no_grad():
            # Process prompt
            outputs = model(input_ids=input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)

            # Generate tokens one by one
            for _ in tqdm(range(512), desc="Generating Tokens (Optimized Forward)"):
                # For each step, create a new dictionary to store activation data
                activation_data_for_step = {}
                capture_state['capture_dict'] = activation_data_for_step

                outputs = model(input_ids=next_token_id, past_key_values=past_key_values, use_cache=True)
                
                per_token_block_data.append(activation_data_for_step)
                
                past_key_values = outputs.past_key_values
                next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
                
                token_id = next_token_id.item()
                generated_ids.append(token_id)
                
                if token_id == tokenizer.eos_token_id:
                    break
    finally:
        # --- Unpatching (final cleanup) ---
        for module, original_forward in original_forwards:
            module.forward = original_forward
            if hasattr(module, 'param_name'):
                del module.param_name

    final_response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_sequence_ids_with_generation = torch.cat([input_ids[0], torch.tensor(generated_ids, device=model.device)])
    prompt_len = input_ids.shape[1]

    return final_response, per_token_block_data, full_sequence_ids_with_generation, prompt_len
