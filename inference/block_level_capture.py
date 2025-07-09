from functools import partial

import torch
from bitsandbytes.nn import Linear4bit

from utils.logging_utils import log_debug, log_message

# Global dictionary to store captured block-level data for the current token
# Structure: {param_name: {"activation": [], "gradient": [], "weight": []}}
block_level_data = {}
# Global variable to hold the current token index being processed
current_token_idx = 0

def _get_param_name(module, model):
    """Helper function to find the name of a module."""
    for name, m in model.named_modules():
        if m is module:
            return name
    return None

def _capture_backward_hook(module, grad_input, grad_output, param_name):
    """
    Backward hook to capture block-level gradient norms.
    grad_output is a tuple, we are interested in the first element.
    """
    if grad_output[0] is not None:
        grad = grad_output[0]
        # Assuming the feature dimension is the last one
        num_features = grad.shape[-1]
        block_size = 64 # bitsandbytes block size
        num_blocks = (num_features + block_size - 1) // block_size
        
        block_grads = []
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, num_features)
            block = grad[..., start_idx:end_idx]
            block_norm = torch.norm(block.float(), p=2).item()
            block_grads.append(block_norm)
        
        if param_name not in block_level_data:
            block_level_data[param_name] = {}
        block_level_data[param_name]["gradient"] = block_grads
        # log_debug(f"Backward hook for {param_name} captured {len(block_grads)} gradient blocks.")

def patched_forward(module, original_forward, *args, **kwargs):
    """
    The patched forward method that captures block-level activation and weight metrics.
    """
    # --- Capture Activations ---
    # Input tensor 'x' is the first argument in 'args'
    x = args[0]
    param_name = module.param_name
    
    # Assuming the feature dimension is the last one
    num_features = x.shape[-1]
    block_size = 64 # bitsandbytes block size
    num_blocks = (num_features + block_size - 1) // block_size

    block_activations = []
    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = min((i + 1) * block_size, num_features)
        block = x[..., start_idx:end_idx]
        block_norm = torch.norm(block.float(), p=2).item()
        block_activations.append(block_norm)

    if param_name not in block_level_data:
        block_level_data[param_name] = {}
    block_level_data[param_name]["activation"] = block_activations
    # log_debug(f"Forward patch for {param_name} captured {len(block_activations)} activation blocks.")

    # --- Capture Weights ---
    if hasattr(module.weight, 'quant_state') and module.weight.quant_state is not None:
        absmax = module.weight.quant_state.absmax
        block_level_data[param_name]["weight"] = [absmax.item()]
        # log_debug(f"Forward patch for {param_name} captured {len(absmax)} weight absmax values.")

    # Call the original forward method to continue the model's computation
    return original_forward(*args, **kwargs)

def patch_model_for_block_level_capture(model):
    """
    Finds all Linear4bit layers in the model, patches their forward method,
    and registers backward hooks.
    """
    log_message("Starting model patching for block-level data capture...")
    for name, module in model.named_modules():
        if isinstance(module, Linear4bit):
            # Store param_name in the module for easy access in hooks
            module.param_name = name
            
            # 1. Patch the forward method
            original_forward = module.forward
            module.forward = partial(patched_forward, module, original_forward)
            
            # 2. Register a backward hook
            module.register_full_backward_hook(partial(_capture_backward_hook, param_name=name))
            
            log_debug(f"Patched {name}")
    log_message("Model patching complete.")

def get_and_clear_block_level_data():
    """
    Returns the current block-level data and clears the global dictionary for the next token.
    """
    global block_level_data
    data_to_return = block_level_data
    block_level_data = {}
    return data_to_return
