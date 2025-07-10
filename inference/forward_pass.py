from functools import partial
from typing import Dict, List

import torch
from bitsandbytes.nn import Linear4bit
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from utils.logging_utils import log_debug


def _patched_forward(module, activation_data, original_forward, *args, **kwargs):
    """Patched forward method to capture block-level activation metrics for a single token pass."""
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
            activation_data[param_name]["weight"] = [weight_values[0]] * num_blocks if weight_values else [0.0] * num_blocks

    return original_forward(*args, **kwargs)

def run_forward_pass_and_capture_activations(model, tokenizer: PreTrainedTokenizer, user_message: str, history: list):
    """
    Runs a token-by-token forward pass to generate a response, showing a progress bar
    and capturing block-level activation data at each step.
    """
    messages = [{"role": "user", "content": user_message}]
    # This logic is flawed for the new message format.
    # The history is already in the correct format.
    # Let's just combine them.
    if history:
        messages = history + messages

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    input_ids = model_inputs.input_ids
    
    # --- Dynamic Patching ---
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, Linear4bit):
            module.param_name = name
            hooks.append((module, module.forward)) # Store original forward

    # --- Token-by-token Generation with TQDM ---
    generated_ids = []
    per_token_block_data = []
    past_key_values = None
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)

        for _ in tqdm(range(256), desc="Generating Tokens"):
            activation_data_for_step: Dict[str, Dict[str, List[float]]] = {}
            
            patched_forwards = []
            for module, original_forward in hooks:
                patched_forward = partial(_patched_forward, module, activation_data_for_step, original_forward)
                module.forward = patched_forward
                patched_forwards.append((module, original_forward))

            outputs = model(input_ids=next_token_id, past_key_values=past_key_values, use_cache=True)
            
            for module, original_forward in patched_forwards:
                module.forward = original_forward

            per_token_block_data.append(activation_data_for_step)
            
            past_key_values = outputs.past_key_values
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
            
            token_id = next_token_id.item()
            generated_ids.append(token_id)
            
            if token_id == tokenizer.eos_token_id:
                break
    
    # --- Unpatching (final cleanup) ---
    for module, original_forward in hooks:
        module.forward = original_forward
        if hasattr(module, 'param_name'):
            del module.param_name

    final_response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_sequence_ids_with_generation = torch.cat([input_ids[0], torch.tensor(generated_ids, device=model.device)])
    prompt_len = input_ids.shape[1]

    return final_response, per_token_block_data, full_sequence_ids_with_generation, prompt_len
