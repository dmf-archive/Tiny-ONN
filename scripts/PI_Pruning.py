import gc
import json
import os
import re
import shutil

import bitsandbytes.nn as bnb_nn
import sys
import torch
import bitsandbytes.nn as bnb_nn
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

# Add project root to path to allow relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pruned_layers import PrunedQwen3DecoderLayer
from utils import log_message


def prune_and_save_model(model_name="Qwen/Qwen3-1.7B", output_dir="weights"):
    log_message(f"Starting pruning process for model: {model_name}")
    
    cache_path = os.path.join(os.getcwd(), output_dir)
    os.makedirs(cache_path, exist_ok=True)
    
    pruned_model_name = f"{model_name.replace('/', '--')}-pruned"
    pruned_model_path = os.path.join(cache_path, pruned_model_name)

    if os.path.exists(pruned_model_path):
        log_message(f"Pruned model already exists at {pruned_model_path}. Skipping pruning.")
        return

    log_message(f"Loading original model for pruning: {model_name}")
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        cache_dir=cache_path, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    config = original_model.config
    
    log_message("Pruning original model in-place...")
    for i in tqdm(range(config.num_hidden_layers), desc="Pruning layers"):
        original_model.model.layers[i] = PrunedQwen3DecoderLayer(config, i)
    
    model = original_model
    log_message("In-place pruning complete.")
    
    log_message("Manually saving pruned model state_dict...")
    os.makedirs(pruned_model_path, exist_ok=True)
    
    pruned_state_dict = {}
    total_layers = model.config.num_hidden_layers
    
    for name, param in model.named_parameters():
        match = re.match(r'model\.layers\.(\d+)\.(self_attn|mlp)\..*', name)
        if match:
            layer_idx = int(match.group(1))
            module_type = match.group(2)
            
            prune_self_attn = layer_idx < 3
            prune_mlp = layer_idx >= (total_layers // 2)

            if (module_type == "self_attn" and prune_self_attn) or \
               (module_type == "mlp" and prune_mlp):
                continue
        
        pruned_state_dict[name] = param

    if "lm_head.weight" in pruned_state_dict and "model.embed_tokens.weight" in pruned_state_dict:
        if torch.equal(pruned_state_dict["lm_head.weight"], pruned_state_dict["model.embed_tokens.weight"]):
            log_message("Detected tied weights. Removing lm_head.weight from state_dict for saving.")
            del pruned_state_dict["lm_head.weight"]

    save_file(pruned_state_dict, os.path.join(pruned_model_path, "model.safetensors"))

    model.config.save_pretrained(pruned_model_path)
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        model.generation_config.save_pretrained(pruned_model_path)
    else:
        log_message("Model does not have a generation_config to save.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_path)
    tokenizer.save_pretrained(pruned_model_path)
    
    log_message(f"Pruned model definition and tokenizer saved to {pruned_model_path}")
    log_message("Note: Quantization is now part of the loading process in the main script, not the pruning script.")
    
    # Clean up memory
    del model
    del original_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prune a Hugging Face model and save it.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B", help="The name of the model to prune.")
    parser.add_argument("--output_dir", type=str, default="weights", help="The directory to save the pruned model.")
    args = parser.parse_args()
    
    prune_and_save_model(model_name=args.model_name, output_dir=args.output_dir)
