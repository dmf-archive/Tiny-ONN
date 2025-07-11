import gc
import os
import sys

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

# Add project root to path to allow relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pruned_layers import PrunedQwen3DecoderLayer, PrunedQwen3ForCausalLM
from utils import log_message


def prune_and_save_model(model_name="Qwen/Qwen3-1.7B", output_dir="weights"):
    """
    Loads a model, prunes its weights by removing specified layers from the state_dict,
    and saves the pruned state_dict and config. This creates a smaller model on disk
    that can be loaded into a full model skeleton with `strict=False`.
    """
    log_message(f"Starting weight pruning process for model: {model_name}")

    cache_path = os.path.join(os.getcwd(), output_dir)
    pruned_model_name = f"{model_name.replace('/', '--')}-pruned"
    pruned_model_path = os.path.join(cache_path, pruned_model_name)

    if os.path.exists(pruned_model_path):
        log_message(f"Pruned model directory already exists at {pruned_model_path}. Skipping.")
        return

    os.makedirs(pruned_model_path, exist_ok=True)
    log_message(f"Created output directory: {pruned_model_path}")

    log_message(f"Loading full model skeleton using PrunedQwen3ForCausalLM for: {model_name}")
    # Load the model with our custom architecture. This creates the full skeleton.
    # We need the full skeleton to know which layers to prune based on the flags.
    model = PrunedQwen3ForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Load to CPU to avoid GPU memory issues, we only need the state_dict
        trust_remote_code=True,
    )

    original_state_dict = model.state_dict()
    pruned_state_dict = {}

    keys_to_prune = set()

    # Identify keys to prune based on the boolean flags in each layer
    log_message("Identifying layers to prune from the state_dict...")
    for i, layer in enumerate(model.model.layers):
        if isinstance(layer, PrunedQwen3DecoderLayer):
            if layer.prune_self_attn:
                # All parameters of self_attn in this layer should be pruned
                attn_prefix = f"model.layers.{i}.self_attn."
                keys_to_prune.update(k for k in original_state_dict if k.startswith(attn_prefix))
                log_message(f"  - Pruning self_attn for layer {i}")

            if layer.prune_mlp:
                # All parameters of mlp in this layer should be pruned
                mlp_prefix = f"model.layers.{i}.mlp."
                keys_to_prune.update(k for k in original_state_dict if k.startswith(mlp_prefix))
                log_message(f"  - Pruning mlp for layer {i}")

    # Create the new state_dict without the pruned keys
    for key, value in original_state_dict.items():
        if key not in keys_to_prune:
            pruned_state_dict[key] = value

    log_message(f"Pruning complete. Original state_dict size: {len(original_state_dict)}. Pruned size: {len(pruned_state_dict)}.")

    # Handle shared weights (e.g., tied embeddings and lm_head) before saving.
    # The safetensors library requires tensors to own their own memory.
    if ("lm_head.weight" in pruned_state_dict and
        "model.embed_tokens.weight" in pruned_state_dict and
        pruned_state_dict["lm_head.weight"].data_ptr() == pruned_state_dict["model.embed_tokens.weight"].data_ptr()):
        log_message("Tied weights detected (lm_head and embed_tokens). Cloning lm_head.weight to save.")
        pruned_state_dict["lm_head.weight"] = pruned_state_dict["lm_head.weight"].clone()

    # Save the pruned state_dict
    pruned_safetensors_path = os.path.join(pruned_model_path, "model.safetensors")
    log_message(f"Saving pruned state_dict to {pruned_safetensors_path}...")
    save_file(pruned_state_dict, pruned_safetensors_path)

    # Update the config to point to the correct architecture and save it
    log_message("Saving updated config...")
    model.config.architectures = [PrunedQwen3ForCausalLM.__name__]
    model.config.save_pretrained(pruned_model_path)

    # Save the tokenizer
    log_message("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_path)
    tokenizer.save_pretrained(pruned_model_path)

    log_message(f"Pruned model weights and config saved to {pruned_model_path}")

    # Clean up memory
    del model, original_state_dict, pruned_state_dict
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prune a Hugging Face model and save it.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B", help="The name of the model to prune.")
    parser.add_argument("--output_dir", type=str, default="weights", help="The directory to save the pruned model.")
    args = parser.parse_args()

    prune_and_save_model(model_name=args.model_name, output_dir=args.output_dir)
