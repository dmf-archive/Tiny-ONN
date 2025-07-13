import gc
import os
from collections import OrderedDict

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

from tinyonn.configuration_tinyonn import TinyONNConfig


def convert_qwen_to_tinyonn(model_name: str, output_dir: str, cache_dir: str):
    """
    Loads a Qwen3 state_dict, converts it to a TinyONN MoE state_dict,
    and saves the new state_dict and a compatible config.
    """
    print(f"Loading original model config: {model_name}")
    original_config = AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir
    )

    # --- Create a new config for TinyONN ---
    print("Creating new TinyONN config...")
    tinyonn_config = TinyONNConfig(
        vocab_size=original_config.vocab_size,
        hidden_size=original_config.hidden_size,
        intermediate_size=original_config.intermediate_size,
        num_hidden_layers=original_config.num_hidden_layers,
        num_attention_heads=original_config.num_attention_heads,
        num_key_value_heads=original_config.num_key_value_heads,
        hidden_act=original_config.hidden_act,
        max_position_embeddings=original_config.max_position_embeddings,
        initializer_range=original_config.initializer_range,
        rms_norm_eps=original_config.rms_norm_eps,
        use_cache=original_config.use_cache,
        tie_word_embeddings=getattr(original_config, 'tie_word_embeddings', False),
        rope_theta=original_config.rope_theta,
        attention_dropout=original_config.attention_dropout,
        num_local_experts=88, # Changed from 96 to 88 for perfect divisibility with intermediate_size=5632
        num_experts_per_tok=8,
    )
    # Explicitly set model_type to be absolutely certain, and add debug print.
    tinyonn_config.model_type = "tinyonn"
    tinyonn_config.architectures = ["TinyONNForCausalLM"]
    print(f"--- !!! SAVING config with model_type = '{tinyonn_config.model_type}' !!! ---")

    print("Loading original model state_dict...")
    # Load the original model to get its state_dict
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16
    )
    original_state_dict = original_model.state_dict()
    del original_model # Free up memory
    gc.collect()

    print("Starting state_dict conversion...")
    new_state_dict = OrderedDict()
    processed_layers = set()

    # First, copy all non-MLP weights
    for key, value in original_state_dict.items():
        if "mlp" not in key:
            new_state_dict[key] = value

    # Then, process each layer's MLP weights only once
    for layer_idx in range(tinyonn_config.num_hidden_layers):
        if layer_idx in processed_layers:
            continue

        print(f"  - Converting MLP for layer {layer_idx}")

        # Get all three original weight tensors for this layer
        gate_proj_w = original_state_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"]
        up_proj_w = original_state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"]
        down_proj_w = original_state_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"]

        # Chunk all three weight tensors for the experts
        # gate_proj and up_proj are chunked along the output dimension (dim=0)
        gate_chunks = torch.chunk(gate_proj_w, tinyonn_config.num_local_experts, dim=0)
        up_chunks = torch.chunk(up_proj_w, tinyonn_config.num_local_experts, dim=0)
        # down_proj is chunked along the input dimension (dim=1)
        down_chunks = torch.chunk(down_proj_w, tinyonn_config.num_local_experts, dim=1)

        # Create the new keys for the self-contained experts and assign the chunked weights
        for i in range(tinyonn_config.num_local_experts):
            # Call .contiguous() to ensure the tensor chunks are stored in a contiguous memory layout,
            # which is required by the safetensors library.
            new_state_dict[f"model.layers.{layer_idx}.mlp.experts.{i}.gate_proj.weight"] = gate_chunks[i].contiguous()
            new_state_dict[f"model.layers.{layer_idx}.mlp.experts.{i}.up_proj.weight"] = up_chunks[i].contiguous()
            new_state_dict[f"model.layers.{layer_idx}.mlp.experts.{i}.down_proj.weight"] = down_chunks[i].contiguous()

        # The centralized down_proj is no longer needed.

        processed_layers.add(layer_idx)

    print("State_dict conversion complete.")

    # --- Save the converted model's state_dict and config ---
    os.makedirs(output_dir, exist_ok=True)

    # Handle tied weights for safetensors.
    # If embeddings are tied, the state_dict should not contain the lm_head.weight,
    # as it's a pointer to the embed_tokens.weight. The model's post_init logic
    # will handle the tying automatically based on the config flag.
    if tinyonn_config.tie_word_embeddings and "lm_head.weight" in new_state_dict:
        print("Removing 'lm_head.weight' from state_dict due to tied embeddings.")
        del new_state_dict["lm_head.weight"]

    # Save the new state dictionary using the safetensors format
    output_path = os.path.join(output_dir, "model.safetensors")
    save_file(new_state_dict, output_path)
    print(f"Converted TinyONN state_dict saved to: {output_path}")

    # Save the modified config
    config_path = os.path.join(output_dir, "config.json")
    tinyonn_config.to_json_file(config_path)
    print(f"Model config saved to: {config_path}")


if __name__ == "__main__":
    QWEN_MODEL_PATH = "Qwen/Qwen3-1.7B"
    CACHE_DIR = "weights"
    OUTPUT_WEIGHT_DIR = os.path.join(CACHE_DIR, "tinyonn_converted_statedict")

    # Ensure the output directory is clean before conversion
    if os.path.exists(OUTPUT_WEIGHT_DIR):
        import shutil
        print(f"Removing existing converted directory: {OUTPUT_WEIGHT_DIR}")
        shutil.rmtree(OUTPUT_WEIGHT_DIR)

    convert_qwen_to_tinyonn(QWEN_MODEL_PATH, OUTPUT_WEIGHT_DIR, CACHE_DIR)
