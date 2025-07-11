import argparse
import gc
import os
import sys
import torch
import shutil
from safetensors.torch import save_file
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def log_message(message: str):
    print(f"LOG: {message}")


def prune_and_save_model(
    model_name="Qwen/Qwen3-1.7B",
    output_dir="weights",
    pruning_mode="physical",
):
    log_message(
        f"Starting pruning process for model: {model_name} with mode: {pruning_mode}"
    )

    if pruning_mode == "physical":
        from common.models.pruned_layers import PrunedQwen3ForCausalLM, PrunedQwen3DecoderLayer
        model_class = PrunedQwen3ForCausalLM
        decoder_layer_class = PrunedQwen3DecoderLayer
        model_definition_filename = "pruned_layers.py"
        log_message("Using PHYSICAL pruning mode.")
    elif pruning_mode == "functional":
        from common.models.func_pruned_layers import FuncPrunedQwen3ForCausalLM, FuncPrunedQwen3DecoderLayer
        model_class = FuncPrunedQwen3ForCausalLM
        decoder_layer_class = FuncPrunedQwen3DecoderLayer
        model_definition_filename = "func_pruned_layers.py"
        log_message("Using FUNCTIONAL pruning mode.")
    else:
        raise ValueError(f"Unknown pruning mode: {pruning_mode}")

    cache_path = os.path.join(os.getcwd(), output_dir)
    pruned_model_name = f"{model_name.replace('/', '--')}-pruned-{pruning_mode}"
    pruned_model_path = os.path.join(cache_path, pruned_model_name)

    if os.path.exists(pruned_model_path):
        log_message(f"Removing existing pruned model directory at {pruned_model_path}.")
        shutil.rmtree(pruned_model_path)

    os.makedirs(pruned_model_path, exist_ok=True)
    log_message(f"Created output directory: {pruned_model_path}")

    log_message(f"Loading NATIVE model {model_name} for monkey-patching...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    
    log_message("Applying pruning to model layers via monkey-patching...")
    for i, layer in enumerate(model.model.layers):
        pruned_layer = decoder_layer_class(model.config, i).to("cpu")
        pruned_layer.load_state_dict(layer.state_dict())
        model.model.layers[i] = pruned_layer
    log_message("Monkey-patching complete.")

    # After patching, the model object is now correctly configured in memory.
    # We can directly save its state_dict and config.
    pruned_state_dict = model.state_dict()
    log_message(f"State_dict size: {len(pruned_state_dict)}.")

    if (
        "lm_head.weight" in pruned_state_dict
        and "model.embed_tokens.weight" in pruned_state_dict
        and pruned_state_dict["lm_head.weight"].data_ptr()
        == pruned_state_dict["model.embed_tokens.weight"].data_ptr()
    ):
        log_message(
            "Tied weights detected (lm_head and embed_tokens). Cloning lm_head.weight to save."
        )
        pruned_state_dict["lm_head.weight"] = pruned_state_dict["lm_head.weight"].clone()

    pruned_safetensors_path = os.path.join(pruned_model_path, "model.safetensors")
    log_message(f"Saving state_dict to {pruned_safetensors_path}...")
    save_file(pruned_state_dict, pruned_safetensors_path)

    custom_model_file_src = os.path.join(
        os.path.dirname(__file__), "..", "common", "models", model_definition_filename
    )
    custom_model_file_dst = os.path.join(pruned_model_path, model_definition_filename)
    shutil.copyfile(custom_model_file_src, custom_model_file_dst)
    log_message(f"Copied custom model definition to {custom_model_file_dst}")

    log_message("Saving updated config...")
    model.config.architectures = [model_class.__name__]
    model.config.model_type = "qwen3"
    model.config.auto_map = {
        "AutoModelForCausalLM": f"{os.path.splitext(model_definition_filename)[0]}.{model_class.__name__}",
        "AutoModel": f"{os.path.splitext(model_definition_filename)[0]}.{model_class.mro()[1].__name__}",
    }
    model.config.save_pretrained(pruned_model_path)

    log_message("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_path)
    tokenizer.save_pretrained(pruned_model_path)

    log_message(f"Pruned model saved to {pruned_model_path}")

    del model, pruned_state_dict
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune a Hugging Face model and save it.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="The name of the model to prune.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="weights",
        help="The directory to save the pruned model.",
    )
    parser.add_argument(
        "--pruning_mode",
        type=str,
        default="physical",
        choices=["physical", "functional"],
        help="Pruning mode: 'physical' removes weights, 'functional' skips computation.",
    )
    args = parser.parse_args()

    prune_and_save_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        pruning_mode=args.pruning_mode,
    )
