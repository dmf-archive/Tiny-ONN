import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tiny_onn.config import TinyOnnConfig
from tiny_onn.modular import TinyOnnForCausalLM


def perform_surgery(
    base_model_name: str, cache_dir: str, base_model: torch.nn.Module | None = None, **kwargs
) -> tuple[TinyOnnForCausalLM, AutoTokenizer]:
    if base_model is None:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, cache_dir=cache_dir, trust_remote_code=True
        )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, cache_dir=cache_dir, trust_remote_code=True
    )

    base_config = base_model.config
    if hasattr(base_config, "to_dict") and callable(base_config.to_dict):
        tiny_onn_config_dict = base_config.to_dict()
    else:
        tiny_onn_config_dict = vars(base_config)

    num_experts = kwargs.get("num_experts_per_layer", 32)
    if hasattr(base_config, "intermediate_size"):
        tiny_onn_config_dict["moe_intermediate_size"] = (
            base_config.intermediate_size // num_experts
        )

    tiny_onn_config_dict.update(kwargs)
    config = TinyOnnConfig(**tiny_onn_config_dict)

    tiny_onn_model = TinyOnnForCausalLM(config)

    base_state_dict = base_model.state_dict()
    tiny_onn_state_dict = tiny_onn_model.state_dict()

    for key in base_state_dict:
        if "mlp" not in key:
            tiny_onn_state_dict[key] = base_state_dict[key]

    for layer_idx in range(config.num_hidden_layers):
        gate_proj = base_state_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"]
        up_proj = base_state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"]
        down_proj = base_state_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"]

        gate_chunks = torch.chunk(gate_proj, config.num_experts_per_layer, dim=0)
        up_chunks = torch.chunk(up_proj, config.num_experts_per_layer, dim=0)
        down_chunks = torch.chunk(down_proj, config.num_experts_per_layer, dim=1)

        for expert_idx in range(config.num_experts_per_layer):
            tiny_onn_state_dict[
                f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.w1.weight"
            ] = gate_chunks[expert_idx]
            tiny_onn_state_dict[
                f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.w3.weight"
            ] = up_chunks[expert_idx]
            tiny_onn_state_dict[
                f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.w2.weight"
            ] = down_chunks[expert_idx]

    tiny_onn_model.load_state_dict(tiny_onn_state_dict, strict=False)

    return tiny_onn_model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Perform surgery on a base model to create a TinyOnn model."
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="The name of the base model to perform surgery on.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="The path to save the surgically modified model.",
    )
    parser.add_argument(
        "--num_experts_per_layer",
        type=int,
        default=32,
        help="Number of experts in the MoE layers.",
    )
    parser.add_argument(
        "--num_experts_per_tok",
        type=int,
        default=-1,
        help="Number of experts to use per token.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="weights",
        help="Directory to cache downloaded models.",
    )

    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Performing surgery on {args.base_model_name}...")
    model, tokenizer = perform_surgery(
        base_model_name=args.base_model_name,
        cache_dir=args.cache_dir,
        num_experts_per_layer=args.num_experts_per_layer,
        num_experts_per_tok=args.num_experts_per_tok,
    )

    print(f"Saving surgically modified model and tokenizer to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Surgery complete.")

    print("\n--- Verifying Surgery ---")
    print("Loading base model for comparison...")
    base_model_verify = AutoModelForCausalLM.from_pretrained(
        args.base_model_name, cache_dir=args.cache_dir, trust_remote_code=True
    )
    print("Loading surgically modified model...")
    surgical_model_verify = TinyOnnForCausalLM.from_pretrained(output_path, trust_remote_code=True)

    print("\nBase Model Architecture (Layer 0 MLP):")
    print(base_model_verify.model.layers[0].mlp)

    print("\nSurgical Model Architecture (Layer 0 MLP):")
    print(surgical_model_verify.model.layers[0].mlp)

    base_mlp_type = type(base_model_verify.model.layers[0].mlp)
    surgical_mlp_type = type(surgical_model_verify.model.layers[0].mlp)

    if base_mlp_type != surgical_mlp_type:
        print("\n✅ Verification PASSED: MLP layer types are different.")
        print(f"   - Base MLP Type: {base_mlp_type.__name__}")
        print(f"   - Surgical MLP Type: {surgical_mlp_type.__name__}")
    else:
        print(f"\n❌ Verification FAILED: MLP layer types are the same ({base_mlp_type.__name__}).")


if __name__ == "__main__":
    main()
