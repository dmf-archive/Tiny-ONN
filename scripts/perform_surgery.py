import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from tiny_onn.config import TinyOnnConfig
from tiny_onn.modular import TinyOnnForCausalLM


def perform_surgery(
    base_model_name: str, cache_dir: str, **kwargs
) -> tuple[TinyOnnForCausalLM, AutoTokenizer]:
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, cache_dir=cache_dir, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, cache_dir=cache_dir, trust_remote_code=True
    )
    base_config = base_model.config
    tiny_onn_config_dict = base_config.to_dict()
    tiny_onn_config_dict.update(kwargs)
    config = TinyOnnConfig(**tiny_onn_config_dict)
    model = TinyOnnForCausalLM(config)
    model.load_state_dict(base_model.state_dict(), strict=False)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Perform surgery on a base model to create a TinyOnn model."
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        required=True,
        help="The name of the base model to perform surgery on.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="The path to save the surgically modified model.",
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=32,
        help="Number of experts in the MoE layers.",
    )
    parser.add_argument(
        "--moe_intermediate_size",
        type=int,
        default=64,
        help="Intermediate size of the MoE layers.",
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
        num_experts=args.num_experts,
        moe_intermediate_size=args.moe_intermediate_size,
        num_experts_per_tok=args.num_experts_per_tok,
    )

    print(f"Saving surgically modified model and tokenizer to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Surgery complete.")


if __name__ == "__main__":
    main()
