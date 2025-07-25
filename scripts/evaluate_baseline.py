import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent))

from tiny_onn.modular import TinyOnnForCausalLM
from training.data import get_dataloaders


def evaluate_baseline(
    model_name: str,
    data_path: str,
    max_seq_length: int,
    batch_size: int,
    use_tiny_onn: bool,
    dynamic_k_inference: bool,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ModelClass = TinyOnnForCausalLM if use_tiny_onn else AutoModelForCausalLM

    load_kwargs = {"trust_remote_code": True, "cache_dir": "weights"}
    if use_tiny_onn:
        load_kwargs["ignore_mismatched_sizes"] = True

    model = ModelClass.from_pretrained(model_name, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        model_max_length=max_seq_length,
        cache_dir="weights",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    _, eval_dataloader = get_dataloaders(
        tokenizer=tokenizer,
        train_path=data_path, # Dummy path, not used
        val_path=data_path,
        batch_size=batch_size,
        num_workers=0,
        max_length=max_seq_length,
    )

    # This script now focuses on generation, not perplexity calculation.
    print(f"\n--- Generating Sample Completion for {model_name} ---")

    # Get a single sample from the validation data
    dataset = eval_dataloader.dataset
    sample = dataset.data[0]  # Access raw data

    # Extract only the user message
    user_message = [sample["messages"][0]]

    # Apply template to create the generation prompt
    prompt_text = tokenizer.apply_chat_template(
        user_message, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    print(f"Prompt: {prompt_text}")

    # Generate text
    generated_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=100,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=inputs["attention_mask"],
    )

    decoded_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print("\nGenerated Text:")
    print(decoded_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model's generation.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Name or path of the baseline model.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/dummy_chat_data.jsonl",
        help="Path to the validation data.",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=256, help="Maximum sequence length."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Evaluation batch size."
    )
    parser.add_argument(
        "--use_tiny_onn", action="store_true", help="Use TinyOnn model architecture."
    )
    parser.add_argument(
        "--dynamic_k_inference",
        action="store_true",
        help="Enable dynamic K inference for TinyOnn.",
    )

    args = parser.parse_args()
    evaluate_baseline(
        args.model_name,
        args.data_path,
        args.max_seq_length,
        args.batch_size,
        args.use_tiny_onn,
        args.dynamic_k_inference,
    )
