import os

import torch
from transformers import AutoTokenizer

from tiny_onn.surgery import perform_surgery


def generate_sample_text():
    base_model_name = "Qwen/Qwen3-0.6B"
    script_dir = os.path.dirname(os.path.realpath(__file__))
    cache_dir = os.path.join(os.path.dirname(script_dir), "weights")
    os.makedirs(cache_dir, exist_ok=True)

    print("Loading model and tokenizer...")
    model = perform_surgery(base_model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompt = "The quick brown fox jumps over the lazy dog"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print("\nGenerating text...")
    output_sequences = model.generate(
        input_ids=inputs["input_ids"],
        max_length=50,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )

    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    print("\n--- Generated Text ---")
    print(generated_text)
    print("----------------------")


if __name__ == "__main__":
    generate_sample_text()
