import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def run_benchmark():
    model_name = "Qwen/Qwen3-0.6B"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    script_dir = os.path.dirname(os.path.realpath(__file__))
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "weights")
    os.makedirs(cache_dir, exist_ok=True)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    print(f"Loading model: {model_name} with NF4 quantization")
    print(f"Forcing device: {device}")
    print(f"Using cache directory: {cache_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quantization_config, cache_dir=cache_dir
    ).to(device)

    print("\n" + "=" * 50)
    print("Model Configuration:")
    print(model.config)
    print("=" * 50 + "\n")

    num_hidden_layers = getattr(model.config, "num_hidden_layers", "N/A")
    print(f"Verified Number of Hidden Layers: {num_hidden_layers}\n")

    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)

    print("Warming up...")
    _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    print("Starting benchmark...")
    start_time = time.time()

    num_tokens_to_generate = 512
    generated_output = model.generate(
        **inputs,
        max_new_tokens=num_tokens_to_generate,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    end_time = time.time()

    duration = end_time - start_time
    generated_tokens = len(generated_output[0]) - len(inputs["input_ids"][0])
    tokens_per_second = generated_tokens / duration

    print("\n" + "=" * 50)
    print("Benchmark Results:")
    print(f"Generated {generated_tokens} tokens in {duration:.2f} seconds.")
    print(f"Inference Speed: {tokens_per_second:.2f} tokens/second")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_benchmark()
