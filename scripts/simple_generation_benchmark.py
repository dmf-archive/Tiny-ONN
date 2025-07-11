import argparse
import gc
import os
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)


def run_simple_benchmark(model_path: str, batch_size: int, max_new_tokens: int):
    """
    Loads a model and runs a simple text generation benchmark.
    """
    print(f"--- Starting Simple Generation Benchmark for {model_path} ---")
    print(f"Batch Size: {batch_size}")
    print(f"Max New Tokens: {max_new_tokens}")

    # --- 1. Load Model and Tokenizer ---
    cache_path = os.path.join(os.getcwd(), "weights")
    os.makedirs(cache_path, exist_ok=True)
    print(f"Using cache directory: {cache_path}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir=cache_path)
    print("Tokenizer loaded.")

    print("Loading model with 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=cache_path,
    )
    model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
    model.generation_config.do_sample = False
    model.eval()
    print("Model loaded successfully.")

    # --- 2. Prepare Simple Input ---
    prompt = "Hello, my name is"
    prompts = [prompt] * batch_size
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    input_token_count = inputs.input_ids.shape[1]
    print(f"Input prepared. Input token count per sample: {input_token_count}")

    # --- 3. Run Generation and Time it ---
    print("Starting generation...")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize() # Wait for all kernels to finish before starting timer

    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id
        )

    torch.cuda.synchronize() # Wait for generation to finish
    end_time = time.time()
    
    duration = end_time - start_time
    generated_tokens_per_sample = outputs.shape[1] - input_token_count
    total_generated_tokens = generated_tokens_per_sample * batch_size

    print("\n--- Benchmark Results ---")
    print(f"Total time for generation: {duration:.4f} seconds")
    print(f"Total generated tokens: {total_generated_tokens}")
    print(f"Tokens per second: {total_generated_tokens / duration:.4f} tokens/s")
    print(f"Time per sample (for {generated_tokens_per_sample} tokens): {duration / batch_size:.4f} seconds/sample")
    print("-------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Simple text generation benchmark for models.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-1.7B", help="Path or name of the model to benchmark.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of new tokens to generate.")
    args = parser.parse_args()
    run_simple_benchmark(args.model_path, args.batch_size, args.max_new_tokens)

if __name__ == "__main__":
    main()
