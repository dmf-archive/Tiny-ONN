import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent))

from tiny_onn.modular import TinyOnnForCausalLM
from training.config import DataConfig
from training.data import get_dataloaders


def evaluate_baseline(
    model_name: str,
    data_path: str,
    max_seq_length: int,
    batch_size: int,
    use_tiny_onn: bool,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- BENCHMARKING on {device} ---")
    print(f"Model: {model_name}, TinyOnn: {use_tiny_onn}")

    ModelClass = TinyOnnForCausalLM if use_tiny_onn else AutoModelForCausalLM

    model = ModelClass.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir="weights",
        ignore_mismatched_sizes=use_tiny_onn,
    ).to(device, dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        model_max_length=max_seq_length,
        cache_dir="weights",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_config = DataConfig(
        mode="local_json",
        train_path=data_path,
        eval_path=data_path,
        max_seq_length=max_seq_length,
    )

    _, eval_dataloader = get_dataloaders(
        data_config=data_config,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=0,
    )

    model.train()  # Use train mode to enable gradients

    # Warm-up
    for _ in range(2):
        batch = next(iter(eval_dataloader))
        inputs = {k: v.to(device) for k, v in batch.items()}
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = F.cross_entropy(
            outputs.logits.view(-1, model.config.vocab_size), labels.view(-1)
        )
        loss.backward()

    torch.cuda.synchronize()

    start_time = time.time()
    num_batches = 0
    progress_bar = tqdm(eval_dataloader, desc="Benchmarking")

    for batch in progress_bar:
        inputs = {k: v.to(device) for k, v in batch.items()}
        labels = inputs.pop("labels")

        # Forward pass
        outputs = model(**inputs)
        loss = F.cross_entropy(
            outputs.logits.view(-1, model.config.vocab_size), labels.view(-1)
        )

        # Backward pass
        loss.backward()
        num_batches += 1

    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_batch = total_time / num_batches if num_batches > 0 else 0

    print("\n--- Benchmark Results ---")
    print(f"Total time for {num_batches} batches: {total_time:.4f} seconds")
    print(f"Average time per batch: {avg_time_per_batch:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark a model's forward/backward pass."
    )
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
    args = parser.parse_args()
    evaluate_baseline(
        args.model_name,
        args.data_path,
        args.max_seq_length,
        args.batch_size,
        args.use_tiny_onn,
    )
