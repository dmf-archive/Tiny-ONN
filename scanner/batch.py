import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from scanner.core.scheduler import Scheduler


class JsonlDataset(Dataset):
    def __init__(self, file_path: Path, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]["text"]
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs


def run_batch_scan(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scheduler = Scheduler(model_name=args.model, device=device)
    scheduler._load_model_and_tokenizer()

    dataset = JsonlDataset(Path(args.corpus_path), scheduler._tokenizer)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{args.model.replace('/', '--')}_{args.corpus_name}_{timestamp}"
    output_path = Path("data/scans") / output_filename

    print("Starting batch scan...")
    print(f"  Model: {args.model}")
    print(f"  Corpus: {args.corpus_path}")
    print(f"  Output: {output_path.with_suffix('.mscan')}")

    progress_bar = tqdm(total=len(dataset), desc="Scanning", unit="sample")

    def progress_callback():
        progress_bar.update(1)

    scheduler.scan_dataset(
        dataset,
        output_path,
        batch_size=args.batch_size,
        progress_callback=progress_callback
    )

    progress_bar.close()
    print("Batch scan complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch scanner.")
    parser.add_argument("--model", type=str, required=True, help="Model name or path.")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to .jsonl corpus file.")
    parser.add_argument("--corpus_name", type=str, default="corpus", help="Name for the output file.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing.")
    args = parser.parse_args()
    run_batch_scan(args)
