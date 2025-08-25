import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .tokenizer import ArcTokenizer


class ArcViTDataset(Dataset):
    def __init__(self, task_files: list[Path], config: Config, use_test_pairs: bool = False):
        self.config = config
        self.pairs = []

        for task_file in task_files:
            try:
                with open(task_file, "r", encoding="utf-8") as f:
                    task_data = json.load(f)

                pair_set = task_data["test" if use_test_pairs else "train"]
                for pair in pair_set:
                    h_in, w_in = len(pair["input"]), len(pair["input"][0])
                    h_out, w_out = len(pair["output"]), len(pair["output"][0])

                    if (
                        h_in <= config.MAX_GRID_SIZE
                        and w_in <= config.MAX_GRID_SIZE
                        and h_out <= config.MAX_GRID_SIZE
                        and w_out <= config.MAX_GRID_SIZE
                    ):
                        self.pairs.append(
                            {
                                "input": torch.tensor(pair["input"], dtype=torch.long),
                                "output": torch.tensor(pair["output"], dtype=torch.long),
                            }
                        )
            except (OSError, json.JSONDecodeError):
                continue

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.pairs[idx]


def custom_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_h_in = max(item["input"].shape[0] for item in batch)
    max_w_in = max(item["input"].shape[1] for item in batch)
    max_h_out = max(item["output"].shape[0] for item in batch)
    max_w_out = max(item["output"].shape[1] for item in batch)

    padded_inputs = torch.full((len(batch), max_h_in, max_w_in), ArcTokenizer.PAD_TOKEN_ID, dtype=torch.long)
    padded_outputs = torch.full((len(batch), max_h_out, max_w_out), ArcTokenizer.PAD_TOKEN_ID, dtype=torch.long)

    for i, item in enumerate(batch):
        h_in, w_in = item["input"].shape
        padded_inputs[i, :h_in, :w_in] = item["input"]
        h_out, w_out = item["output"].shape
        padded_outputs[i, :h_out, :w_out] = item["output"]

    return {"input": padded_inputs, "output": padded_outputs}


def get_arc_dataloaders(config: Config) -> tuple[DataLoader, DataLoader, int, int]:
    data_path = Path("data/ARC-AGI-2/data")
    train_files = sorted(list(data_path.glob("training/*.json")))
    eval_files = sorted(list(data_path.glob("evaluation/*.json")))

    train_dataset = ArcViTDataset(train_files, config, use_test_pairs=False)
    eval_dataset = ArcViTDataset(eval_files, config, use_test_pairs=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        collate_fn=custom_collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        collate_fn=custom_collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, eval_loader, len(train_dataset), len(eval_dataset)
