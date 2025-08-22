import json
import random
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .config import Config


def apply_augmentations(grid: torch.Tensor) -> torch.Tensor:
    if random.random() > 0.5:
        grid = torch.fliplr(grid)
    if random.random() > 0.5:
        grid = torch.flipud(grid)
    k = random.randint(0, 3)
    if k > 0:
        grid = torch.rot90(grid, k, [0, 1])
    return grid

def serialize_grid(grid: torch.Tensor) -> torch.Tensor:
    h, w = grid.shape
    newline_token = 10
    rows = [
        torch.cat([grid[i], torch.tensor([newline_token], dtype=torch.long, device=grid.device)])
        for i in range(h)
    ]
    return torch.cat(rows)

class GpuArcDataset(Dataset):
    def __init__(self, task_files: list[Path], config: Config, use_test_pairs: bool = False):
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.pairs = []

        for task_file in task_files:
            try:
                with open(task_file) as f:
                    task_data = json.load(f)

                pair_set = task_data["test" if use_test_pairs else "train"]
                for pair in pair_set:
                    h_in, w_in = len(pair["input"]), len(pair["input"][0])
                    h_out, w_out = len(pair["output"]), len(pair["output"][0])

                    if h_in <= config.MAX_GRID_SIZE and w_in <= config.MAX_GRID_SIZE and \
                       h_out <= config.MAX_GRID_SIZE and w_out <= config.MAX_GRID_SIZE:
                        self.pairs.append({
                            "input": torch.tensor(pair["input"], dtype=torch.long),
                            "output": torch.tensor(pair["output"], dtype=torch.long)
                        })
            except (OSError, json.JSONDecodeError):
                continue

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pair = self.pairs[idx]

        input_grid = pair["input"]
        output_grid = pair["output"]

        if random.random() > 0.5:
            input_grid = apply_augmentations(input_grid)
            output_grid = apply_augmentations(output_grid)

        color_map = torch.randperm(10)
        input_grid = color_map[input_grid]
        output_grid = color_map[output_grid]

        input_seq = serialize_grid(input_grid)
        output_seq = serialize_grid(output_grid)

        full_seq = torch.cat([input_seq, output_seq])
        labels = full_seq.clone()
        labels[:len(input_seq)] = -100

        return {"input_ids": full_seq, "labels": labels}

class JitCollator:
    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids_list = [item["input_ids"] for item in batch]
        labels_list = [item["labels"] for item in batch]

        input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=0).to(self.device)
        labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100).to(self.device)

        attention_mask = (input_ids_padded != 0).long()

        return input_ids_padded, labels_padded, attention_mask

def get_arc_dataloaders(config: Config) -> tuple[DataLoader, DataLoader]:
    data_path = Path("data/ARC-AGI-2/data")
    train_files = list(data_path.glob("training/*.json"))
    eval_files = list(data_path.glob("evaluation/*.json"))

    train_dataset = GpuArcDataset(train_files, config, use_test_pairs=False)
    eval_dataset = GpuArcDataset(eval_files, config, use_test_pairs=True)

    collator = JitCollator(torch.device(config.DEVICE))

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=collator)

    return train_loader, eval_loader
