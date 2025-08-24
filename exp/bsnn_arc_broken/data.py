import json
import random
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import DataLoader, Dataset

from .bayesian_config import BayesianConfig


def apply_augmentations(grid: torch.Tensor) -> torch.Tensor:
    if random.random() > 0.5:
        grid = torch.fliplr(grid)
    if random.random() > 0.5:
        grid = torch.flipud(grid)
    k = random.randint(0, 3)
    if k > 0:
        grid = torch.rot90(grid, k, [0, 1])
    return grid


class GpuArcDataset(Dataset):
    def __init__(self, task_files: list[Path], config: BayesianConfig, use_test_pairs: bool = False):
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
        return {"input": input_grid, "output": output_grid}


class GridCollator:
    def __init__(self, device: torch.device, max_grid_size: int):
        self.device = device
        self.max_grid_size = max_grid_size

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        input_grids = [item["input"] for item in batch]
        output_grids = [item["output"] for item in batch]

        def pad_grid(grid: torch.Tensor) -> torch.Tensor:
            h, w = grid.shape
            padding = (0, self.max_grid_size - w, 0, self.max_grid_size - h)
            return torch.nn.functional.pad(grid, padding, "constant", 0)

        padded_inputs = torch.stack([pad_grid(grid) for grid in input_grids]).to(self.device)
        padded_outputs = torch.stack([pad_grid(grid) for grid in output_grids]).to(self.device)
        return padded_inputs, padded_outputs


def get_arc_dataloaders(
    config: BayesianConfig,
) -> tuple[DataLoader, DataLoader, int, int]:
    data_path = Path("data/ARC-AGI-2/data")
    train_files = list(data_path.glob("training/*.json"))
    eval_files = list(data_path.glob("evaluation/*.json"))

    train_dataset = GpuArcDataset(train_files, config, use_test_pairs=False)
    eval_dataset = GpuArcDataset(eval_files, config, use_test_pairs=True)

    collator = GridCollator(torch.device(config.DEVICE), config.MAX_GRID_SIZE)

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collator, drop_last=True
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collator, drop_last=True
    )

    return train_loader, eval_loader, len(train_dataset), len(eval_dataset)
