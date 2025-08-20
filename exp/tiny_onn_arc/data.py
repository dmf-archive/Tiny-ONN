import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, default_collate

from .config import TinyOnnArcConfig


def pad_grid(grid: list[list[int]], max_h: int, max_w: int) -> torch.Tensor:
    grid_tensor = torch.tensor(grid, dtype=torch.long)
    h, w = grid_tensor.shape

    pad_h = max_h - h
    pad_w = max_w - w

    if pad_h > 0 or pad_w > 0:
        grid_tensor = F.pad(grid_tensor, (0, pad_w, 0, pad_h), "constant", 0)

    return grid_tensor


def augment_grid(grid: torch.Tensor, flip_lr: bool, flip_ud: bool, rot_k: int) -> torch.Tensor:
    if flip_lr:
        grid = torch.fliplr(grid)
    if flip_ud:
        grid = torch.flipud(grid)
    if rot_k > 0:
        grid = torch.rot90(grid, rot_k, [0, 1])
    return grid


class ArcDataset(Dataset):
    def __init__(self, task_files: list[Path], config: TinyOnnArcConfig, use_test_pairs: bool = False):
        self.config = config
        self.max_h = 30
        self.max_w = 30
        self.use_test_pairs = use_test_pairs
        self.samples = []
        for task_file in task_files:
            with open(task_file, "r") as f:
                task = json.load(f)
                pairs = task["test"] if self.use_test_pairs else task["train"]
                for pair in pairs:
                    self.samples.append(pair)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        pair = self.samples[idx]

        input_tensor = pad_grid(pair["input"], self.max_h, self.max_w)
        output_tensor = pad_grid(pair["output"], self.max_h, self.max_w)

        flip_lr = random.random() > 0.5
        flip_ud = random.random() > 0.5
        rot_k = random.randint(0, 3)

        aug_input = augment_grid(input_tensor, flip_lr, flip_ud, rot_k)
        aug_output = augment_grid(output_tensor, flip_lr, flip_ud, rot_k)

        newline_token = 10
        input_rows = [torch.cat((row, torch.tensor([newline_token], dtype=torch.long))) for row in aug_input]
        output_rows = [torch.cat((row, torch.tensor([newline_token], dtype=torch.long))) for row in aug_output]

        input_seq = torch.cat(input_rows)
        output_seq = torch.cat(output_rows)
        
        # For "N+1 visible" training, input is the full sequence, labels are the same
        model_input = torch.cat([input_seq, output_seq], dim=0)
        
        labels = model_input.clone()
        # Mask out the input part from the loss calculation
        labels[:len(input_seq)] = -100

        return model_input, labels


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None
    return default_collate(batch)


def get_arc_dataset(config: TinyOnnArcConfig, data_dir: str = "data/ARC-AGI-2/data") -> tuple[ArcDataset, ArcDataset]:
    data_path = Path(data_dir)
    train_files = list((data_path / "training").glob("*.json"))
    eval_files = list((data_path / "evaluation").glob("*.json"))

    train_dataset = ArcDataset(train_files, config, use_test_pairs=False)
    eval_dataset = ArcDataset(eval_files, config, use_test_pairs=True)

    return train_dataset, eval_dataset
