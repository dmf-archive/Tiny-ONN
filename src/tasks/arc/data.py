import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class ARCTokenizer:
    def __init__(self):
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.im_start_id = 3
        self.im_end_id = 4
        self.row_sep_id = 5
        self.pair_sep_id = 6
        self.color_offset = 7  # 7-16 are colors 0-9
        self.vocab_size = 17

    def color_to_id(self, color: torch.Tensor) -> torch.Tensor:
        return color + self.color_offset

    def id_to_color(self, token_id: torch.Tensor) -> torch.Tensor:
        return token_id - self.color_offset

class ARCProcessor:
    def __init__(self, tokenizer: ARCTokenizer):
        self.tokenizer = tokenizer

    def encode_grid(self, grid: torch.Tensor) -> torch.Tensor:
        h, w = grid.shape
        tokens = []
        for r in range(h):
            tokens.append(self.tokenizer.color_to_id(grid[r]))
            if r < h - 1:
                tokens.append(torch.tensor([self.tokenizer.row_sep_id], device=grid.device))
        return torch.cat(tokens)

    def decode_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        rows = []
        current_row = []
        for t in tokens:
            t_id = t.item()
            if t_id == self.tokenizer.row_sep_id:
                if current_row:
                    rows.append(current_row)
                current_row = []
            elif self.tokenizer.color_offset <= t_id < self.tokenizer.color_offset + 10:
                current_row.append(t_id - self.tokenizer.color_offset)
            elif t_id in [self.tokenizer.im_end_id, self.tokenizer.eos_id]:
                break
        if current_row:
            rows.append(current_row)

        if not rows:
            return torch.zeros((0, 0), dtype=torch.long)

        # Handle potential ragged rows by padding or truncating to first row width
        width = len(rows[0])
        valid_rows = [r[:width] + [0]*(width-len(r)) for r in rows]
        return torch.tensor(valid_rows, dtype=torch.long)

    def serialize_for_inference(self, task: dict[str, Any]) -> torch.Tensor:
        all_ids = [torch.tensor([self.tokenizer.bos_id])]

        for pair in task["train"]:
            in_grid = torch.tensor(pair["input"], dtype=torch.long)
            out_grid = torch.tensor(pair["output"], dtype=torch.long)

            all_ids.append(torch.tensor([self.tokenizer.im_start_id]))
            all_ids.append(self.encode_grid(in_grid))
            all_ids.append(torch.tensor([self.tokenizer.im_end_id]))

            all_ids.append(torch.tensor([self.tokenizer.im_start_id]))
            all_ids.append(self.encode_grid(out_grid))
            all_ids.append(torch.tensor([self.tokenizer.im_end_id]))
            all_ids.append(torch.tensor([self.tokenizer.pair_sep_id]))

        test_in = torch.tensor(task["test"][0]["input"], dtype=torch.long)
        all_ids.append(torch.tensor([self.tokenizer.im_start_id]))
        all_ids.append(self.encode_grid(test_in))
        all_ids.append(torch.tensor([self.tokenizer.im_end_id]))
        all_ids.append(torch.tensor([self.tokenizer.im_start_id])) # Prompt for output

        return torch.cat(all_ids)

class ARCDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "training",
        max_seq_len: int = 2048,
        augment: bool = True,
        h1_all_predict: bool = True,
    ):
        self.tokenizer = ARCTokenizer()
        self.processor = ARCProcessor(self.tokenizer)
        self.data_path = Path(data_dir) / split
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.h1_all_predict = h1_all_predict

        self.tasks = []
        for p in sorted(self.data_path.glob("*.json")):
            with open(p) as f:
                self.tasks.append(json.load(f))

    def __len__(self) -> int:
        return len(self.tasks)

    def _apply_augment(self, grid: torch.Tensor, flip_dim: int | None, rot_k: int, color_perm: torch.Tensor) -> torch.Tensor:
        if flip_dim is not None:
            grid = torch.flip(grid, dims=[flip_dim])
        if rot_k > 0:
            grid = torch.rot90(grid, k=rot_k, dims=[0, 1])

        # Vectorized color mapping
        mask = (grid >= 0) & (grid <= 9)
        grid[mask] = color_perm[grid[mask]]
        return grid

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        task = self.tasks[idx]

        # Generate augmentation params
        color_perm = torch.arange(10)
        flip_dim = None
        rot_k = 0

        if self.augment:
            # 1. Color Permutation (excluding background 0 often helps, but ARC allows all)
            if random.random() > 0.5:
                color_perm = torch.randperm(10)

            # 2. Geometric Symmetry (D4 Group: 8 symmetries)
            flip_dim = random.choice([None, 0, 1])
            rot_k = random.randint(0, 3)

        all_ids = [torch.tensor([self.tokenizer.bos_id])]
        all_labels = [torch.tensor([-100])]
        all_diff_masks = [torch.tensor([0.0])]

        def process_pair(pair: dict[str, Any], is_test: bool):
            in_grid = torch.tensor(pair["input"], dtype=torch.long)
            out_grid = torch.tensor(pair["output"], dtype=torch.long) if "output" in pair else None

            in_grid = self._apply_augment(in_grid, flip_dim, rot_k, color_perm)
            in_tokens = self.processor.encode_grid(in_grid)

            # Input block
            all_ids.append(torch.tensor([self.tokenizer.im_start_id]))
            all_ids.append(in_tokens)
            all_ids.append(torch.tensor([self.tokenizer.im_end_id]))

            all_labels.append(torch.tensor([-100] * (len(in_tokens) + 2)))
            all_diff_masks.append(torch.tensor([0.0] * (len(in_tokens) + 2)))

            if out_grid is not None:
                out_grid = self._apply_augment(out_grid, flip_dim, rot_k, color_perm)
                out_tokens = self.processor.encode_grid(out_grid)

                all_ids.append(torch.tensor([self.tokenizer.im_start_id]))
                all_ids.append(out_tokens)
                all_ids.append(torch.tensor([self.tokenizer.im_end_id]))

                # H1: All Predict logic
                if is_test or self.h1_all_predict:
                    all_labels.append(torch.tensor([-100]))
                    all_labels.append(out_tokens)
                    all_labels.append(torch.tensor([self.tokenizer.im_end_id]))
                else:
                    all_labels.append(torch.tensor([-100] * (len(out_tokens) + 2)))

                # H2: Diff Mask logic (Adaptive Differential Loss)
                # Binary mask: 1.0 for innovation (diff), 0.0 for noise (identity/copy)
                d_mask = torch.zeros(len(out_tokens) + 2)
                if in_grid.shape == out_grid.shape:
                    in_tokens_for_diff = self.processor.encode_grid(in_grid)
                    if len(in_tokens_for_diff) == len(out_tokens):
                        # Only mark tokens that actually CHANGED
                        is_diff = (in_tokens_for_diff != out_tokens).float()
                        d_mask[1:-1] = is_diff
                    else:
                        d_mask[1:-1] = 1.0 # Length change implies total innovation
                else:
                    d_mask[1:-1] = 1.0 # Shape change implies total innovation

                all_diff_masks.append(d_mask)

        for pair in task["train"]:
            process_pair(pair, is_test=False)
            all_ids.append(torch.tensor([self.tokenizer.pair_sep_id]))
            all_labels.append(torch.tensor([-100]))
            all_diff_masks.append(torch.tensor([0.0]))

        for pair in task["test"]:
            process_pair(pair, is_test=True)

        all_ids.append(torch.tensor([self.tokenizer.eos_id]))
        all_labels.append(torch.tensor([self.tokenizer.eos_id]))
        all_diff_masks.append(torch.tensor([0.0]))

        input_ids = torch.cat(all_ids)[:self.max_seq_len]
        labels = torch.cat(all_labels)[:self.max_seq_len]
        diff_mask = torch.cat(all_diff_masks)[:self.max_seq_len]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "diff_mask": diff_mask,
        }

class PackedARCDataLoader:
    def __init__(
        self,
        dataset: ARCDataset,
        max_tokens: int = 8192,
        shuffle: bool = True
    ):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.tokenizer = dataset.tokenizer

    def __iter__(self):
        samples = []
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            samples.append({
                "data": sample,
                "len": len(sample["input_ids"])
            })

        if self.shuffle:
            random.shuffle(samples)

        # FFD: First Fit Decreasing
        samples.sort(key=lambda x: x["len"], reverse=True)

        batches = []
        current_batch = []
        current_tokens = 0

        for s in samples:
            if current_tokens + s["len"] > self.max_tokens and current_batch:
                batches.append(self._collate(current_batch))
                current_batch = []
                current_tokens = 0

            current_batch.append(s["data"])
            current_tokens += s["len"]

        if current_batch:
            batches.append(self._collate(current_batch))

        if self.shuffle:
            random.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self):
        return len(self.dataset) // 4 # Rough estimate for progress bars

    def _collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = [s["input_ids"] for s in batch]
        labels = [s["labels"] for s in batch]
        diff_masks = [s["diff_mask"] for s in batch]

        max_len = max(len(x) for x in input_ids)

        padded_input_ids = []
        padded_labels = []
        padded_diff_masks = []
        attention_masks = []

        for ids, lbls, dms in zip(input_ids, labels, diff_masks):
            pad_len = max_len - len(ids)
            padded_input_ids.append(torch.cat([ids, torch.full((pad_len,), self.tokenizer.pad_id)]))
            padded_labels.append(torch.cat([lbls, torch.full((pad_len,), -100)]))
            padded_diff_masks.append(torch.cat([dms, torch.zeros(pad_len)]))
            attention_masks.append(torch.cat([torch.ones(len(ids)), torch.zeros(pad_len)]))

        return {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "diff_mask": torch.stack(padded_diff_masks),
            "attention_mask": torch.stack(attention_masks)
        }

def get_arc_dataloader(data_dir: str, batch_size: int, split: str = "training", num_workers: int = 4):
    dataset = ARCDataset(data_dir, split=split)
    return PackedARCDataLoader(
        dataset,
        max_tokens=8192 if split == "training" else 2048,
        shuffle=(split == "training")
    )
