import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .tokenizer import ArcColorTokenizer


class GridSerializer:
    def __init__(self, tokenizer: ArcColorTokenizer):
        self.tokenizer = tokenizer

    def _serialize_grid(self, grid: list[list[int]]) -> tuple[list[int], list[tuple[int, int]]]:
        tokens = []
        coords = []
        for r, row in enumerate(grid):
            if r > 0:
                tokens.append(self.tokenizer.row_sep_token_id)
                coords.append((-1, -1))
            for c, color in enumerate(row):
                tokens.append(self.tokenizer.color_to_token_id(color))
                coords.append((r, c))
        return tokens, coords

    def serialize_task(self, task_data: dict[str, Any]) -> tuple[list[int], list[int], list[tuple[int, int]]]:
        full_ids: list[int] = [self.tokenizer.bos_token_id]
        full_coords: list[tuple[int, int]] = [(-1, -1)]
        labels: list[int] = [-100]

        im_start_id = self.tokenizer.vocab["<im_start>"]
        im_end_id = self.tokenizer.vocab["<im_end>"]

        def extend_and_mask(
            ids: list[int],
            coords: list[tuple[int, int]],
            is_input: bool,
        ):
            full_ids.extend([im_start_id] + ids + [im_end_id])
            full_coords.extend([(-1, -1)] + coords + [(-1, -1)])
            if is_input:
                labels.extend([-100] * (len(ids) + 2))
            else:
                labels.extend([-100] + ids + [im_end_id])

        for pair in task_data["train"]:
            input_ids, input_coords = self._serialize_grid(pair["input"])
            extend_and_mask(input_ids, input_coords, is_input=True)

            output_ids, output_coords = self._serialize_grid(pair["output"])
            extend_and_mask(output_ids, output_coords, is_input=False)

        test_input_ids, test_input_coords = self._serialize_grid(
            task_data["test"][0]["input"]
        )
        extend_and_mask(test_input_ids, test_input_coords, is_input=True)

        test_output_ids, test_output_coords = self._serialize_grid(
            task_data["test"][0]["output"]
        )
        extend_and_mask(test_output_ids, test_output_coords, is_input=False)

        full_ids.append(self.tokenizer.eos_token_id)
        full_coords.append((-1, -1))
        labels.append(self.tokenizer.eos_token_id)

        return full_ids, labels, full_coords

    def serialize_for_inference(self, task_data: dict[str, Any]) -> tuple[list[int], list[tuple[int, int]]]:
        prompt_ids: list[int] = [self.tokenizer.bos_token_id]
        prompt_coords: list[tuple[int, int]] = [(-1, -1)]

        im_start_id = self.tokenizer.vocab["<im_start>"]
        im_end_id = self.tokenizer.vocab["<im_end>"]

        for pair in task_data["train"]:
            input_ids, input_coords = self._serialize_grid(pair["input"])
            output_ids, output_coords = self._serialize_grid(pair["output"])

            prompt_ids.extend([im_start_id] + input_ids + [im_end_id])
            prompt_coords.extend([(-1, -1)] + input_coords + [(-1, -1)])

            prompt_ids.extend([im_start_id] + output_ids + [im_end_id])
            prompt_coords.extend([(-1, -1)] + output_coords + [(-1, -1)])

        test_input_ids, test_input_coords = self._serialize_grid(task_data["test"][0]["input"])
        prompt_ids.extend([im_start_id] + test_input_ids + [im_end_id, im_start_id])
        prompt_coords.extend([(-1, -1)] + test_input_coords + [(-1, -1), (-1, -1)])

        return prompt_ids, prompt_coords


class InMemoryArcDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: ArcColorTokenizer, split: str = "training", warmup_ratio: float = 1.0):
        self.data_path = Path(data_path) / split
        self.split = split
        self.warmup_ratio = warmup_ratio
        
        all_tasks = []
        file_paths = sorted(list(self.data_path.glob("*.json")))
        for path in file_paths:
            with open(path) as f:
                task_data = json.load(f)
            if split == "training" and "test" in task_data and task_data["test"] and "output" in task_data["test"][0]:
                all_tasks.append(task_data)
            elif split == "evaluation" and "test" in task_data and task_data["test"]:
                all_tasks.append(task_data)

        tasks_with_metrics = [
            (task, self._calculate_task_difficulty(task)) for task in all_tasks
        ]
        
        sorted_tasks = sorted(
            tasks_with_metrics, key=lambda x: (x[1]["max_pixels"], x[1]["entropy"])
        )
        
        self.full_tasks = [task for task, metrics in sorted_tasks]
        self.warmup_size = int(len(self.full_tasks) * self.warmup_ratio)
        
        self.tasks = self.full_tasks
        if self.split == "training":
            self.set_stage(1)

    @staticmethod
    def _normalize(values: list[float]) -> list[float]:
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return [0.0] * len(values)
        return [(v - min_val) / (max_val - min_val) for v in values]

    @staticmethod
    def _calculate_grid_entropy(grid: list[list[int]]) -> float:
        flat_grid = [item for sublist in grid for item in sublist]
        if not flat_grid:
            return 0.0
        _, counts = np.unique(flat_grid, return_counts=True)
        probs = counts / len(flat_grid)
        return -np.sum(probs * np.log2(probs))

    def _calculate_task_difficulty(self, task_data: dict) -> dict[str, float]:
        max_pixels = 0
        entropies = []

        for pair in task_data["train"] + task_data["test"]:
            input_grid, output_grid = np.array(pair["input"]), np.array(pair["output"])
            max_pixels = max(max_pixels, input_grid.size, output_grid.size)
            entropies.append(self._calculate_grid_entropy(pair["output"]))

        return {
            "max_pixels": float(max_pixels),
            "entropy": np.mean(entropies) if entropies else 0.0,
        }

    def set_stage(self, stage: int):
        if self.split != "training":
            return
        if stage == 1:
            self.tasks = self.full_tasks[:self.warmup_size]
        else:
            self.tasks = self.full_tasks

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.tasks[idx]


class ArcCollator:
    def __init__(self, tokenizer: ArcColorTokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.serializer = GridSerializer(tokenizer)
        self.max_len = max_len

    @staticmethod
    def _calculate_sample_entropy(labels: list[int]) -> float:
        valid_labels = [l for l in labels if l != -100]
        if not valid_labels:
            return 0.0

        counts = torch.bincount(torch.tensor(valid_labels))
        probs = counts.float() / len(valid_labels)
        probs = probs[probs > 0]
        return -torch.sum(probs * torch.log2(probs)).item()

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        all_input_ids, all_labels, all_coords, all_entropies = [], [], [], []

        for task_data in batch:
            input_ids, labels, coords = self.serializer.serialize_task(task_data)

            if len(input_ids) > self.max_len:
                continue

            entropy = self._calculate_sample_entropy(labels)
            all_entropies.append(entropy)
            all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            all_labels.append(torch.tensor(labels, dtype=torch.long))
            all_coords.append(torch.tensor(coords, dtype=torch.long))

        if not all_input_ids:
            return {}

        padded_input_ids = pad_sequence(all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_labels = pad_sequence(all_labels, batch_first=True, padding_value=-100)
        padded_coords = pad_sequence(all_coords, batch_first=True, padding_value=-1)

        return {
            "input_ids": padded_input_ids,
            "labels": padded_labels,
            "coords": padded_coords,
            "task_data": batch,
            "sample_entropy": torch.tensor(all_entropies, dtype=torch.float32),
        }


class GridDeserializer:
    def __init__(self, tokenizer: ArcColorTokenizer):
        self.tokenizer = tokenizer

    def deserialize(self, tokens: list[int]) -> torch.Tensor:
        grid_rows = []
        current_row: list[int] = []

        clean_tokens = []
        for token_id in tokens:
            if token_id in [
                self.tokenizer.vocab["<im_start>"],
                self.tokenizer.vocab["<im_end>"],
                self.tokenizer.eos_token_id,
            ]:
                continue
            clean_tokens.append(token_id)

        for token_id in clean_tokens:
            if token_id == self.tokenizer.row_sep_token_id:
                if current_row:
                    grid_rows.append(current_row)
                current_row = []
            else:
                color = self.tokenizer.token_id_to_color(token_id)
                if color is not None:
                    current_row.append(color)

        if current_row:
            grid_rows.append(current_row)

        if not grid_rows:
            return torch.zeros((1, 1), dtype=torch.long)

        try:
            return torch.tensor(grid_rows, dtype=torch.long)
        except ValueError:
            max_len = max(len(r) for r in grid_rows)
            padded_rows = [r + [0] * (max_len - len(r)) for r in grid_rows]
            return torch.tensor(padded_rows, dtype=torch.long)
