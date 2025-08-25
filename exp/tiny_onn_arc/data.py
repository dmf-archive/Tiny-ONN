import json
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .tokenizer import ArcChatMLTokenizer


class GridSerializer:
    def __init__(self, tokenizer: ArcChatMLTokenizer):
        self.tokenizer = tokenizer

    def _serialize_grid(self, grid: list[list[int]]) -> str:
        return " ".join([f"<row_start> {' '.join(map(str, row))} <row_end>" for row in grid])

    def serialize_task(self, input_grid: list[list[int]], output_grid: list[list[int]]) -> tuple[list[int], list[int]]:
        problem_str = f"problem <|im_start|> {self._serialize_grid(input_grid)} <|im_end|>"
        solution_str = f"solution <|im_start|> {self._serialize_grid(output_grid)} <|im_end|>"
        
        full_sequence = f"{problem_str} {solution_str}"
        
        problem_ids = self.tokenizer.encode(problem_str)
        full_ids = self.tokenizer.encode(full_sequence)
        
        labels = [-100] * len(problem_ids) + full_ids[len(problem_ids):]
        
        return full_ids, labels


class ArcDataset(Dataset):
    def __init__(self, data_path: str, split: str = "training"):
        self.data_path = Path(data_path) / split
        self.file_paths = sorted(list(self.data_path.glob("*.json")))
        self.samples = []
        for file_path in self.file_paths:
            with open(file_path) as f:
                task_data = json.load(f)
                for sample in task_data['train']:
                    self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


class ArcCollator:
    def __init__(self, tokenizer: ArcChatMLTokenizer):
        self.tokenizer = tokenizer
        self.serializer = GridSerializer(tokenizer)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids_list, labels_list = [], []

        for item in batch:
            input_ids, labels = self.serializer.serialize_task(item['input'], item['output'])
            input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        return {
            "input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id),
            "labels": pad_sequence(labels_list, batch_first=True, padding_value=-100),
            "input_grids": [item['input'] for item in batch],
            "output_grids": [item['output'] for item in batch],
        }


class GridDeserializer:
    def __init__(self, tokenizer: ArcChatMLTokenizer):
        self.tokenizer = tokenizer

    def deserialize(self, tokens: list[int]) -> torch.Tensor:
        decoded_str = self.tokenizer.decode(tokens)
        
        try:
            solution_str = decoded_str.split("solution <|im_start|>")[1].split("<|im_end|>")[0].strip()
        except IndexError:
            return torch.zeros((1, 1), dtype=torch.long)

        if not solution_str:
            return torch.zeros((1, 1), dtype=torch.long)
        
        rows_str = solution_str.split("<row_end>")
        grid = []
        for row_str in rows_str:
            if "<row_start>" in row_str:
                clean_row_str = row_str.split("<row_start>")[1].strip()
                if clean_row_str:
                    grid.append(list(map(int, clean_row_str.split())))

        if not grid or not all(isinstance(cell, int) for row in grid for cell in row):
            return torch.zeros((1, 1), dtype=torch.long)

        return torch.tensor(grid, dtype=torch.long)
