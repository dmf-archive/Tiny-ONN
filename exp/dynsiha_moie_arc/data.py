import json
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .tokenizer import ArcColorTokenizer


class GridSerializer:
    def __init__(self, tokenizer: ArcColorTokenizer):
        self.tokenizer = tokenizer

    def _serialize_grid(self, grid: list[list[int]]) -> list[int]:
        tokens = []
        for r, row in enumerate(grid):
            if r > 0:
                tokens.append(self.tokenizer.row_sep_token_id)
            for color in row:
                tokens.append(self.tokenizer.color_to_token_id(color))
        return tokens

    def serialize_task_with_context(self, task_data: dict[str, Any]) -> tuple[list[int], list[int]]:

        # This is a list of token IDs
        context_ids = [self.tokenizer.bos_token_id]

        for train_pair in task_data['train']:
            context_ids.append(self.tokenizer.vocab["<|im_start|>"])
            context_ids.append(self.tokenizer.vocab["problem"])
            context_ids.extend(self._serialize_grid(train_pair['input']))
            context_ids.append(self.tokenizer.vocab["<|im_end|>"])

            context_ids.append(self.tokenizer.vocab["<|im_start|>"])
            context_ids.append(self.tokenizer.vocab["solution"])
            context_ids.extend(self._serialize_grid(train_pair['output']))
            context_ids.append(self.tokenizer.vocab["<|im_end|>"])

        test_input_grid_ids = self._serialize_grid(task_data['test'][0]['input'])
        test_output_grid_ids = self._serialize_grid(task_data['test'][0]['output'])

        prompt_part1 = context_ids + [self.tokenizer.vocab["<|im_start|>"], self.tokenizer.vocab["problem"]] + test_input_grid_ids + [self.tokenizer.vocab["<|im_end|>"]]
        prompt_part2 = [self.tokenizer.vocab["<|im_start|>"], self.tokenizer.vocab["solution"]]

        full_ids = prompt_part1 + prompt_part2 + test_output_grid_ids + [self.tokenizer.eos_token_id]

        mask_len = len(prompt_part1)
        labels = [-100] * mask_len + full_ids[mask_len:]

        return full_ids, labels

    def serialize_for_inference(self, task_data: dict[str, Any]) -> list[int]:
        context_ids = [self.tokenizer.bos_token_id]
        for train_pair in task_data['train']:
            context_ids.append(self.tokenizer.vocab["<|im_start|>"])
            context_ids.append(self.tokenizer.vocab["problem"])
            context_ids.extend(self._serialize_grid(train_pair['input']))
            context_ids.append(self.tokenizer.vocab["<|im_end|>"])

            context_ids.append(self.tokenizer.vocab["<|im_start|>"])
            context_ids.append(self.tokenizer.vocab["solution"])
            context_ids.extend(self._serialize_grid(train_pair['output']))
            context_ids.append(self.tokenizer.vocab["<|im_end|>"])

        test_input_grid_ids = self._serialize_grid(task_data['test'][0]['input'])
        problem_ids = context_ids + [self.tokenizer.vocab["<|im_start|>"], self.tokenizer.vocab["problem"]] + test_input_grid_ids + [self.tokenizer.vocab["<|im_end|>"]]

        return problem_ids


class ArcDataset(Dataset):
    def __init__(self, data_path: str, split: str = "training"):
        self.data_path = Path(data_path) / split
        self.file_paths = sorted(list(self.data_path.glob("*.json")))

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        with open(self.file_paths[idx]) as f:
            return json.load(f)


class ArcCollator:
    def __init__(self, tokenizer: ArcColorTokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.serializer = GridSerializer(tokenizer)
        self.max_len = max_len

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        all_input_ids = []
        all_labels = []

        for task_data in batch:
            input_ids, labels = self.serializer.serialize_task_with_context(task_data)

            if len(input_ids) > self.max_len:
                continue

            all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            all_labels.append(torch.tensor(labels, dtype=torch.long))

        if not all_input_ids:
            return {}

        padded_input_ids = pad_sequence(all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_labels = pad_sequence(all_labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": padded_input_ids,
            "labels": padded_labels,
            "task_data": batch
        }

class GridDeserializer:
    def __init__(self, tokenizer: ArcColorTokenizer):
        self.tokenizer = tokenizer

    def deserialize(self, tokens: list[int]) -> torch.Tensor:
        grid_rows = []
        current_row = []

        for token_id in tokens:
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
        except ValueError: # Ragged tensor
            max_len = max(len(r) for r in grid_rows)
            padded_rows = [r + [0] * (max_len - len(r)) for r in grid_rows]
            return torch.tensor(padded_rows, dtype=torch.long)
