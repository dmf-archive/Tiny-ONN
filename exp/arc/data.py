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

    def serialize_mini_task(self, mini_task: dict[str, Any]) -> tuple[list[int], list[int]]:
        input_grid_ids = self._serialize_grid(mini_task['input'])
        output_grid_ids = self._serialize_grid(mini_task['output'])

        prompt_part1 = [self.tokenizer.bos_token_id, self.tokenizer.vocab["problem"]] + input_grid_ids
        prompt_part2 = [self.tokenizer.vocab["solution"]]

        full_ids = prompt_part1 + prompt_part2 + output_grid_ids + [self.tokenizer.eos_token_id]

        prompt_len = len(prompt_part1) + len(prompt_part2)
        labels = [-100] * prompt_len + output_grid_ids + [self.tokenizer.eos_token_id]

        return full_ids, labels

    def serialize_for_inference(self, task_data: dict[str, Any]) -> list[int]:
        test_input_grid_ids = self._serialize_grid(task_data['test'][0]['input'])
        prompt_ids = [self.tokenizer.bos_token_id, self.tokenizer.vocab["problem"]] + test_input_grid_ids + [self.tokenizer.vocab["solution"]]
        return prompt_ids

class InMemoryArcDataset(Dataset):
    def __init__(self, data_path: str, split: str = "training"):
        self.data_path = Path(data_path) / split
        self.mini_tasks = []

        file_paths = sorted(list(self.data_path.glob("*.json")))
        for path in file_paths:
            with open(path) as f:
                task_data = json.load(f)

            if split == "training":
                for pair in task_data['train']:
                    self.mini_tasks.append(pair)
                for pair in task_data['test']:
                    self.mini_tasks.append(pair)
            elif split == "evaluation":
                for pair in task_data['train']:
                    self.mini_tasks.append(pair)

                if 'test' in task_data and task_data['test'] and 'output' in task_data['test']:
                    for pair in task_data['test']:
                        self.mini_tasks.append(pair)


        tokenizer_for_sorting = ArcColorTokenizer()
        serializer_for_sorting = GridSerializer(tokenizer_for_sorting)

        tasks_with_lengths = []
        for mini_task in self.mini_tasks:
            input_ids, _ = serializer_for_sorting.serialize_mini_task(mini_task)
            tasks_with_lengths.append((mini_task, len(input_ids)))

        sorted_tasks = sorted(tasks_with_lengths, key=lambda x: x[1])
        self.mini_tasks = [task for task, length in sorted_tasks]

    def __len__(self) -> int:
        return len(self.mini_tasks)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.mini_tasks[idx]

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
        all_input_ids, all_labels, all_entropies = [], [], []

        for mini_task in batch:
            input_ids, labels = self.serializer.serialize_mini_task(mini_task)

            if len(input_ids) > self.max_len:
                continue
            
            entropy = self._calculate_sample_entropy(labels)
            all_entropies.append(entropy)
            all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            all_labels.append(torch.tensor(labels, dtype=torch.long))

        if not all_input_ids:
            return {}

        padded_input_ids = pad_sequence(all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_labels = pad_sequence(all_labels, batch_first=True, padding_value=-100)
        
        return {
            "input_ids": padded_input_ids,
            "labels": padded_labels,
            "task_data": batch,
            "sample_entropy": torch.tensor(all_entropies, dtype=torch.float32)
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
        except ValueError:
            max_len = max(len(r) for r in grid_rows)
            padded_rows = [r + [0] * (max_len - len(r)) for r in grid_rows]
            return torch.tensor(padded_rows, dtype=torch.long)
