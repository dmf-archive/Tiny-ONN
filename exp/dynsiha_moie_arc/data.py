import json
from pathlib import Path
from typing import Any
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from .tokenizer import ArcPositionalTokenizer


class GridSerializer:
    def __init__(self, tokenizer: ArcPositionalTokenizer):
        self.tokenizer = tokenizer

    def _serialize_grid(self, grid: list[list[int]]) -> list[int]:
        tokens = []
        for r, row in enumerate(grid):
            for c, color in enumerate(row):
                tokens.append(self.tokenizer.grid_to_token_id(r, c, color))
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

        problem_ids = context_ids + [self.tokenizer.vocab["<|im_start|>"], self.tokenizer.vocab["problem"]] + test_input_grid_ids + [self.tokenizer.vocab["<|im_end|>"], self.tokenizer.vocab["<|im_start|>"], self.tokenizer.vocab["solution"]]
        
        full_ids = problem_ids + test_output_grid_ids + [self.tokenizer.eos_token_id]
        
        labels = [-100] * len(problem_ids) + full_ids[len(problem_ids):]
        
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
        problem_ids = context_ids + [self.tokenizer.vocab["<|im_start|>"], self.tokenizer.vocab["problem"]] + test_input_grid_ids + [self.tokenizer.vocab["<|im_end|>"], self.tokenizer.vocab["<|im_start|>"], self.tokenizer.vocab["solution"]]
        
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
    def __init__(self, tokenizer: ArcPositionalTokenizer, max_len: int):
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
    def __init__(self, tokenizer: ArcPositionalTokenizer):
        self.tokenizer = tokenizer

    def deserialize(self, tokens: list[int]) -> torch.Tensor:
        grid_points = []
        max_h, max_w = 0, 0
        
        # Heuristic to find the last solution block of tokens
        try:
            last_solution_start_idx = -1
            for i in range(len(tokens) - 1, -1, -1):
                if tokens[i] == self.tokenizer.vocab["solution"]:
                    last_solution_start_idx = i
                    break
            
            if last_solution_start_idx == -1: return torch.zeros((1, 1), dtype=torch.long)
            
            solution_tokens = tokens[last_solution_start_idx + 1:]
            
            for token_id in solution_tokens:
                grid_info = self.tokenizer.token_id_to_grid(token_id)
                if grid_info:
                    r, c, color = grid_info
                    grid_points.append((r, c, color))
                    max_h = max(max_h, r)
                    max_w = max(max_w, c)

        except (IndexError, ValueError):
            return torch.zeros((1, 1), dtype=torch.long)

        if not grid_points:
            return torch.zeros((1, 1), dtype=torch.long)

        grid = torch.zeros((max_h + 1, max_w + 1), dtype=torch.long)
        for r, c, color in grid_points:
            grid[r, c] = color

        return grid
