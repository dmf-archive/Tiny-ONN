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
        return " \n ".join([" ".join(map(str, row)) for row in grid])

    def serialize_task_with_context(self, task_data: dict[str, Any]) -> tuple[list[int], list[int]]:
        context_parts = ["<|bos|>"]
        
        for train_pair in task_data['train']:
            input_grid = self._serialize_grid(train_pair['input'])
            output_grid = self._serialize_grid(train_pair['output'])
            context_parts.append(f"<|im_start|> problem {input_grid} <|im_end|> <|im_start|> solution {output_grid} <|im_end|>")

        test_input_grid = self._serialize_grid(task_data['test'][0]['input'])
        test_output_grid = self._serialize_grid(task_data['test'][0]['output'])
        
        problem_str = f" <|im_start|> problem {test_input_grid} <|im_end|> <|im_start|> solution"
        solution_str = f" {test_output_grid} <|im_end|> <|eos|>"
        
        context_str = " ".join(context_parts)
        full_sequence_str = f"{context_str}{problem_str}{solution_str}"

        full_ids = self.tokenizer.encode(full_sequence_str)
        
        # Re-encode to find the exact length of the problem part
        problem_only_str = f"{context_str}{problem_str}"
        problem_ids_len = len(self.tokenizer.encode(problem_only_str))

        labels = [-100] * problem_ids_len + full_ids[problem_ids_len:]
        
        return full_ids, labels

    def serialize_for_inference(self, task_data: dict[str, Any]) -> list[int]:
        context_parts = ["<|bos|>"]
        for train_pair in task_data['train']:
            input_grid = self._serialize_grid(train_pair['input'])
            output_grid = self._serialize_grid(train_pair['output'])
            context_parts.append(f"<|im_start|> problem {input_grid} <|im_end|> <|im_start|> solution {output_grid} <|im_end|>")
            
        test_input_grid = self._serialize_grid(task_data['test'][0]['input'])
        problem_str = f" <|im_start|> problem {test_input_grid} <|im_end|> <|im_start|> solution"
        
        full_prompt_str = " ".join(context_parts) + problem_str
        return self.tokenizer.encode(full_prompt_str)


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
    def __init__(self, tokenizer: ArcChatMLTokenizer):
        self.tokenizer = tokenizer
        self.serializer = GridSerializer(tokenizer)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        task_data = batch[0]
        
        input_ids, labels = self.serializer.serialize_task_with_context(task_data)
        
        # Note: We are not using pad_sequence here because batch size is 1 and each task is a single sequence.
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
            "labels": torch.tensor(labels, dtype=torch.long).unsqueeze(0),
            "task_data": task_data,
        }


class GridDeserializer:
    def __init__(self, tokenizer: ArcChatMLTokenizer):
        self.tokenizer = tokenizer

    def deserialize(self, tokens: list[int]) -> torch.Tensor:
        decoded_str = self.tokenizer.decode(tokens)
        
        try:
            # Find the last occurrence of solution pattern
            solution_str = decoded_str.rsplit("solution", 1)[1].split("<|im_start|>")[1].split("<|im_end|>")[0].strip()
        except IndexError:
            return torch.zeros((1, 1), dtype=torch.long)

        if not solution_str:
            return torch.zeros((1, 1), dtype=torch.long)
        
        grid = []
        rows_str = solution_str.split("\n")
        for row_str in rows_str:
            clean_row_str = row_str.strip()
            if clean_row_str:
                try:
                    grid.append(list(map(int, clean_row_str.split())))
                except ValueError:
                    continue
        
        if not grid or not any(grid):
            return torch.zeros((1,1), dtype=torch.long)

        # Pad rows to be of the same length to form a valid tensor
        max_len = max(len(row) for row in grid) if grid else 0
        padded_grid = [row + [self.tokenizer.pad_token_id] * (max_len - len(row)) for row in grid]

        return torch.tensor(padded_grid, dtype=torch.long)
