import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import ArcColorTokenizerFast


class GridSerializer:
    def __init__(self, tokenizer: ArcColorTokenizerFast):
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

    def serialize_task(self, task_data: dict[str, Any]) -> tuple[list[int], list[int], list[tuple[int, int]], list[bool]]:
        full_ids: list[int] = [self.tokenizer.bos_token_id]
        full_coords: list[tuple[int, int]] = [(-1, -1)]
        labels: list[int] = [-100]
        diff_mask: list[bool] = [False]

        im_start_id = self.tokenizer.vocab["<im_start>"]
        im_end_id = self.tokenizer.vocab["<im_end>"]

        def extend_and_mask(
            ids: list[int],
            coords: list[tuple[int, int]],
            is_input: bool,
            input_grid_for_diff: list[list[int]] | None = None,
        ):
            full_ids.extend([im_start_id] + ids + [im_end_id])
            full_coords.extend([(-1, -1)] + coords + [(-1, -1)])
            
            diff_mask.extend([False] * (len(ids) + 2))

            if is_input:
                labels.extend([-100] * (len(ids) + 2))
            else:
                labels.extend([-100] + ids + [im_end_id])
                if input_grid_for_diff:
                    output_grid = self.tokenizer.decode_grid(ids)
                    
                    token_idx = 0
                    diff_mask_start_index = len(full_ids) - len(ids) - 1
                    
                    for r, row in enumerate(output_grid):
                        if r > 0:
                            token_idx += 1
                        for c, color in enumerate(row):
                            is_diff = False
                            if r < len(input_grid_for_diff) and c < len(input_grid_for_diff[r]):
                                if color != input_grid_for_diff[r][c]:
                                    is_diff = True
                            else: # Pixel exists in output but not input (size change)
                                is_diff = True
                            
                            diff_mask[diff_mask_start_index + token_idx] = is_diff
                            token_idx += 1


        for pair in task_data["train"]:
            input_ids, input_coords = self._serialize_grid(pair["input"])
            extend_and_mask(input_ids, input_coords, is_input=True)

            output_ids, output_coords = self._serialize_grid(pair["output"])
            extend_and_mask(output_ids, output_coords, is_input=False, input_grid_for_diff=pair["input"])

        test_input_ids, test_input_coords = self._serialize_grid(
            task_data["test"][0]["input"]
        )
        extend_and_mask(test_input_ids, test_input_coords, is_input=True)

        test_output_ids, test_output_coords = self._serialize_grid(
            task_data["test"][0]["output"]
        )
        extend_and_mask(test_output_ids, test_output_coords, is_input=False, input_grid_for_diff=task_data["test"][0]["input"])

        full_ids.append(self.tokenizer.eos_token_id)
        full_coords.append((-1, -1))
        labels.append(self.tokenizer.eos_token_id)
        diff_mask.append(False)

        return full_ids, labels, full_coords, diff_mask

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

        test_data = task_data.get("test", [])
        if test_data and len(test_data) > 0 and "input" in test_data[0]:
            test_input_ids, test_input_coords = self._serialize_grid(test_data[0]["input"])
            prompt_ids.extend([im_start_id] + test_input_ids + [im_end_id, im_start_id])
            prompt_coords.extend([(-1, -1)] + test_input_coords + [(-1, -1), (-1, -1)])
        else:
            prompt_ids.extend([im_start_id, im_end_id, im_start_id])
            prompt_coords.extend([(-1, -1), (-1, -1), (-1, -1)])

        return prompt_ids, prompt_coords


class InMemoryArcDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: ArcColorTokenizerFast, split: str = "training", max_len: int = 4096):
        self.data_path = Path(data_path) / split
        self.split = split
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.serializer = GridSerializer(self.tokenizer)

        all_tasks = []
        file_paths = sorted(list(self.data_path.glob("*.json")))
        for path in file_paths:
            with open(path) as f:
                task_data = json.load(f)
            
            # 基础有效性检查
            if not (split == "training" and "test" in task_data and task_data["test"] and len(task_data["test"]) > 0 and "output" in task_data["test"][0] or split == "evaluation" and "test" in task_data and task_data["test"] and len(task_data["test"]) > 0):
                continue
            
            # 长度过滤检查 (Pre-filtering)
            # 使用无增强的原始数据进行序列化长度预估
            ids, _, _, _ = self.serializer.serialize_task(task_data)
            if len(ids) > self.max_len:
                continue

            all_tasks.append(task_data)

        tasks_with_metrics = [
            (task, self._calculate_task_difficulty(task)) for task in all_tasks
        ]

        sorted_tasks = sorted(
            tasks_with_metrics, key=lambda x: (x[1]["max_pixels"], x[1]["entropy"])
        )

        self.tasks = [task for task, metrics in sorted_tasks]

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


    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.tasks[idx]


# ArcCollator is no longer needed as we are handling batching manually in the training loop
# and using the tokenizer for padding.


class GridDeserializer:
    def __init__(self, tokenizer: ArcColorTokenizerFast):
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
