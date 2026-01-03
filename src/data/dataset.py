import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


class ArcTaskDataset(Dataset):
    def __init__(self, data_path: str, split: str = "training", warmup_ratio: float = 1.0) -> None:
        self.data_path: Path = Path(data_path) / split
        self.split: str = split

        self.tasks: list[dict[str, list[torch.Tensor]]] = []
        file_paths: list[Path] = sorted(list(self.data_path.glob("*.json")))
        for path in file_paths:
            with open(path) as f:
                task_data: dict = json.load(f)
            if "train" in task_data and "test" in task_data:
                self.tasks.append(self._preprocess_task(task_data))

        self.tasks.sort(key=lambda x: sum(t.numel() for t in x["train_output"]))

        self.warmup_size: int = int(len(self.tasks) * warmup_ratio)
        self.active_tasks: list[dict[str, list[torch.Tensor]]] = self.tasks[:self.warmup_size] if split == "training" else self.tasks

    def _preprocess_task(self, task_data: dict) -> dict[str, list[torch.Tensor]]:
        processed: dict[str, list[torch.Tensor]] = {
            "train_input": [torch.tensor(p["input"], dtype=torch.long) for p in task_data["train"]],
            "train_output": [torch.tensor(p["output"], dtype=torch.long) for p in task_data["train"]],
            "test_input": [torch.tensor(p["input"], dtype=torch.long) for p in task_data["test"]],
            "test_output": [torch.tensor(p["output"], dtype=torch.long) for p in task_data["test"] if "output" in p]
        }
        return processed

    def set_stage(self, stage: int) -> None:
        if self.split == "training":
            self.active_tasks = self.tasks if stage > 1 else self.tasks[:self.warmup_size]

    def __len__(self) -> int:
        return len(self.active_tasks)

    def __getitem__(self, idx: int) -> dict[str, list[torch.Tensor]]:
        task: dict[str, list[torch.Tensor]] = self.active_tasks[idx]
        return self._augment_task(task)

    def _augment_task(self, task: dict[str, list[torch.Tensor]]) -> dict[str, list[torch.Tensor]]:
        transform_idx: int = random.randint(0, 7)

        perm: torch.Tensor = torch.arange(10)
        active_colors: torch.Tensor = torch.unique(torch.cat([t.flatten() for t in task["train_input"] + task["train_output"]]))
        active_colors = active_colors[active_colors != 0]
        if len(active_colors) >= 2:
            c1, c2 = active_colors[torch.randperm(len(active_colors))[:2]]
            perm[c1], perm[c2] = perm[c2], perm[c1]

        def transform(g: torch.Tensor) -> torch.Tensor:
            if transform_idx == 1: g = torch.rot90(g, 1, [0, 1])
            elif transform_idx == 2: g = torch.rot90(g, 2, [0, 1])
            elif transform_idx == 3: g = torch.rot90(g, 3, [0, 1])
            elif transform_idx == 4: g = torch.flip(g, [0])
            elif transform_idx == 5: g = torch.flip(g, [1])
            elif transform_idx == 6: g = g.T
            elif transform_idx == 7: g = torch.rot90(torch.flip(g, [0]), 1, [0, 1])
            return perm[g]

        return {k: [transform(g) for g in v] for k, v in task.items()}
