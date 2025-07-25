from pathlib import Path
from typing import Mapping

from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir: Path):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_metrics(self, metrics: Mapping[str, float | int | str], step: int):
        for key, value in metrics.items():
            if isinstance(value, int | float):
                self.writer.add_scalar(f"metrics/{key}", value, step)

    def close(self):
        self.writer.close()
