import os
from typing import Any

from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict[str, Any], step: int):
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_scalar(self, tag: str, scalar_value: Any, step: int):
        self.writer.add_scalar(tag, scalar_value, step)

    def log_hyperparams(self, hparam_dict: dict[str, Any], metric_dict: dict[str, Any]):
        self.writer.add_hparams(hparam_dict, metric_dict)

    def close(self):
        self.writer.close()
