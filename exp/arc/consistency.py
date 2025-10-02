import torch
import torch.nn.functional as F


class ConsistencyTools:
    def __init__(self):
        self.transforms = self._get_base_transforms()

    @staticmethod
    def _get_base_transforms() -> list[torch.autograd.Function]:
        return [
            lambda x: x,
            lambda x: torch.rot90(x, 1, [0, 1]),
            lambda x: torch.rot90(x, 2, [0, 1]),
            lambda x: torch.rot90(x, 3, [0, 1]),
            lambda x: torch.flip(x, [0]),
            lambda x: torch.flip(x, [1]),
            lambda x: torch.transpose(x, 0, 1),
            lambda x: torch.rot90(torch.flip(x, [0]), 1, [0, 1]),
        ]

    def get_transforms(self) -> list[torch.autograd.Function]:
        return self.transforms
