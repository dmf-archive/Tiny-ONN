from collections.abc import Callable
from typing import List

import torch

class ConsistencyTools:
    @staticmethod
    @torch.jit.script
    def _jit_apply_color_transform(grids: torch.Tensor, color_map: torch.Tensor) -> torch.Tensor:
        mapping = torch.arange(10, device=grids.device) 
        mapping[:10] = color_map
        return mapping[grids]

    @staticmethod
    @torch.jit.script
    def _jit_apply_transforms(grids: torch.Tensor) -> torch.Tensor:
        B, H, W = grids.shape
        transformed = torch.empty(8, B, H, W, device=grids.device, dtype=grids.dtype)

        transformed[0] = grids
        transformed[1] = torch.rot90(grids, k=3, dims=(-2, -1))
        transformed[2] = torch.rot90(grids, k=2, dims=(-2, -1))
        transformed[3] = torch.rot90(grids, k=1, dims=(-2, -1))

        flipped = torch.flip(grids, dims=(-1,))
        transformed[4] = flipped
        transformed[5] = torch.rot90(flipped, k=3, dims=(-2, -1))
        transformed[6] = torch.rot90(flipped, k=2, dims=(-2, -1))
        transformed[7] = torch.rot90(flipped, k=1, dims=(-2, -1))
        return transformed

    @staticmethod
    @torch.jit.script
    def _jit_apply_inverse_transforms(grids_per_view: torch.Tensor) -> torch.Tensor:
        B, N, H, W = grids_per_view.shape
        inversed = torch.empty(B, 8, H, W, device=grids_per_view.device, dtype=grids_per_view.dtype)

        inversed[:, 0] = grids_per_view[:, 0]
        inversed[:, 1] = torch.rot90(grids_per_view[:, 1], k=1, dims=(-2, -1))
        inversed[:, 2] = torch.rot90(grids_per_view[:, 2], k=2, dims=(-2, -1))
        inversed[:, 3] = torch.rot90(grids_per_view[:, 3], k=3, dims=(-2, -1))

        inversed[:, 4] = torch.flip(grids_per_view[:, 4], dims=(-1,))
        inversed[:, 5] = torch.flip(torch.rot90(grids_per_view[:, 5], k=1, dims=(-2, -1)), dims=(-1,))
        inversed[:, 6] = torch.flip(torch.rot90(grids_per_view[:, 6], k=2, dims=(-2, -1)), dims=(-1,))
        inversed[:, 7] = torch.flip(torch.rot90(grids_per_view[:, 7], k=3, dims=(-2, -1)), dims=(-1,))
        return inversed

    @classmethod
    def get_transforms(cls) -> list[Callable[[torch.Tensor], torch.Tensor]]:
        return [
            lambda x: x,
            lambda x: torch.rot90(x, k=3, dims=(-2, -1)),
            lambda x: torch.rot90(x, k=2, dims=(-2, -1)),
            lambda x: torch.rot90(x, k=1, dims=(-2, -1)),
            lambda x: torch.flip(x, dims=(-1,)),
            lambda x: torch.rot90(torch.flip(x, dims=(-1,)), k=3, dims=(-2, -1)),
            lambda x: torch.rot90(torch.flip(x, dims=(-1,)), k=2, dims=(-2, -1)),
            lambda x: torch.rot90(torch.flip(x, dims=(-1,)), k=1, dims=(-2, -1)),
        ]

    @classmethod
    def get_inverse_transforms(cls) -> list[Callable[[torch.Tensor], torch.Tensor]]:
        return [
            lambda x: x,
            lambda x: torch.rot90(x, k=1, dims=(-2, -1)),
            lambda x: torch.rot90(x, k=2, dims=(-2, -1)),
            lambda x: torch.rot90(x, k=3, dims=(-2, -1)),
            lambda x: torch.flip(x, dims=(-1,)),
            lambda x: torch.flip(torch.rot90(x, k=1, dims=(-2, -1)), dims=(-1,)),
            lambda x: torch.flip(torch.rot90(x, k=2, dims=(-2, -1)), dims=(-1,)),
            lambda x: torch.flip(torch.rot90(x, k=3, dims=(-2, -1)), dims=(-1,)),
        ]

    @classmethod
    def apply_transforms(cls, grids: torch.Tensor) -> torch.Tensor:
        return cls._jit_apply_transforms(grids)

    @classmethod
    def apply_inverse_transforms(cls, grids_per_view: torch.Tensor) -> torch.Tensor:
        if grids_per_view.dim() != 4 or grids_per_view.shape[1] != 8:
            raise ValueError("Input must be a [B, 8, H, W] tensor")
        return cls._jit_apply_inverse_transforms(grids_per_view)
