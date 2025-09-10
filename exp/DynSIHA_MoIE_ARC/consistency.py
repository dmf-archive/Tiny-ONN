from collections.abc import Callable

import torch
import torch.nn.functional as F


class ConsistencyTools:

    @staticmethod
    def _identity(x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def _rot90_cw(x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=3, dims=(-2, -1))

    @staticmethod
    def _rot180_cw(x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=2, dims=(-2, -1))

    @staticmethod
    def _rot270_cw(x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=1, dims=(-2, -1))

    @staticmethod
    def _flip(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=(-1,))

    @classmethod
    def get_transforms(cls) -> list[Callable[[torch.Tensor], torch.Tensor]]:
        return [
            cls._identity,
            cls._rot90_cw,
            cls._rot180_cw,
            cls._rot270_cw,
            cls._flip,
            lambda x: cls._rot90_cw(cls._flip(x)),
            lambda x: cls._rot180_cw(cls._flip(x)),
            lambda x: cls._rot270_cw(cls._flip(x)),
        ]

    @classmethod
    def get_inverse_transforms(cls) -> list[Callable[[torch.Tensor], torch.Tensor]]:
        return [
            cls._identity,
            cls._rot270_cw,
            cls._rot180_cw,
            cls._rot90_cw,
            cls._flip,
            lambda x: cls._flip(cls._rot270_cw(x)),
            lambda x: cls._flip(cls._rot180_cw(x)),
            lambda x: cls._flip(cls._rot90_cw(x)),
        ]

    @classmethod
    def apply_transforms(cls, grids: torch.Tensor) -> torch.Tensor:
        transforms = cls.get_transforms()
        transformed_grids = [transform(grids) for transform in transforms]
        
        max_h = max(g.shape[-2] for g in transformed_grids)
        max_w = max(g.shape[-1] for g in transformed_grids)
        
        padded_grids = []
        for grid in transformed_grids:
            pad_h = max_h - grid.shape[-2]
            pad_w = max_w - grid.shape[-1]
            padded_grid = F.pad(grid, (0, pad_w, 0, pad_h), "constant", 0)
            padded_grids.append(padded_grid)
            
        return torch.stack(padded_grids, dim=0)

    @classmethod
    def apply_inverse_transforms(cls, grids_per_view: torch.Tensor) -> torch.Tensor:
        if grids_per_view.dim() != 4 or grids_per_view.shape[1] != 8:
            raise ValueError("Input must be a [B, 8, H, W] tensor")

        inverse_transforms = cls.get_inverse_transforms()
        inversed_grids = []
        for i in range(8):
            inversed_grids.append(inverse_transforms[i](grids_per_view[:, i, :, :]))
        return torch.stack(inversed_grids, dim=1)
