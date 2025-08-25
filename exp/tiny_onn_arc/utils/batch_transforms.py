import random
from typing import Callable

import torch


def _apply_single_augmentation(grid: torch.Tensor) -> torch.Tensor:
    if random.random() > 0.5:
        grid = torch.fliplr(grid)
    if random.random() > 0.5:
        grid = torch.flipud(grid)
    k = random.randint(0, 3)
    if k > 0:
        grid = torch.rot90(grid, k, dims=[0, 1])
    return grid


def apply_batch_augmentations(
    inputs: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    
    aug_inputs_list = [_apply_single_augmentation(grid) for grid in torch.unbind(inputs)]
    aug_targets_list = [_apply_single_augmentation(grid) for grid in torch.unbind(targets)]

    max_h_in = max(t.shape[0] for t in aug_inputs_list)
    max_w_in = max(t.shape[1] for t in aug_inputs_list)
    max_h_out = max(t.shape[0] for t in aug_targets_list)
    max_w_out = max(t.shape[1] for t in aug_targets_list)

    b = inputs.shape[0]
    padded_aug_inputs = torch.full(
        (b, max_h_in, max_w_in), 0, dtype=inputs.dtype, device=inputs.device
    )
    padded_aug_targets = torch.full(
        (b, max_h_out, max_w_out), 0, dtype=targets.dtype, device=targets.device
    )

    for i in range(b):
        h_in, w_in = aug_inputs_list[i].shape
        padded_aug_inputs[i, :h_in, :w_in] = aug_inputs_list[i]
        h_out, w_out = aug_targets_list[i].shape
        padded_aug_targets[i, :h_out, :w_out] = aug_targets_list[i]

    return padded_aug_inputs, padded_aug_targets


def apply_batch_color_remap(
    *batch_tensors: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    from exp.tiny_onn_arc.tokenizer import ArcTokenizer
    
    remapped_outputs = []
    for tensor in batch_tensors:
        b = tensor.shape[0]
        
        color_shuffles = torch.stack([torch.randperm(10, device=tensor.device) for _ in range(b)])
        
        pad_token = ArcTokenizer.PAD_TOKEN_ID
        full_map = torch.arange(ArcTokenizer.VOCAB_SIZE, device=tensor.device).unsqueeze(0).expand(b, -1)
        full_map[:, :10] = color_shuffles

        
        remapped_tensor = torch.gather(full_map, 1, tensor.view(b, -1)).view(tensor.shape)
        
        mask = tensor == pad_token
        remapped_tensor[mask] = pad_token
        
        remapped_outputs.append(remapped_tensor)

    return tuple(remapped_outputs)