import json
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# --- JIT-Compilable Augmentation Function ---
@torch.jit.script
def process_batch(
    inputs: torch.Tensor, 
    outputs: torch.Tensor, 
    indices: List[int], 
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    batch_inputs = inputs[indices]
    batch_outputs = outputs[indices]
    
    # Vectorized random augmentation parameters
    batch_size = batch_inputs.shape[0]
    flips_lr = torch.rand(batch_size, device=device) > 0.5
    flips_ud = torch.rand(batch_size, device=device) > 0.5
    rots_k = torch.randint(0, 4, (batch_size,), device=device)

    # Vectorized augmentations
    for i in range(batch_size):
        if flips_lr[i]: batch_inputs[i] = torch.fliplr(batch_inputs[i])
        if flips_ud[i]: batch_inputs[i] = torch.flipud(batch_inputs[i])
        if rots_k[i] > 0: batch_inputs[i] = torch.rot90(batch_inputs[i], int(rots_k[i]), [0, 1])
        
        if flips_lr[i]: batch_outputs[i] = torch.fliplr(batch_outputs[i])
        if flips_ud[i]: batch_outputs[i] = torch.flipud(batch_outputs[i])
        if rots_k[i] > 0: batch_outputs[i] = torch.rot90(batch_outputs[i], int(rots_k[i]), [0, 1])

    # Vectorized color permutation
    color_map = torch.randperm(10, device=device)
    batch_inputs = color_map[batch_inputs]
    batch_outputs = color_map[batch_outputs]
    
    # Vectorized serialization
    newline_token = torch.tensor([10], dtype=torch.long, device=device)
    h, w = batch_inputs.shape[1], batch_inputs.shape[2]
    
    input_seqs = torch.cat((batch_inputs.flatten(1), newline_token.expand(batch_size, h).flatten(1)), dim=1).view(batch_size, h, w + 1)
    output_seqs = torch.cat((batch_outputs.flatten(1), newline_token.expand(batch_size, h).flatten(1)), dim=1).view(batch_size, h, w + 1)

    input_seq = input_seqs.flatten(1)
    output_seq = output_seqs.flatten(1)

    model_input = torch.cat([input_seq, output_seq], dim=1)
    labels = model_input.clone()
    labels[:, :input_seq.shape[1]] = -100
    
    return model_input, labels

# --- GPU-Cached Dataset ---
class GpuArcDataset(Dataset):
    def __init__(self, task_files: List[Path], use_test_pairs: bool = False, device: torch.device = torch.device("cpu"), max_grid_size: int = 10):
        self.device = device
        self.max_h, self.max_w = max_grid_size, max_grid_size
        
        all_inputs, all_outputs = [], []
        for task_file in task_files:
            try:
                task = json.load(open(task_file))
                pairs = task["test" if use_test_pairs else "train"]
                for pair in pairs:
                    h_in, w_in = len(pair["input"]), len(pair["input"][0])
                    h_out, w_out = len(pair["output"]), len(pair["output"][0])

                    if h_in <= self.max_h and w_in <= self.max_w and h_out <= self.max_h and w_out <= self.max_w:
                        all_inputs.append(self.pad_grid(pair["input"]))
                        all_outputs.append(self.pad_grid(pair["output"]))
            except (IOError, json.JSONDecodeError):
                continue

        self.inputs = torch.stack(all_inputs).to(device)
        self.outputs = torch.stack(all_outputs).to(device)

    def pad_grid(self, grid: List[List[int]]) -> torch.Tensor:
        grid_tensor = torch.tensor(grid, dtype=torch.long)
        h, w = grid_tensor.shape
        return F.pad(grid_tensor, (0, self.max_w - w, 0, self.max_h - h), "constant", 0)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> int:
        return idx

class JitCollator:
    def __init__(self, dataset: GpuArcDataset):
        self.inputs = dataset.inputs
        self.outputs = dataset.outputs
        self.device = dataset.device
    
    def __call__(self, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        return process_batch(self.inputs, self.outputs, indices, self.device)

def get_arc_dataset(data_dir: str = "data/ARC-AGI-2/data", device: torch.device = torch.device("cpu")) -> Tuple[GpuArcDataset, GpuArcDataset, JitCollator, JitCollator]:
    device = torch.device("cpu")
    data_path = Path(data_dir)
    train_files = list(data_path.glob("training/*.json"))
    eval_files = list(data_path.glob("evaluation/*.json"))
    
    train_dataset = GpuArcDataset(train_files, use_test_pairs=False, device=device, max_grid_size=10)
    eval_dataset = GpuArcDataset(eval_files, use_test_pairs=True, device=device, max_grid_size=10)
    
    train_collator = JitCollator(train_dataset)
    eval_collator = JitCollator(eval_dataset)
    
    return train_dataset, eval_dataset, train_collator, eval_collator
