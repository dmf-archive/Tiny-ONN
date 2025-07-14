import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

from scanner.mscan import METRICS_DTYPE

def _quantize_to_uint16(data: np.ndarray) -> np.ndarray:
    min_val, max_val = data.min(), data.max()
    if min_val == max_val:
        return np.zeros_like(data, dtype=np.uint16)
    return (65535 * (data - min_val) / (max_val - min_val)).astype(np.uint16)

def _vectorized_block_norm(tensor: torch.Tensor, block_size: int) -> torch.Tensor:
    padding_needed = (block_size - (tensor.numel() % block_size)) % block_size
    if padding_needed > 0:
        tensor = F.pad(tensor, (0, padding_needed))
    
    blocks = tensor.reshape(-1, block_size)
    return torch.linalg.vector_norm(blocks.float(), ord=2, dim=1)

def process_and_quantize_data(
    activations: Dict[str, torch.Tensor],
    gradients: Dict[str, torch.Tensor],
    param_name_to_id_map: Dict[str, int],
    block_size: int = 64,
    token_idx: int = 0,
) -> np.ndarray:
    
    records = []
    
    for name, act_tensor in activations.items():
        grad_tensor = gradients.get(name)
        if grad_tensor is None or name not in param_name_to_id_map:
            continue
            
        param_id = param_name_to_id_map[name]
        
        act_norms = _vectorized_block_norm(act_tensor.view(-1), block_size)
        grad_norms = _vectorized_block_norm(grad_tensor.view(-1), block_size)
        
        num_blocks = min(len(act_norms), len(grad_norms))
        block_indices = np.arange(num_blocks)
        
        param_records = np.empty(num_blocks, dtype=METRICS_DTYPE)
        param_records['token_idx'] = token_idx
        param_records['param_id'] = param_id
        param_records['block_idx'] = block_indices
        param_records['activation'] = _quantize_to_uint16(act_norms.cpu().numpy())
        param_records['grad_norm'] = _quantize_to_uint16(grad_norms.cpu().numpy())
        records.append(param_records)

    if not records:
        return np.empty(0, dtype=METRICS_DTYPE)

    return np.concatenate(records)
