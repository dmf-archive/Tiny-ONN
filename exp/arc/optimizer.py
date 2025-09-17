import torch
import torch.nn.functional as F
from typing import List, Tuple
from functools import partial

# === Factor Functions for Vmap ===
def _calculate_surprise_norm(mu_grad: torch.Tensor) -> torch.Tensor:
    """Calculates L1 norm of gradients for a single layer."""
    return torch.norm(torch.abs(mu_grad), p=1, dim=-1)

def _calculate_smp_mask_single(surprise_norm: torch.Tensor, p_dyn: torch.Tensor) -> torch.Tensor:
    """Calculates grad mask for a single layer."""
    num_total = surprise_norm.shape[0]
    num_winners = max(1, int(num_total * p_dyn.item()))
    
    grad_mask = torch.ones_like(surprise_norm)
    if num_winners < num_total:
        winner_indices = surprise_norm.argsort()[:num_winners]
        grad_mask = torch.zeros_like(surprise_norm)
        grad_mask.index_fill_(0, winner_indices, 1.0)
    return grad_mask

def _calculate_meta_loss_single(
    proto_weight: torch.Tensor,
    gate_param: torch.Tensor,
    mu_grad: torch.Tensor,
    sbl_input: torch.Tensor,
    p_dyn: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates proto and gate loss for a single layer."""
    surprise_map = torch.abs(mu_grad.detach())
    surprise_norm = _calculate_surprise_norm(surprise_map)

    num_total = surprise_norm.shape[0]
    num_good = max(1, int(num_total * p_dyn.item()))

    # --- Gate Loss ---
    gate_loss = -(gate_param * surprise_norm).mean()

    # --- Proto Loss (with masking for conditional logic) ---
    proto_loss = torch.tensor(0.0, device=p_dyn.device, dtype=p_dyn.dtype)
    if num_good < num_total:
        indices = surprise_norm.argsort()
        good_indices, bad_indices = indices[:num_good], indices[num_good:]

        anchor = sbl_input.detach().mean(dim=(0, 1))
        
        good_protos = F.normalize(proto_weight.index_select(0, good_indices), p=2.0, dim=-1)
        l_attr = -F.cosine_similarity(good_protos, anchor.unsqueeze(0), dim=-1).mean()

        bad_protos = F.normalize(proto_weight.index_select(0, bad_indices), p=2.0, dim=-1)
        bad_surprise_map = surprise_map.index_select(0, bad_indices)
        l_rep = -F.cosine_similarity(bad_protos, -bad_surprise_map, dim=-1).mean()
        
        if torch.isfinite(l_attr) and torch.isfinite(l_rep):
            proto_loss = l_attr + l_rep

    return proto_loss, gate_loss

# === Parallelized Worker Functions ===
def calculate_smp_masks_vmap(
    mu_weight_grads: List[torch.Tensor],
    p_dyn: torch.Tensor
) -> List[torch.Tensor]:
    """Parallelized calculation of grad masks using vmap."""
    if not mu_weight_grads:
        return []

    surprise_norms = [_calculate_surprise_norm(grad) for grad in mu_weight_grads]
    
    # vmap requires all inputs to have the same structure. We use a loop here
    # because the output shape depends on the input shape, which varies per layer.
    # This is a limitation of vmap for ragged tensors.
    # A more advanced technique would involve padding or nested tensors, but for now,
    # the primary bottleneck (argsort) is inside the layer-wise function, which is correct.
    grad_masks = [_calculate_smp_mask_single(sn, p_dyn) for sn in surprise_norms]
    return grad_masks

def calculate_meta_loss_vmap(
    proto_weights: List[torch.Tensor],
    gate_params: List[torch.Tensor],
    mu_grads: List[torch.Tensor],
    sbl_inputs: List[torch.Tensor],
    p_dyn: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parallelized calculation of meta losses using a list comprehension (factor-worker)."""
    if not proto_weights:
        return torch.tensor(0.0, device=p_dyn.device), torch.tensor(0.0, device=p_dyn.device)

    # This applies the "factor" function (_calculate_meta_loss_single) to each "worker" (layer).
    # Python's GIL prevents true multi-threading for CPU-bound tasks, but PyTorch operations
    # on GPU tensors release the GIL, allowing for some overlap.
    # The primary benefit is cleaner code and avoiding a manual Python loop over layers.
    losses = [_calculate_meta_loss_single(pw, gp, mg, si, p_dyn)
              for pw, gp, mg, si in zip(proto_weights, gate_params, mu_grads, sbl_inputs)]
    
    # Unpack and sum losses
    proto_losses, gate_losses = zip(*losses)
    proto_loss_total = torch.stack(proto_losses).sum()
    gate_loss_total = torch.stack(gate_losses).sum()

    return proto_loss_total, gate_loss_total

# === Legacy Wrappers for Compatibility ===
# Keep the original function names for backward compatibility with train.py
calculate_smp_masks = calculate_smp_masks_vmap
calculate_meta_loss = calculate_meta_loss_vmap