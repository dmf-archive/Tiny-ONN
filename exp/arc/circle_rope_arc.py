
import math
import torch

@torch.jit.script
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

@torch.jit.script
def get_circle_rope_embeddings(
    coords_3d: torch.Tensor, hidden_size: int, radius: float = 10.0, alpha: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    
    coords_2d = coords_3d[:, :2]
    x, y = coords_2d[:, 0], coords_2d[:, 1]
    
    valid_mask = (x != -1) & (y != -1)
    projected_coords_2d = coords_2d.clone()

    if valid_mask.any():
        x_valid, y_valid = x[valid_mask], y[valid_mask]
        
        theta_orig = torch.atan2(y_valid, x_valid)
        theta_min, theta_max = theta_orig.min(), theta_orig.max()
        theta_range = theta_max - theta_min
        theta_norm = (
            (theta_orig - theta_min) / theta_range * (2 * math.pi)
            if theta_range > 0
            else theta_orig
        )
        
        indices = torch.arange(x_valid.shape[0], dtype=torch.float32, device=x.device)
        theta_index = indices / x_valid.shape[0] * (2 * math.pi)
        
        theta_uniform = alpha * theta_norm + (1 - alpha) * theta_index

        new_x_valid = radius * torch.cos(theta_uniform)
        new_y_valid = radius * torch.sin(theta_uniform)

        projected_coords_2d[valid_mask, 0] = new_x_valid
        projected_coords_2d[valid_mask, 1] = new_y_valid

    projected_coords_3d = torch.cat([projected_coords_2d, coords_3d[:, 2].unsqueeze(-1)], dim=-1)

    dim_per_component = hidden_size // 3
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 2 * dim_per_component, 2, device=coords_3d.device).float() / dim_per_component))

    pos_emb = torch.einsum("i,j->ij", projected_coords_3d[:, 0], inv_freq)
    grid_y_emb = torch.einsum("i,j->ij", projected_coords_3d[:, 1], inv_freq)
    grid_x_emb = torch.einsum("i,j->ij", projected_coords_3d[:, 2], inv_freq)

    cos = torch.cat([torch.cos(pos_emb), torch.cos(grid_y_emb), torch.cos(grid_x_emb)], dim=-1)
    sin = torch.cat([torch.sin(pos_emb), torch.sin(grid_y_emb), torch.sin(grid_x_emb)], dim=-1)
    
    return cos, sin

@torch.jit.script
def build_3d_coords(grid_coords: torch.Tensor, seq_positions: torch.Tensor) -> torch.Tensor:
    coords_3d = torch.zeros((seq_positions.shape[0], 3), device=grid_coords.device, dtype=torch.float32)
    valid_mask = (grid_coords[:, 0] != -1) & (grid_coords[:, 1] != -1)
    coords_3d[valid_mask, 0] = grid_coords[valid_mask, 1].float()
    coords_3d[valid_mask, 1] = grid_coords[valid_mask, 0].float()
    coords_3d[:, 2] = seq_positions.float()
    return coords_3d

@torch.jit.script
def apply_arc_rope(q: torch.Tensor, k: torch.Tensor, coords_2d: torch.Tensor, position_ids: torch.Tensor, hidden_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    gathered_coords = coords_2d.squeeze(0)[position_ids.squeeze(0)].unsqueeze(0)
    
    coords_3d = build_3d_coords(gathered_coords.squeeze(0), position_ids.squeeze(0).float())
    cos, sin = get_circle_rope_embeddings(coords_3d, hidden_size)

    cos_emb = cos.unsqueeze(0).unsqueeze(0)
    sin_emb = sin.unsqueeze(0).unsqueeze(0)
    
    q_rot = (q * cos_emb) + (_rotate_half(q) * sin_emb)
    k_rot = (k * cos_emb) + (_rotate_half(k) * sin_emb)
    return q_rot, k_rot
