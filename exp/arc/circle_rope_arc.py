
import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def get_arc_rope_3d_index(coords_3d: torch.Tensor, hidden_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    coords_3d.shape[0]
    dim = hidden_size // 3

    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

    x_emb = torch.outer(coords_3d[:, 0].float(), inv_freq)
    y_emb = torch.outer(coords_3d[:, 1].float(), inv_freq)
    z_emb = torch.outer(coords_3d[:, 2].float(), inv_freq)

    x_cos = torch.cos(x_emb)
    x_sin = torch.sin(x_emb)
    y_cos = torch.cos(y_emb)
    y_sin = torch.sin(y_emb)
    z_cos = torch.cos(z_emb)
    z_sin = torch.sin(z_emb)

    cos = torch.cat([x_cos, y_cos, z_cos], dim=-1)
    sin = torch.cat([x_sin, y_sin, z_sin], dim=-1)

    return cos, sin

def apply_arc_rope_3d(q: torch.Tensor, k: torch.Tensor, coords_3d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = get_arc_rope_3d_index(coords_3d, q.shape[-1])

    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)

    return q_rot, k_rot

def build_3d_coords_1d(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(seq_len, device=device).unsqueeze(-1).expand(seq_len, 3).float()

def build_3d_coords_2d(grid_coords: torch.Tensor, seq_positions: torch.Tensor) -> torch.Tensor:
    coords_3d = torch.zeros((seq_positions.shape[0], 3), device=grid_coords.device, dtype=torch.float32)
    valid_mask = (grid_coords[:, 0] != -1) & (grid_coords[:, 1] != -1)
    coords_3d[valid_mask, 0] = grid_coords[valid_mask, 1].float()
    coords_3d[valid_mask, 1] = grid_coords[valid_mask, 0].float()
    coords_3d[:, 2] = seq_positions.float()
    return coords_3d

def normalize_3d_coords(coords_3d: torch.Tensor) -> torch.Tensor:
    norm_coords = coords_3d.clone()
    for i in range(3):
        col = coords_3d[:, i]
        min_val = col.min()
        max_val = col.max()
        if max_val > min_val:
            norm_coords[:, i] = (col - min_val) / (max_val - min_val + 1e-8)
        else:
            norm_coords[:, i] = 0.0
    return norm_coords
