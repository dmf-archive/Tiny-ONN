import torch
import torch.nn.functional as F
import time
from rich.console import Console
from rich.table import Table

console = Console()

CONFIG = {
    "D_MODEL": 64,
    "D_FFN": 64,
    "BATCH_SIZE": 16,
    "SEQ_LEN": 256,
    "NUM_RUNS": 100,
    "DEVICE": "cpu",
    "DTYPE": torch.float32
}

def forward_dense(x, w_ffn1, w_ffn2, active_mask):
    """
    方案 A: 逻辑稀疏，计算稠密
    Gating is dense, main computation is dense, output is logically sparse.
    """
    ffn1_out = F.silu(x @ w_ffn1.T)
    masked_ffn1_out = ffn1_out * active_mask
    final_out = masked_ffn1_out @ w_ffn2.T
    return final_out

def forward_sparse(x, w_ffn1, w_ffn2, active_mask):
    """
    方案 B: 真正的前向稀疏 (门控稠密，计算稀疏)
    Uses gather-compute-scatter pattern for the first FFN layer.
    """
    batch_size, seq_len, _ = x.shape
    x_reshaped = x.view(-1, CONFIG["D_MODEL"])
    num_tokens = x_reshaped.shape[0]

    num_active_per_token = active_mask.sum(dim=1).long()
    
    if num_active_per_token.sum() == 0:
        return torch.zeros(batch_size, seq_len, CONFIG["D_MODEL"], device=CONFIG["DEVICE"], dtype=CONFIG["DTYPE"])

    k_max = num_active_per_token.max().item()
    
    active_indices = torch.where(active_mask)[1]
    
    padded_indices = torch.full((num_tokens, k_max), -1, device=CONFIG["DEVICE"])
    
    token_indices_for_scatter = torch.arange(num_tokens, device=CONFIG["DEVICE"]).unsqueeze(1).expand(-1, k_max)
    
    # Create a mask to select only the valid indices for each token
    k_indices_mask = torch.arange(k_max, device=CONFIG["DEVICE"]) < num_active_per_token.unsqueeze(1)
    padded_indices[k_indices_mask] = active_indices

    # Set padding indices to a valid index (e.g., 0) to avoid gather errors
    valid_indices = padded_indices.clone()
    valid_indices[padded_indices == -1] = 0
    
    gathered_weights = w_ffn1[valid_indices]

    x_expanded = x_reshaped.unsqueeze(1).expand(-1, k_max, -1)
    
    # Perform batched dot product
    sparse_ffn1_out_padded = torch.sum(x_expanded * gathered_weights, dim=-1)
    
    # Mask out the results from padded indices
    sparse_ffn1_out_padded[padded_indices == -1] = 0
    
    # Scatter results back to a dense tensor
    ffn1_out_sparse = torch.zeros(num_tokens, CONFIG["D_FFN"], device=CONFIG["DEVICE"], dtype=CONFIG["DTYPE"])

    # To prevent index errors with scatter_add_, replace -1 with a valid index (e.g., 0).
    # The values at these positions are already zeroed out, so this is a safe no-op.
    scatter_indices = padded_indices.clone()
    scatter_indices[padded_indices == -1] = 0
    
    ffn1_out_sparse.scatter_add_(1, scatter_indices, sparse_ffn1_out_padded)
    
    final_out_reshaped = F.silu(ffn1_out_sparse) @ w_ffn2.T
    
    return final_out_reshaped.view(batch_size, seq_len, CONFIG["D_MODEL"])


def run_benchmark():
    console.print(f"Benchmark Config: {CONFIG}", style="bold yellow")
    
    x = torch.randn(
        CONFIG["BATCH_SIZE"], CONFIG["SEQ_LEN"], CONFIG["D_MODEL"],
        device=CONFIG["DEVICE"], dtype=CONFIG["DTYPE"]
    )
    w_ffn1 = torch.randn(
        CONFIG["D_FFN"], CONFIG["D_MODEL"],
        device=CONFIG["DEVICE"], dtype=CONFIG["DTYPE"]
    )
    w_ffn2 = torch.randn(
        CONFIG["D_MODEL"], CONFIG["D_FFN"],
        device=CONFIG["DEVICE"], dtype=CONFIG["DTYPE"]
    )
    
    sparsity_levels = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0]

    table = Table(title="MoIE Forward Pass Benchmark (CPU)")
    table.add_column("Sparsity", justify="right", style="cyan", no_wrap=True)
    table.add_column("Avg Active Experts", justify="right", style="magenta")
    table.add_column("Dense Time (ms)", justify="right", style="green")
    table.add_column("Sparse Time (ms)", justify="right", style="blue")
    table.add_column("Speedup", justify="right", style="red")

    for sparsity in sparsity_levels:
        # Simulate gating by creating a random mask
        active_mask = (torch.rand(
            CONFIG["BATCH_SIZE"] * CONFIG["SEQ_LEN"], CONFIG["D_FFN"],
            device=CONFIG["DEVICE"]
        ) < sparsity).to(CONFIG["DTYPE"])

        avg_active = active_mask.sum() / (CONFIG["BATCH_SIZE"] * CONFIG["SEQ_LEN"])

        # Benchmark dense forward
        start_time = time.perf_counter()
        for _ in range(CONFIG["NUM_RUNS"]):
            _ = forward_dense(x, w_ffn1, w_ffn2, active_mask.view(CONFIG["BATCH_SIZE"], CONFIG["SEQ_LEN"], -1))
        dense_time = (time.perf_counter() - start_time) * 1000 / CONFIG["NUM_RUNS"]
        
        # Benchmark sparse forward
        start_time = time.perf_counter()
        for _ in range(CONFIG["NUM_RUNS"]):
            _ = forward_sparse(x, w_ffn1, w_ffn2, active_mask)
        sparse_time = (time.perf_counter() - start_time) * 1000 / CONFIG["NUM_RUNS"]
        
        speedup = dense_time / sparse_time if sparse_time > 0 else float('inf')

        table.add_row(
            f"{sparsity*100:.1f}%",
            f"{avg_active:.2f}",
            f"{dense_time:.4f}",
            f"{sparse_time:.4f}",
            f"{speedup:.2f}x"
        )

    console.print(table)


if __name__ == "__main__":
    run_benchmark()