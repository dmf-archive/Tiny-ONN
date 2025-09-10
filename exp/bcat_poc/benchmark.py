import torch
import torch.nn.functional as F
import time
import math
from rich.console import Console
from rich.table import Table

from .sparse_bayesian_linear import SparseBayesianLinear as SBL_JIT_Dense
from .sparse_bayesian_linear import bcat_cluster
from .config import CONFIG

console = Console()

def benchmark(func, name, *args, **kwargs):
    warmup_steps = 10
    bench_steps = 50
    
    for _ in range(warmup_steps):
        func(*args, **kwargs)
        
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for _ in range(bench_steps):
        func(*args, **kwargs)
        
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    avg_time_ms = ((end_time - start_time) / bench_steps) * 1000
    console.print(f"[cyan]{name}[/cyan]: {avg_time_ms:.3f} ms")
    return avg_time_ms

@torch.no_grad()
def forward_jit_dense(sbl, x, raw_weights):
    computation_output = F.linear(x, sbl.mu_weight)
    _ = computation_output * raw_weights

@torch.no_grad()
def forward_loop_memcpy(sbl, x, raw_weights):
    block_meta, _ = bcat_cluster(raw_weights, sbl.bcat_grid_size, 0.0)
    
    if block_meta.numel() == 0:
        return

    total_x_rows = torch.sum(block_meta[:, 2]).item()
    total_mu_rows = torch.sum(block_meta[:, 3]).item()

    compact_x = torch.empty((int(total_x_rows), sbl.in_features), device=x.device, dtype=x.dtype)
    compact_mu = torch.empty((int(total_mu_rows), sbl.in_features), device=x.device, dtype=x.dtype)
    
    x_offset = 0
    mu_offset = 0
    for i in range(block_meta.shape[0]):
        r_start, c_start, r_len, c_len = block_meta[i].long()
        
        x_end = x_offset + r_len
        compact_x[x_offset:x_end] = x[r_start:r_start+r_len]
        x_offset = x_end
        
        mu_end = mu_offset + c_len
        compact_mu[mu_offset:mu_end] = sbl.mu_weight[c_start:c_start+c_len]
        mu_offset = mu_end

    # This GEMM is conceptually incorrect for the gathered slices,
    # but for a pure forward performance benchmark, it is a valid compute load.
    # A correct implementation would require a bmm like in the vectorized version.
    if compact_x.shape[0] > 0 and compact_mu.shape[0] > 0:
        # A placeholder computation that is shape-compatible
        _ = torch.matmul(compact_x, compact_mu.T.contiguous())

@torch.no_grad()
def forward_vectorized_memcpy(sbl, x, raw_weights):
    block_meta, _ = bcat_cluster(raw_weights, sbl.bcat_grid_size, 0.0)

    if block_meta.numel() == 0:
        return

    block_meta = block_meta.long()
    r_starts, c_starts, r_lens, c_lens = block_meta.T

    # Vectorized gather for x
    x_indices = torch.cat([torch.arange(start, start + length, device=x.device) for start, length in zip(r_starts, r_lens)])
    gathered_x = x.index_select(0, x_indices)

    # Vectorized gather for mu_weight
    mu_indices = torch.cat([torch.arange(start, start + length, device=x.device) for start, length in zip(c_starts, c_lens)])
    gathered_mu = sbl.mu_weight.index_select(0, mu_indices)

    # The gathered tensors are now "compact". We need to split and pad them for bmm.
    x_splits = torch.split(gathered_x, r_lens.tolist())
    mu_splits = torch.split(gathered_mu, c_lens.tolist())
    
    padded_x = torch.nn.utils.rnn.pad_sequence(x_splits, batch_first=True)
    padded_mu = torch.nn.utils.rnn.pad_sequence(mu_splits, batch_first=True)

    _ = torch.bmm(padded_x, padded_mu.transpose(1, 2))

def run_benchmark():
    device = CONFIG["DEVICE"]
    dtype = CONFIG["DTYPE"]
    d_model = CONFIG["D_MODEL"]
    d_ffn = d_model * CONFIG["D_FFN_FACTOR"]
    batch_size = CONFIG["BATCH_SIZE"]
    seq_len = CONFIG["SEQ_LEN"]
    
    sbl_baseline = SBL_JIT_Dense(d_ffn, d_model, bcat_grid_size=CONFIG["BCAT_GRID_SIZE"], dtype=dtype).to(device).eval()
    x = torch.randn(batch_size * seq_len, d_ffn, device=device, dtype=dtype)

    table = Table(title="BCAT Forward Pass Benchmark")
    table.add_column("Activation Rate (%)", justify="right", style="magenta")
    table.add_column("JIT Dense (ms)", justify="right", style="green")
    table.add_column("Loop Memcpy (ms)", justify="right", style="yellow")
    table.add_column("Vectorized Memcpy (ms)", justify="right", style="cyan")

    for act_rate in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
        console.print(f"\n--- Benchmarking for Activation Rate: {act_rate*100:.1f}% ---")
        
        num_elements = batch_size * seq_len * d_model
        num_active = int(num_elements * act_rate)
        
        raw_weights = torch.zeros(batch_size * seq_len, d_model, device=device, dtype=dtype)
        active_indices = torch.randperm(num_elements, device=device)[:num_active]
        raw_weights.view(-1)[active_indices] = torch.rand(num_active, device=device, dtype=dtype)

        jit_dense_time = benchmark(forward_jit_dense, "JIT Dense", sbl_baseline, x, raw_weights)
        loop_time = benchmark(forward_loop_memcpy, "Loop Memcpy", sbl_baseline, x, raw_weights)
        vec_time = benchmark(forward_vectorized_memcpy, "Vectorized Memcpy", sbl_baseline, x, raw_weights)
        
        # Placeholder for other implementations
        loop_time_str = f"{loop_time:.3f}"
        vec_time_str = f"{vec_time:.3f}"

        table.add_row(f"{act_rate*100:.1f}", f"{jit_dense_time:.3f}", loop_time_str, vec_time_str)

    console.print(table)

if __name__ == "__main__":
    run_benchmark()