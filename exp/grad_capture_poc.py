import torch
import torch.nn as nn
import torch.nn.functional as F
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16

# --- Minimal Model ---

class Config:
    hidden_size = 8
    intermediate_size = 16
    seq_len = 256
    batch_size = 16

class TinyExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.w2(F.gelu(self.w1(hidden_states)))

# --- PoC Main Logic ---

def run_grad_capture_poc():
    print("--- Starting Gradient Capture PoC ---")
    config = Config()
    
    # --- Setup ---
    expert = TinyExpert(config).to(DEVICE, dtype=DTYPE)
    hidden_states = torch.randn(config.batch_size, config.seq_len, config.hidden_size, device=DEVICE, dtype=DTYPE)
    labels = torch.randn(config.batch_size, config.seq_len, config.hidden_size, device=DEVICE, dtype=DTYPE)
    
    hidden_states_flat = hidden_states.view(-1, config.hidden_size)
    labels_flat = labels.view(-1, config.hidden_size)
    num_tokens = hidden_states_flat.shape[0]

    # --- Method 1: Manual Python Loop ---
    print("\n--- Method 1: Manual autograd.grad loop ---")
    grad_matrix_manual = torch.zeros(num_tokens, device=DEVICE, dtype=DTYPE)
    
    start_time = time.time()
    for i in range(num_tokens):
        token_input = hidden_states_flat[i:i+1]
        token_label = labels_flat[i:i+1]
        token_output = expert(token_input)
        loss = F.mse_loss(token_output, token_label, reduction='sum')
        grad_output, = torch.autograd.grad(loss, token_output, retain_graph=True)
        grad_matrix_manual[i] = torch.linalg.norm(grad_output.float(), dim=-1).to(DTYPE)
    end_time = time.time()
    
    print(f"Time taken: {end_time - start_time:.4f} seconds")

    # --- Method 2: Tensorized autograd.grad ---
    print("\n--- Method 2: Tensorized autograd.grad ---")
    
    start_time = time.time()
    expert_output = expert(hidden_states_flat)
    
    # We need a scalar loss to start the backward pass. We sum the per-token losses.
    # The key is that `autograd.grad` can compute the gradient of this sum
    # w.r.t to `expert_output` (a non-scalar), which will give us the per-token gradients.
    loss_tensorized = F.mse_loss(expert_output, labels_flat, reduction='sum')
    
    grad_output_tensorized, = torch.autograd.grad(loss_tensorized, expert_output)
    
    grad_matrix_tensorized = torch.linalg.norm(grad_output_tensorized.float(), dim=1).to(DTYPE)
    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.4f} seconds")

    # --- Comparison ---
    print("\n--- Comparison ---")
    are_close = torch.allclose(grad_matrix_manual, grad_matrix_tensorized, atol=1e-2)
    print(f"Are the two gradient matrices close? {are_close}")
    
    if are_close:
        print("\n✅ PoC successful. Tensorized autograd is validated.")
    else:
        print("\n❌ PoC FAILED. Methods produce different results.")
        diff = torch.abs(grad_matrix_manual - grad_matrix_tensorized)
        print(f"Max difference: {diff.max().item()}")
        print("Manual sample:  ", grad_matrix_manual[:5])
        print("Tensorized sample:", grad_matrix_tensorized[:5])


if __name__ == "__main__":
    run_grad_capture_poc()