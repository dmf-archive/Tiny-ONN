import torch
import torch.nn as nn
import torch.nn.functional as F
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16

class Config:
    hidden_size = 8
    intermediate_size = 16
    seq_len = 16
    batch_size = 2

class TinyExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.w2(F.gelu(self.w1(hidden_states)))

def run_grad_disaggregation_poc():
    print("--- Starting Gradient Disaggregation PoC ---")
    config = Config()
    
    expert = TinyExpert(config).to(DEVICE, dtype=DTYPE)
    hidden_states = torch.randn(config.batch_size, config.seq_len, config.hidden_size, device=DEVICE, dtype=DTYPE)
    labels = torch.randn(config.batch_size, config.seq_len, config.hidden_size, device=DEVICE, dtype=DTYPE)
    
    hidden_states_flat = hidden_states.view(-1, config.hidden_size)
    labels_flat = labels.view(-1, config.hidden_size)
    num_tokens = hidden_states_flat.shape[0]

    # --- Method 1: Ground Truth (Manual Loop) ---
    print("\n--- Method 1: Ground Truth (Manual Loop) ---")
    # --- Method 1: Ground Truth (Per-token parameter grad norm) ---
    print("\n--- Method 1: Ground Truth (Per-token parameter grad norm) ---")
    per_token_grad_norms_manual = torch.zeros(num_tokens, device=DEVICE, dtype=DTYPE)
    for i in range(num_tokens):
        token_input = hidden_states_flat[i:i+1]
        token_label = labels_flat[i:i+1]
        token_output = expert(token_input)
        loss = F.mse_loss(token_output, token_label)
        grads = torch.autograd.grad(loss, expert.parameters(), retain_graph=True)
        per_token_grad_norms_manual[i] = torch.linalg.norm(torch.cat([g.flatten() for g in grads])).to(DTYPE)
    
    print(f"Per-token grad norms (Manual) sample: {per_token_grad_norms_manual[:5]}")

    # --- Method 2: Hook Approximation (grad_output norm) ---
    print("\n--- Method 2: Hook Approximation (grad_output norm) ---")
    expert.zero_grad()
    
    grad_norm_matrix_hook = torch.zeros(num_tokens, device=DEVICE, dtype=DTYPE)

    def disaggregate_grad_hook(module, grad_input, grad_output):
        nonlocal grad_norm_matrix_hook
        grad_norm_matrix_hook = torch.linalg.norm(grad_output[0].float(), dim=1).to(DTYPE)

    expert.register_full_backward_hook(disaggregate_grad_hook)

    # Main forward and backward pass
    output_full = expert(hidden_states_flat)
    loss_full = F.mse_loss(output_full, labels_flat)
    loss_full.backward()
    
    print(f"Per-token grad_output norms (Hook) sample: {grad_norm_matrix_hook[:5]}")
    
    # --- Comparison ---
    # This is an apples-to-oranges comparison, but we check if they are correlated.
    print("\n--- Correlation Analysis ---")
    correlation = torch.corrcoef(torch.stack([per_token_grad_norms_manual, grad_norm_matrix_hook]))[0, 1].item()
    print(f"Correlation between manual param grad norms and hook grad_output norms: {correlation:.4f}")

    if correlation > 0.8:
        print("\n✅ PoC successful. Hook grad_output norm is a strong proxy for param grad norm.")
    else:
        print("\n❌ PoC FAILED. Hook grad_output norm is not a reliable proxy.")

    # grad_norm_hook is not defined in this version, removing the print statement

    # --- Comparison ---
    print("\n--- Comparison ---")
    are_close = np.isclose(grad_norm_manual, grad_norm_hook)
    print(f"Are the two total gradient norms close? {are_close}")
    
    if are_close:
        print("\n✅ PoC successful. Hook disaggregation is validated.")
    else:
        print("\n❌ PoC FAILED. Methods produce different results.")

if __name__ == "__main__":
    run_grad_disaggregation_poc()