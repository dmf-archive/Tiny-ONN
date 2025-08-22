import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
from torch.optim import AdamW

# --- Shared Components & Config ---
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 512
HIDDEN_SIZE = 128
SEQ_LEN = 64
BATCH_SIZE = 4
NUM_ATTENTION_HEADS = 8
MAX_MOE_EXPERTS = 32
INTERMEDIATE_SIZE = 8

class Config:
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTENTION_HEADS
    max_moe_experts = MAX_MOE_EXPERTS
    intermediate_size = INTERMEDIATE_SIZE

# --- Model Definition (Simplified) ---
class DynamicMoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(config.max_moe_experts, config.hidden_size, config.intermediate_size))
        self.w2 = nn.Parameter(torch.randn(config.max_moe_experts, config.intermediate_size, config.hidden_size))
    def forward(self, x, routing_weights):
        y = torch.einsum('btc,eci->btei', x, self.w1)
        y = F.gelu(y)
        y = torch.einsum('btei,eic->btec', y, self.w2) # Per-expert outputs
        y_agg = torch.einsum('btec,bte->btc', y, routing_weights)
        return y_agg, y # Return both aggregated and per-expert outputs

class SimpleModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, config.hidden_size)
        self.moe = DynamicMoELayer(config)
        self.lm_head = nn.Linear(config.hidden_size, VOCAB_SIZE)
    def forward(self, x, routing_weights):
        x = self.embedding(x)
        moe_agg, moe_experts = self.moe(x, routing_weights)
        logits = self.lm_head(moe_agg)
        return logits, moe_experts

# --- Surprise Calculation Methods ---

def measure_gradient_surprise(model, input_ids, labels, routing_weights):
    logits, moe_experts = model(input_ids, routing_weights)
    moe_experts.requires_grad_(True)
    main_loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), labels.view(-1))
    
    grads, = torch.autograd.grad(main_loss, moe_experts, retain_graph=True)
    surprise_per_token_per_expert = torch.linalg.norm(grads.view(BATCH_SIZE * SEQ_LEN, MAX_MOE_EXPERTS, -1), dim=-1)
    return surprise_per_token_per_expert, main_loss

def measure_kl_surprise(model, input_ids, labels, routing_weights, lr=1e-3):
    # 1. Store original params & create optimizer for copy
    params_before = {name: p.clone() for name, p in model.moe.named_parameters()}
    optimizer_copy = AdamW(model.parameters(), lr=lr)
    
    # 2. First forward and backward to get grads on original model
    logits, _ = model(input_ids, routing_weights)
    main_loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), labels.view(-1))
    main_loss.backward()
    
    # 3. Apply update
    optimizer_copy.step()
    
    # 4. Calculate parameter shift (vectorized)
    with torch.no_grad():
        w1_shift = torch.linalg.norm((params_before['w1'] - model.moe.w1).view(MAX_MOE_EXPERTS, -1), dim=1)
        w2_shift = torch.linalg.norm((params_before['w2'] - model.moe.w2).view(MAX_MOE_EXPERTS, -1), dim=1)
        surprise_per_expert = w1_shift + w2_shift

    # 5. Reset model to original state for fair comparison
    with torch.no_grad():
        for name, p in model.moe.named_parameters():
            p.copy_(params_before[name])
    
    return surprise_per_expert, main_loss

# --- Measurement Runner ---

def run_benchmark():
    config = Config()
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    labels = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    routing_weights = F.softmax(torch.rand(BATCH_SIZE, SEQ_LEN, MAX_MOE_EXPERTS, device=DEVICE), dim=-1).to(DTYPE)

    print("--- Corrected Benchmarking of Surprise Calculation ---")

    # --- Gradient Method ---
    torch.cuda.reset_peak_memory_stats(DEVICE)
    model_grad = SimpleModel(config).to(DEVICE, dtype=DTYPE)
    
    start_time = time.perf_counter()
    surprise_grad, _ = measure_gradient_surprise(model_grad, input_ids, labels, routing_weights)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    grad_time = end_time - start_time
    grad_mem = torch.cuda.max_memory_allocated(DEVICE) / 1e6
    print(f"1. Gradient Norm Method (Corrected):")
    print(f"   - Time: {grad_time:.4f} seconds")
    print(f"   - Peak Memory: {grad_mem:.2f} MB")
    print(f"   - Surprise Shape: {surprise_grad.shape}") # Should be (B*T, E)
    del model_grad, surprise_grad

    # --- KL / Parameter Shift Method (Corrected) ---
    torch.cuda.reset_peak_memory_stats(DEVICE)
    model_kl = SimpleModel(config).to(DEVICE, dtype=DTYPE)

    start_time = time.perf_counter()
    surprise_kl, _ = measure_kl_surprise(model_kl, input_ids, labels, routing_weights)
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    kl_time = end_time - start_time
    kl_mem = torch.cuda.max_memory_allocated(DEVICE) / 1e6
    print(f"\n2. Parameter Shift Method (Vectorized, No deepcopy):")
    print(f"   - Time: {kl_time:.4f} seconds")
    print(f"   - Peak Memory: {kl_mem:.2f} MB")
    print(f"   - Surprise Shape: {surprise_kl.shape}") # Should be (E,)
    del model_kl, surprise_kl

if __name__ == "__main__":
    run_benchmark()