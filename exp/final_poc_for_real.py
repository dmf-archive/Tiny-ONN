import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy

# --- Configuration ---
BATCH_SIZE = 4
SEQ_LEN = 64
D_IN = 128
VOCAB_SIZE = 256
NUM_EXPERTS = 8
INTERMEDIATE_SIZE = 16
NUM_LAYERS = 4  # A deeper, more realistic model
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

print(f"--- Final High-Fidelity PoC ---")
print(f"Device: {DEVICE}, Batch Size: {BATCH_SIZE}, Layers: {NUM_LAYERS}\n")

# --- Model Components ---
class SimplifiedMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(NUM_EXPERTS, D_IN, INTERMEDIATE_SIZE))
        self.w2 = nn.Parameter(torch.randn(NUM_EXPERTS, INTERMEDIATE_SIZE, D_IN))

    def forward(self, x, routing_weights):
        x_exp = x.unsqueeze(2).expand(-1, -1, NUM_EXPERTS, -1)
        y = torch.einsum('btec,eci->btei', x_exp, self.w1)
        y = F.gelu(y)
        moe_experts_out = torch.einsum('btei,eic->btec', y, self.w2)
        moe_agg = torch.einsum('btec,bte->btc', moe_experts_out, routing_weights)
        return moe_agg, moe_experts_out

class RealisticBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(D_IN)
        self.gating_net = nn.Linear(D_IN, NUM_EXPERTS)
        self.moe = SimplifiedMoE()

    def forward(self, x):
        residual = x
        x_norm = self.ln(x)
        gating_logits = self.gating_net(x_norm)
        routing_weights = F.softmax(gating_logits, dim=-1)
        
        moe_agg, moe_experts = self.moe(x_norm, routing_weights)
        
        return residual + moe_agg, gating_logits, moe_experts

class RealisticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_IN)
        self.layers = nn.ModuleList([RealisticBlock() for _ in range(NUM_LAYERS)])
        self.lm_head = nn.Linear(D_IN, VOCAB_SIZE)

    def forward(self, x):
        x = self.embedding(x)
        all_gating_logits = []
        all_moe_experts = []
        for layer in self.layers:
            x, gating_logits, moe_experts = layer(x)
            all_gating_logits.append(gating_logits)
            all_moe_experts.append(moe_experts)
        
        final_logits = self.lm_head(x)
        return final_logits, all_gating_logits, all_moe_experts

def create_model_and_optimizer(seed=0):
    torch.manual_seed(seed)
    model = RealisticModel().to(DEVICE, dtype=DTYPE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    return model, optimizer

def calculate_losses(logits, labels, all_moe_experts, all_gating_logits):
    main_loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), labels.view(-1))
    
    total_gating_loss = 0.0
    # This loop simulates the logic in our real train script
    for i in range(NUM_LAYERS):
        moe_experts = all_moe_experts[i]
        gating_logits = all_gating_logits[i]
        
        grads, = torch.autograd.grad(main_loss, moe_experts, retain_graph=True)
        surprise = torch.linalg.norm(grads, dim=-1) # B, T, E
        gating_loss = F.mse_loss(gating_logits, -surprise.detach())
        total_gating_loss += gating_loss
        
    return main_loss, total_gating_loss / NUM_LAYERS

# --- Dummy Data ---
x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
labels = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)

# --- METHOD 1: Standard Aggregated Update ---
print("--- 1. Standard Aggregated Update ---")
model_a, optimizer_a = create_model_and_optimizer()
torch.cuda.synchronize()
start_time = time.perf_counter()

logits_a, gatings_a, experts_a = model_a(x)
main_loss_a, gating_loss_a = calculate_losses(logits_a, labels, experts_a, gatings_a)
total_loss_a = main_loss_a + gating_loss_a
total_loss_a.backward()
optimizer_a.step()

torch.cuda.synchronize()
time_a = time.perf_counter() - start_time
print(f"Time: {time_a:.6f}s")
params_after_a = {name: p.clone() for name, p in model_a.named_parameters()}
del model_a, optimizer_a

# --- METHOD 2: Per-Sequence Update via Loop ---
print("\n--- 2. Per-Sequence AdamW Update (For-Loop) ---")
model_b, optimizer_b = create_model_and_optimizer()
torch.cuda.synchronize()
start_time = time.perf_counter()

for i in range(BATCH_SIZE):
    optimizer_b.zero_grad()
    x_sample, labels_sample = x[i:i+1], labels[i:i+1]
    
    logits_b, gatings_b, experts_b = model_b(x_sample)
    main_loss_b, gating_loss_b = calculate_losses(logits_b, labels_sample, experts_b, gatings_b)
    total_loss_b = main_loss_b + gating_loss_b
    total_loss_b.backward()
    optimizer_b.step()

torch.cuda.synchronize()
time_b = time.perf_counter() - start_time
print(f"Time: {time_b:.6f}s")
params_after_b = {name: p.clone() for name, p in model_b.named_parameters()}
del model_b, optimizer_b

# --- VERIFICATION ---
print("\n--- Verification ---")
param_name_to_check = 'layers.0.moe.w1'
delta_norm = torch.norm(params_after_a[param_name_to_check] - params_after_b[param_name_to_check])
is_different = delta_norm.item() > 1e-5 # Use a slightly more tolerant threshold for complex models

print(f"Comparing final parameters of '{param_name_to_check}':")
print(f"  - Norm of parameter difference: {delta_norm.item():.8f}")
print(f"  - Are they different? -> {is_different}")

if is_different:
    print("\n✅ Success! Per-sequence updates lead to a DIFFERENT final parameter state.")
else:
    print("\n❌ Failure! The final parameters are the same. Hypothesis is incorrect.")

print(f"\nPerformance ratio (Per-Seq Loop / Aggregated): {time_b/time_a:.2f}x slower")
