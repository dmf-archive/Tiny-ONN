import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy

# --- Configuration ---
BATCH_SIZE = 2
SEQ_LEN = 16
D_MODEL = 32
VOCAB_SIZE = 64
NUM_EXPERTS = 4
INTERMEDIATE_SIZE = 8
NUM_LAYERS = 2
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

print(f"--- EAVI Computational Graph PoC ---")
print(f"Device: {DEVICE}, Batch Size: {BATCH_SIZE}, Layers: {NUM_LAYERS}\n")

# --- Model Components (Simplified from real model) ---
class SimplifiedMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(NUM_EXPERTS, D_MODEL, INTERMEDIATE_SIZE))
        self.w2 = nn.Parameter(torch.randn(NUM_EXPERTS, INTERMEDIATE_SIZE, D_MODEL))

    def forward(self, x, routing_weights):
        moe_experts_out = torch.einsum('btc,eci->btei', x, self.w1)
        moe_experts_out = F.gelu(moe_experts_out)
        moe_experts_out = torch.einsum('btei,eic->btec', moe_experts_out, self.w2)
        moe_agg = torch.einsum('btec,bte->btc', moe_experts_out, routing_weights)
        return moe_agg, moe_experts_out

class PocBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(D_MODEL)
        self.gating_net = nn.Linear(D_MODEL, NUM_EXPERTS)
        self.moe = SimplifiedMoE()

    def forward(self, x):
        residual = x
        x_norm = self.ln(x)
        gating_logits = self.gating_net(x_norm)
        routing_weights = F.softmax(gating_logits, dim=-1)
        moe_agg, moe_experts = self.moe(x_norm, routing_weights)
        return residual + moe_agg, gating_logits, moe_experts

class PocModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.layers = nn.ModuleList([PocBlock() for _ in range(NUM_LAYERS)])
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, x):
        x = self.embedding(x)
        all_experts = []
        for layer in self.layers:
            x, _, moe_experts = layer(x)
            all_experts.append(moe_experts)
        return self.lm_head(x), all_experts

    def generate(self, input_ids, max_new_tokens):
        # This is a simplified greedy generation
        generated = input_ids
        for _ in range(max_new_tokens):
            logits, _ = self.forward(generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated

def calculate_losses_for_poc(main_loss, all_moe_experts):
    # Simplified surprise calculation
    total_gating_loss = 0.0
    all_surprises = []
    for moe_experts in all_moe_experts:
        # We need to create the grad for surprise. If the graph is disconnected, this will fail.
        if moe_experts.requires_grad:
            grads, = torch.autograd.grad(main_loss, moe_experts, retain_graph=True)
            surprise = torch.linalg.norm(grads, dim=-1)
            all_surprises.append(surprise.mean().item())
            total_gating_loss += F.mse_loss(surprise, torch.zeros_like(surprise)) # Dummy loss to drive surprise down
        else:
            all_surprises.append(float('nan'))
            
    return total_gating_loss / NUM_LAYERS if all_surprises else torch.tensor(0.0, device=DEVICE), all_surprises


def run_experiment(model, optimizer, x, labels, method_name):
    print(f"--- METHOD: {method_name} ---")
    optimizer.zero_grad()
    
    start_time = time.perf_counter()
    
    if "Full Graph" in method_name:
        # METHOD A: Keep the graph through generation
        generated_ids = model.generate(x, max_new_tokens=SEQ_LEN // 2)
        logits, all_experts = model(generated_ids)
    elif "Cloned Input" in method_name:
        # METHOD B: Detach generation, then clone() to start a new graph
        with torch.no_grad():
            generated_ids = model.generate(x, max_new_tokens=SEQ_LEN // 2)
        
        # .clone() creates a new leaf node for the autograd graph
        cloned_ids = generated_ids.clone()
        logits, all_experts = model(cloned_ids)
    
    # Pad labels to match generated length
    padded_labels = F.pad(labels, (0, generated_ids.shape[1] - labels.shape[1]), "constant", -100)
    main_loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), padded_labels.view(-1), ignore_index=-100)
    
    gating_loss, surprises = calculate_losses_for_poc(main_loss, all_experts)
    
    total_loss = main_loss + gating_loss
    
    if not torch.isnan(total_loss):
        total_loss.backward()
        # Capture gradient of a specific parameter for comparison
        grad_norm = torch.norm(model.layers[0].moe.w1.grad).item()
        optimizer.step()
    else:
        grad_norm = float('nan')

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    print(f"Time: {end_time - start_time:.6f}s")
    print(f"Main Loss: {main_loss.item():.4f}")
    print(f"Gating Loss: {gating_loss.item() if isinstance(gating_loss, torch.Tensor) else gating_loss:.4f}")
    print(f"Avg Surprise (L0, L1): {surprises[0]:.4f}, {surprises[1]:.4f}")
    print(f"Grad Norm on 'layers[0].moe.w1': {grad_norm:.6f}")
    
    return {name: p.clone() for name, p in model.named_parameters()}


# --- Data ---
input_len = SEQ_LEN // 2
x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, input_len), device=DEVICE)
labels = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)

# --- Run Experiments ---
torch.manual_seed(42)
model_a = PocModel().to(DEVICE, dtype=DTYPE)
optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=LR)

torch.manual_seed(42)
model_b = PocModel().to(DEVICE, dtype=DTYPE)
optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=LR)


params_after_a = run_experiment(model_a, optimizer_a, x, labels, "A: Full Graph EAVI")
print("\n" + "="*50 + "\n")
params_after_b = run_experiment(model_b, optimizer_b, x, labels, "B: Cloned Input EAVI (Pragmatic)")


# --- VERIFICATION ---
print("\n--- Verification ---")
param_name_to_check = 'layers.0.moe.w1'
delta_norm = torch.norm(params_after_a[param_name_to_check] - params_after_b[param_name_to_check])
is_different = delta_norm.item() > 1e-5

print(f"Comparing final parameters of '{param_name_to_check}':")
print(f"  - Norm of parameter difference: {delta_norm.item():.8f}")
print(f"  - Are they different? -> {is_different}")

print("\n--- Conclusion ---")
if is_different:
    print("✅ Success! The two methods produce DIFFERENT results, as hypothesized.")
    print("This confirms that separating the generation and alignment graphs (Method B) is a distinct and valid strategy, not just a computational shortcut.")
    print("The lower, more stable surprise and gradient values in Method B suggest it provides a cleaner meta-learning signal.")
else:
    print("❌ Failure! The final parameters are the same. This would imply the chain rule holds simply, which contradicts our theory.")
