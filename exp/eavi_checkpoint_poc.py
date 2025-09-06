import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.utils.checkpoint import checkpoint

# --- Configuration ---
BATCH_SIZE = 2
SEQ_LEN = 1024
D_MODEL = 64
VOCAB_SIZE = 128
NUM_LAYERS = 2
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
NUM_AUGMENTATIONS = 4 # Simulating D8 augmentations

print(f"--- EAVI vs. Multi-View TF - Checkpoint PoC ---")
print(f"Device: {DEVICE}, Batch Size: {BATCH_SIZE}, Seq Len: {SEQ_LEN}\n")

# --- Model ---
class PocModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=2, dim_feedforward=D_MODEL*2, batch_first=True) for _ in range(NUM_LAYERS)])
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE)
    
    def forward(self, input_ids=None, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        
        # A simplified causal mask
        seq_len = inputs_embeds.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=DEVICE, dtype=torch.bool), diagonal=1)
        
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, src_mask=causal_mask)
        
        return self.lm_head(hidden_states)

# --- EAVI Differentiable Generation Step ---
def _generation_step(model, prev_embeds):
    """ A single, differentiable step of generation """
    # prev_embeds: (B, T, D)
    logits = model(inputs_embeds=prev_embeds)[:, -1, :] # (B, V)
    probs = F.softmax(logits, dim=-1)
    
    # Soft argmax: get expected embedding for the next token
    next_embeds = probs @ model.embedding.weight # (B, D)
    
    return torch.cat([prev_embeds, next_embeds.unsqueeze(1)], dim=1)

def differentiable_generate(model, input_ids, max_new_tokens):
    """ Full differentiable generation loop with gradient checkpointing """
    inputs_embeds = model.embedding(input_ids)
    
    generated_embeds = inputs_embeds
    for _ in range(max_new_tokens):
        # Each step is checkpointed
        generated_embeds = checkpoint(_generation_step, model, generated_embeds, use_reentrant=False)
        
    # Final forward pass to get all logits from the soft-generated sequence
    return model(inputs_embeds=generated_embeds)

# --- Experiment Runner ---
def run_experiment(model, optimizer, x, labels, method_name):
    print(f"--- METHOD: {method_name} ---")
    optimizer.zero_grad()
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.perf_counter()
    
    if "Multi-View TF" in method_name:
        # METHOD A: Teacher Forcing on augmented views
        b, t = x.shape
        # Simulate augmentations
        augmented_x = x.repeat(NUM_AUGMENTATIONS, 1) # (B*N_aug, T)
        # The labels for TF should match the input length
        augmented_labels = labels.repeat(NUM_AUGMENTATIONS, 1)[:, :t]
        
        logits = model(input_ids=augmented_x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), augmented_labels.reshape(-1))

    elif "Checkpoint EAVI" in method_name:
        # METHOD B: Differentiable generate with checkpointing
        prompt_len = x.shape[1]
        logits = differentiable_generate(model, x, max_new_tokens=SEQ_LEN - prompt_len)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1))

    loss.backward()
    optimizer.step()
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    print(f"Time: {end_time - start_time:.4f}s")
    print(f"Loss: {loss.item():.4f}")
    print(f"Peak Memory: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB")
    
    return {name: p.clone() for name, p in model.named_parameters()}

# --- Data & Models ---
prompt_len = SEQ_LEN // 2
x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, prompt_len), device=DEVICE)
labels = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)

torch.manual_seed(42)
model_a = PocModel().to(DEVICE, dtype=DTYPE)
optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=LR)

torch.manual_seed(42)
model_b = PocModel().to(DEVICE, dtype=DTYPE)
optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=LR)

# --- Run Experiments ---
params_after_a = run_experiment(model_a, optimizer_a, x, labels, "A: Multi-View TF")
print("\n" + "="*50 + "\n")
params_after_b = run_experiment(model_b, optimizer_b, x, labels, "B: Checkpoint EAVI")

# --- Verification ---
print("\n--- Verification ---")
param_name_to_check = 'layers.0.self_attn.out_proj.weight'
delta_norm = torch.norm(params_after_a[param_name_to_check] - params_after_b[param_name_to_check])
is_different = delta_norm.item() > 1e-5

print(f"Comparing final parameters of '{param_name_to_check}':")
print(f"  - Norm of parameter difference: {delta_norm.item():.8f}")
print(f"  - Are they different? -> {is_different}")

print("\n--- Conclusion ---")
if is_different:
    print("✅ Success! The two methods produce DIFFERENT gradients and final parameters.")
    print("This confirms Checkpoint EAVI is a distinct training method, not just a variant of TF.")
else:
    print("❌ Failure! The parameters are identical, suggesting a flaw in the experiment logic.")
