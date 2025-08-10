import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple

DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 512
HIDDEN_SIZE = 128
NUM_HEADS = 8
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
SEQ_LEN = 64
BATCH_SIZE = 4

class GatedMHALayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False)

    def forward(self, hidden_states: torch.Tensor, routing_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = rearrange(q, 'b t (h d) -> b h t d', h=NUM_HEADS)
        k = rearrange(k, 'b t (h d) -> b h t d', h=NUM_HEADS)
        v = rearrange(v, 'b t (h d) -> b h t d', h=NUM_HEADS)
        
        all_head_outputs = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        weighted_output = all_head_outputs * routing_weights.view(B, NUM_HEADS, 1, 1)

        combined_heads = rearrange(weighted_output, 'b h t d -> b t (h d)')
        final_output = self.o_proj(combined_heads)
        
        return final_output, all_head_outputs

class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.attn = GatedMHALayer()
        self.lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

    def forward(self, input_ids: torch.Tensor, routing_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.embedding(input_ids)
        attn_output, all_head_outputs = self.attn(hidden_states, routing_weights)
        logits = self.lm_head(attn_output)
        return logits, all_head_outputs

def main():
    print(f"Running on device: {DEVICE}")

    model = SimpleTransformer().to(DEVICE, dtype=DTYPE)
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    labels = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)

    activation_mask = torch.zeros(BATCH_SIZE, NUM_HEADS, device=DEVICE, dtype=DTYPE)
    activation_mask[0, 0] = 1
    activation_mask[0, 3] = 1
    activation_mask[1, 1] = 1
    activation_mask[1, 2] = 1
    activation_mask[1, 5] = 1
    activation_mask[2, :] = 1 
    activation_mask[3, 4] = 1
    print("--- Activation Mask (used as routing_weights) ---")
    print(activation_mask)
    print("-------------------------------------------------\n")

    logits, all_head_outputs = model(input_ids, activation_mask)
    all_head_outputs.requires_grad_(True)
    
    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), labels.view(-1))
    print(f"Loss: {loss.item():.4f}")

    print("\nAttempting to capture gradient for 'all_head_outputs'...")
    
    try:
        head_grads, = torch.autograd.grad(
            outputs=loss,
            inputs=all_head_outputs,
            retain_graph=True,
            allow_unused=False
        )
    except RuntimeError as e:
        print(f"\n❌ Gradient capture FAILED with RuntimeError:")
        print(e)
        return

    print("✅ Gradient capture SUCCEEDED!")
    print(f"Gradient shape: {head_grads.shape}")
    print(f"Gradient norm: {head_grads.norm().item():.4f}")
    
    print("\n--- Per-Head Gradient Norms Analysis ---")
    all_correct = True
    for b in range(BATCH_SIZE):
        print(f"  Batch {b}:")
        for h in range(NUM_HEADS):
            norm = head_grads[b, h].norm().item()
            is_active = activation_mask[b, h].item() > 0
            
            status = ""
            correct = False
            if is_active and norm > 1e-9:
                status = "✅ CORRECT (Active, Grad > 0)"
                correct = True
            elif not is_active and norm < 1e-9:
                status = "✅ CORRECT (Inactive, Grad == 0)"
                correct = True
            elif is_active and norm < 1e-9:
                status = "❌ WRONG (Active, Grad == 0)"
            else: # not is_active and norm > 0
                status = "❌ WRONG (Inactive, Grad > 0)"
            
            if not correct:
                all_correct = False

            print(f"    Head {h}: Active={is_active}, Norm={norm:.4e} -> {status}")
    
    print("\n------------------------------------------")
    if all_correct:
        print("✅ SUCCESS: Gradient sparsity matches activation mask.")
    else:
        print("❌ FAILURE: Gradient sparsity does not match activation mask.")
    print("------------------------------------------")


if __name__ == "__main__":
    main()
