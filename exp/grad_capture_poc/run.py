import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple

# --- Constants ---
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 512
HIDDEN_SIZE = 128
NUM_HEADS = 8
HEAD_DIM = 16
SEQ_LEN = 64
BATCH_SIZE = 2

# --- Model Definition ---
class ManualMHALayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = rearrange(q, 'b t (h d) -> b h t d', h=NUM_HEADS)
        k = rearrange(k, 'b t (h d) -> b h t d', h=NUM_HEADS)
        v = rearrange(v, 'b t (h d) -> b h t d', h=NUM_HEADS)

        # We compute attention for all heads in a single vectorized call
        all_head_outputs = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        combined_heads = rearrange(all_head_outputs, 'b h t d -> b t (h d)')
        final_output = self.o_proj(combined_heads)
        
        return final_output, all_head_outputs

class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.attn = ManualMHALayer()
        self.lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.embedding(input_ids)
        attn_output, all_head_outputs = self.attn(hidden_states)
        logits = self.lm_head(attn_output)
        return logits, all_head_outputs

def main():
    print(f"Running on device: {DEVICE}")

    model = SimpleTransformer().to(DEVICE, dtype=DTYPE)
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    labels = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)

    # --- Forward Pass ---
    logits, all_head_outputs = model(input_ids)
    
    # Ensure the intermediate tensor requires grad
    # This is often not needed if it's the output of a module with grad-requiring params,
    # but we do it explicitly for this test.
    all_head_outputs.requires_grad_(True)

    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), labels.view(-1))
    print(f"Loss: {loss.item():.4f}")

    # --- Gradient Capture Attempt ---
    print("\nAttempting to capture gradient for 'all_head_outputs'...")
    
    head_grads, = torch.autograd.grad(
        outputs=loss,
        inputs=all_head_outputs,
        retain_graph=True,
        allow_unused=False # Set to False to raise an error if grad is None
    )

    if head_grads is not None:
        print("✅ Gradient capture SUCCESSFUL!")
        print(f"Gradient shape: {head_grads.shape}")
        print(f"Gradient norm: {head_grads.norm().item():.4f}")
        print(f"Gradient mean: {head_grads.mean().item():.4f}")
        print(f"Per-head gradient norms: {[head_grads[:, i].norm().item() for i in range(NUM_HEADS)]}")
    else:
        # This part should not be reached if allow_unused=False
        print("❌ Gradient capture FAILED. Gradient is None.")

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"\n❌ Gradient capture FAILED with RuntimeError:")
        print(e)