import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import PretrainedConfig

from tiny_onn.config import TinyOnnConfig
from tiny_onn.modular import TinyOnnForCausalLM

def get_tiny_onn_model():
    config = TinyOnnConfig(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=1000,
        num_experts_per_layer=4,
        moe_intermediate_size=16,
    )
    model = TinyOnnForCausalLM(config)
    return model

def run_poc():
    # --- Config ---
    batch_size = 2
    seq_len = 8
    smk_loss_weight = 0.1

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_tiny_onn_model().to(device)
    
    expert_params = [p for n, p in model.named_parameters() if 'gate' not in n]
    selection_params = [p for n, p in model.named_parameters() if 'gate' in n]

    optimizer = AdamW([
        {'params': expert_params, 'lr': 1e-4},
        {'params': selection_params, 'lr': 1e-3},
    ])

    loss_fn = torch.nn.CrossEntropyLoss()
    print("--- Running Final Corrected TinyONN PoC (2-Backward Pass) ---")

    for step in range(5):
        # --- Data ---
        input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
        labels = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids)
        
        # --- Forward Pass & Surprise Context Setup ---
        surprise_context = {}
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            surprise_context=surprise_context,
            output_router_logits=True,
        )
        main_loss = outputs.loss
        
        # --- Decoupled Backward Passes ---
        optimizer.zero_grad()
        
        # 1. First backward for main_loss (populates surprise_context and expert_params.grad)
        main_loss.backward(retain_graph=True) # retain_graph is not strictly needed if smk_loss graph is separate
        
        # 2. Calculate smk_loss using the populated surprise_context
        all_router_logits = [layer.mlp.last_router_logits for layer in model.model.layers]
        concatenated_logits = torch.cat(all_router_logits, dim=0)
        
        per_token_surprise = torch.full_like(concatenated_logits, float("inf"), device=device)
        
        # We need to map layer-local token indices to global indices
        layer_token_offsets = {}
        current_offset = 0
        for i, layer in enumerate(model.model.layers):
            layer_token_offsets[i] = current_offset
            current_offset += layer.mlp.last_router_logits.shape[0]

        for (layer_idx, expert_idx), (token_indices, surprise) in surprise_context.items():
            offset = layer_token_offsets[layer_idx]
            per_token_surprise[offset + token_indices, expert_idx] = surprise.to(per_token_surprise.dtype)

        with torch.no_grad():
            optimal_indices = torch.argmin(per_token_surprise, dim=1)
        
        # Ensure smk_loss is computed within the gradient tape
        smk_loss = loss_fn(concatenated_logits, optimal_indices)
        
        # 3. Second backward for smk_loss (accumulates grad on selection_params.grad)
        (smk_loss * smk_loss_weight).backward()

        # --- Optimizer Step ---
        optimizer.step()

        # --- Logging ---
        print(f"\n--- Step {step+1} ---")
        print(f"Main Loss: {main_loss.item():.4f}, SMK Loss: {smk_loss.item():.4f}")
        
        expert_grad_norm = torch.stack([p.grad.norm() for p in expert_params if p.grad is not None]).norm().item()
        gate_grad_norm = torch.stack([p.grad.norm() for p in selection_params if p.grad is not None]).norm().item()
        
        print(f"Expert Grad Norm: {expert_grad_norm:.4f}")
        print(f"Gate Grad Norm: {gate_grad_norm:.4f}")

    print("\n--- PoC Finished ---")

if __name__ == "__main__":
    run_poc()