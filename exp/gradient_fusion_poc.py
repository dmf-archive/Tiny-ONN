import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

class PocExpert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))

class PocGate(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts)

    def forward(self, x):
        return self.gate(x)

class PocMoE(nn.Module):
    def __init__(self, d_model, n_experts, d_ff):
        super().__init__()
        self.experts = nn.ModuleList([PocExpert(d_model, d_ff) for _ in range(n_experts)])
        self.gate = PocGate(d_model, n_experts)

    def forward(self, x):
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        expert_outputs_stacked = torch.stack([expert(x) for expert in self.experts], dim=2)
        
        final_output = torch.einsum('bsn,bsne->bse', routing_weights, expert_outputs_stacked)
        return final_output, router_logits, expert_outputs_stacked

class PocModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_experts, d_ff):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.moe = PocMoE(d_model, n_experts, d_ff)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        moe_output, router_logits, expert_outputs = self.moe(x)
        final_logits = self.lm_head(moe_output)
        return final_logits, router_logits, expert_outputs

def run_poc():
    # --- Config ---
    vocab_size = 100
    d_model = 32
    d_ff = 64
    n_experts = 4
    batch_size = 5
    seq_len = 10
    smk_loss_weight = 0.1

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PocModel(vocab_size, d_model, n_experts, d_ff).to(device)
    
    expert_params = [p for n, p in model.named_parameters() if 'gate' not in n]
    selection_params = [p for n, p in model.named_parameters() if 'gate' in n]

    optimizer = AdamW([
        {'params': expert_params, 'lr': 1e-4},
        {'params': selection_params, 'lr': 1e-3},
    ])

    loss_fn = nn.CrossEntropyLoss()

    print("--- Running Gradient Fusion PoC ---")

    for step in range(5):
        # --- Data ---
        input_data = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        target_data = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        # --- Forward Pass ---
        output_logits, router_logits, expert_outputs = model(input_data)
        main_loss = loss_fn(output_logits.view(-1, vocab_size), target_data.view(-1))

        # --- Differentiable Surprise Calculation ---
        surprise_grads = torch.autograd.grad(
            main_loss, 
            expert_outputs, 
            grad_outputs=torch.ones_like(main_loss),
            create_graph=True,
            retain_graph=True
        )[0]
        
        surprise = torch.linalg.norm(surprise_grads.flatten(start_dim=3), dim=3)

        # --- Meta Loss Calculation ---
        with torch.no_grad():
            optimal_indices = torch.argmin(surprise, dim=-1).flatten()
        
        smk_loss = loss_fn(router_logits.view(-1, n_experts), optimal_indices)

        # --- Combined Loss and Backward Pass ---
        total_loss = main_loss + smk_loss * smk_loss_weight
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # --- Logging ---
        print(f"\n--- Step {step+1} ---")
        print(f"Total Loss: {total_loss.item():.4f}, Main Loss: {main_loss.item():.4f}, SMK Loss: {smk_loss.item():.4f}")
        
        expert_grad_norm = torch.stack([p.grad.norm() for p in expert_params if p.grad is not None]).norm().item()
        gate_grad_norm = torch.stack([p.grad.norm() for p in selection_params if p.grad is not None]).norm().item()
        
        print(f"Expert Grad Norm: {expert_grad_norm:.4f}")
        print(f"Gate Grad Norm: {gate_grad_norm:.4f}")

    print("\n--- PoC Finished ---")


if __name__ == "__main__":
    run_poc()