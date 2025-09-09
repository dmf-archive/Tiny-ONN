import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from rich.console import Console

from .config import CONFIG
from .data import generate_arithmetic_data, TOKENIZER, IGNORE_INDEX
from .dynamic_infinite_head_attention import DynamicInfiniteHeadAttention
from .dynamic_infinite_expert import DynamicInfiniteExpert
from .sparse_bayesian_linear import SparseBayesianLinear

console = Console()

class MoIETransformerBlock(nn.Module):
    def __init__(self, d_model, d_ffn_factor, dropout=0.1, dtype=torch.float32):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, dtype=dtype)
        self.attn = DynamicInfiniteHeadAttention(d_model, dtype=dtype)
        self.ln2 = nn.LayerNorm(d_model, dtype=dtype)
        self.ffn = DynamicInfiniteExpert(d_model, d_ffn_factor, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_in = self.ln1(x)
        attn_out, _, attn_masked_tuple = self.attn(attn_in)
        x = x + self.dropout(attn_out)
        
        ffn_in = self.ln2(x)
        ffn_out, _, ffn_masked_tuple = self.ffn(ffn_in)
        x = x + self.dropout(ffn_out)
        
        return x, list(attn_masked_tuple) + list(ffn_masked_tuple)

class ReferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(CONFIG["VOCAB_SIZE"], CONFIG["D_MODEL"], dtype=CONFIG["DTYPE"])
        self.pos_embedding = nn.Embedding(CONFIG["SEQ_LEN"], CONFIG["D_MODEL"], dtype=CONFIG["DTYPE"])
        self.blocks = nn.ModuleList([MoIETransformerBlock(
            CONFIG["D_MODEL"], CONFIG["D_FFN_FACTOR"], dtype=CONFIG["DTYPE"]
        ) for _ in range(CONFIG["NUM_TRANSFORMER_BLOCKS"])])
        self.lm_head = nn.Linear(CONFIG["D_MODEL"], CONFIG["VOCAB_SIZE"], dtype=CONFIG["DTYPE"])

    def forward(self, x):
        tok_emb = self.embedding(x)
        pos = torch.arange(0, x.size(1), device=x.device)
        pos_emb = self.pos_embedding(pos)
        x = tok_emb + pos_emb
        
        all_masked_outputs = []
        for block in self.blocks:
            x, masked_outputs_from_block = block(x)
            all_masked_outputs.extend(masked_outputs_from_block)
            
        return self.lm_head(x), all_masked_outputs

def run_training_loop():
    model = ReferenceModel().to(device=CONFIG["DEVICE"], dtype=CONFIG["DTYPE"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["LR"], betas=(0.9, 0.98))
    
    console.print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters on {CONFIG['DEVICE']} with {CONFIG['DTYPE']}.")
    model.train()
    last_log_time = time.time()

    for step in range(CONFIG["TRAINING_STEPS"]):
        x, labels = generate_arithmetic_data(CONFIG["BATCH_SIZE"], CONFIG["SEQ_LEN"], CONFIG["OPERAND_RANGES"])
        x, labels = x.to(CONFIG["DEVICE"]), labels.to(CONFIG["DEVICE"])
        
        optimizer.zero_grad(set_to_none=True)
        logits, masked_outputs = model(x)
        
        loss_mask = (labels != TOKENIZER['PAD']) & (labels != IGNORE_INDEX)
        masked_labels = labels.masked_fill(~loss_mask, IGNORE_INDEX)
        main_loss = F.cross_entropy(logits.view(-1, CONFIG["VOCAB_SIZE"]), masked_labels.view(-1), ignore_index=IGNORE_INDEX)
        
        gate_loss = torch.tensor(0.0, device=x.device, dtype=CONFIG["DTYPE"])
        activation_rate = 0.0

        if masked_outputs:
            total_active = sum((mo.abs() > 1e-5).float().sum() for mo in masked_outputs if mo is not None)
            total_elements = sum(mo.numel() for mo in masked_outputs if mo is not None)
            activation_rate = (total_active / total_elements).item() if total_elements > 0 else 0.0

            surprise_grads = torch.autograd.grad(main_loss, masked_outputs, retain_graph=True, allow_unused=True)

            for grad_tensor in surprise_grads:
                if grad_tensor is not None:
                    surprise_per_neuron = grad_tensor.view(-1, grad_tensor.shape[-1]).norm(p=2, dim=0)
                    active_surprise = surprise_per_neuron[surprise_per_neuron > 1e-9]
                    if active_surprise.numel() > 0:
                        weighted_surprise = active_surprise * activation_rate
                        gate_loss += (-torch.log(weighted_surprise + 1e-9) * weighted_surprise).sum()
        
        avg_tau = torch.distributions.Categorical(logits=logits.detach()).entropy()[loss_mask].mean() if loss_mask.any() else torch.tensor(0.0)
        prior_std = torch.clamp(avg_tau, min=CONFIG["KL_PRIOR_EPSILON"])
        
        kl_loss = torch.tensor(0.0, device=x.device, dtype=CONFIG["DTYPE"])
        num_sbl_layers = 0
        for module in model.modules():
            if isinstance(module, SparseBayesianLinear):
                q_w = torch.distributions.Normal(module.mu_weight, F.softplus(module.sigma_weight))
                p_w = torch.distributions.Normal(torch.zeros_like(module.mu_weight), prior_std)
                kl_loss += torch.distributions.kl_divergence(q_w, p_w).mean()
                num_sbl_layers += 1
        if num_sbl_layers > 0: 
            kl_loss /= num_sbl_layers
        
        w_gate = 1.0 - torch.sigmoid(main_loss.detach())
        total_loss = main_loss + w_gate * gate_loss + kl_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step > 0 and step % CONFIG["LOG_INTERVAL"] == 0:
            end_time = time.time()
            steps_per_sec = CONFIG["LOG_INTERVAL"] / (end_time - last_log_time) if (end_time - last_log_time) > 0 else float('inf')
            last_log_time = end_time

            with torch.no_grad():
                main_acc = (logits.argmax(-1) == labels)[loss_mask].float().mean() if loss_mask.any() else torch.tensor(0.0)
                all_sigmas = [p for name, p in model.named_parameters() if 'sigma_weight' in name]
                all_gates = [p for name, p in model.named_parameters() if 'gate_param' in name]
                avg_sigma = torch.mean(torch.stack([F.softplus(s).mean() for s in all_sigmas])).item() if all_sigmas else 0
                avg_gate = torch.mean(torch.stack([g.mean() for g in all_gates])).item() if all_gates else 0
                
                log_str = (
                    f"Step {step:5d} | Loss(m/g/k): {main_loss.item():.3f}/{gate_loss.item():.3f}/{kl_loss.item():.3f} | "
                    f"Acc: {main_acc.item():.3f} | Avg σ/g: {avg_sigma:.4f}/{avg_gate:.4f} | "
                    f"τ/p_std: {avg_tau.item():.3f}/{prior_std.item():.3f} | Act%: {activation_rate*100:.2f} | "
                    f"it/s: {steps_per_sec:.2f}"
                )
                console.print(log_str)

if __name__ == "__main__":
    run_training_loop()