import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from rich.console import Console

from .config import CONFIG
from .data import generate_arithmetic_data, TOKENIZER, IGNORE_INDEX
from .model import ReferenceModel, SparseBayesianLinear

console = Console()


def run_training_loop():
    model = ReferenceModel().to(device=CONFIG["DEVICE"], dtype=CONFIG["DTYPE"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["LR"], betas=(0.9, 0.98))
    
    console.print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters on {CONFIG['DEVICE']} with {CONFIG['DTYPE']}.")
    model.train()
    last_log_time = time.time()
    prior_std = 1.0

    for step in range(CONFIG["TRAINING_STEPS"]):
        x, labels = generate_arithmetic_data(CONFIG["BATCH_SIZE"], CONFIG["SEQ_LEN"], CONFIG["OPERAND_RANGES"])
        x, labels = x.to(CONFIG["DEVICE"]), labels.to(CONFIG["DEVICE"])
        
        optimizer.zero_grad(set_to_none=True)

        kl_epsilon = CONFIG["KL_PRIOR_EPSILON"]
        logits, masked_outputs, kl_loss = model(x, prior_std, kl_epsilon)
        
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
        
        with torch.no_grad():
            avg_tau = torch.distributions.Categorical(logits=logits.detach()).entropy()[loss_mask].mean() if loss_mask.any() else torch.tensor(0.5)
            prior_std = torch.clamp(avg_tau.detach(), min=0.01, max=3.0).item()

        w_gate = 1.0 - torch.sigmoid(main_loss.detach())
        total_loss = main_loss + w_gate * gate_loss
        
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
                    f"τ/p_std: {avg_tau.item():.3f}/{prior_std:.3f} | Act%: {activation_rate*100:.2f} | "
                    f"it/s: {steps_per_sec:.2f}"
                )
                console.print(log_str)

if __name__ == "__main__":
    run_training_loop()