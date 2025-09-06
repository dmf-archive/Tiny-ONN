import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from rich.console import Console
from rich.progress import track
import math
import random
import os

CONFIG = {
    "BATCH_SIZE": 16, "SEQ_LEN": 64, "D_MODEL": 64, "VOCAB_SIZE": 18,
    "NUM_HEADS": 4, "NUM_TRANSFORMER_BLOCKS": 2, 
    "LR": 3e-3, "DEVICE": "cpu", "DTYPE": torch.float32,
    "TRAINING_STEPS": 50000, "DATASET_SIZE": 16384
}
CKPT_PATH = "latest.pt"
console = Console()
console.print(f"Using device: {CONFIG['DEVICE']}")

TOKENIZER = {
    'BOS': 0, 'EOS': 1,
    '0': 2, '1': 3, '2': 4, '3': 5, '4': 6, '5': 7, '6': 8, '7': 9, '8': 10, '9': 11,
    '+': 12, '-': 13, '×': 14, '÷': 15, '=': 16, 'PAD': 17
}
REVERSE_TOKENIZER = {v: k for k, v in TOKENIZER.items()}
IGNORE_INDEX = -100

def generate_arithmetic_data(num_samples, seq_len, operand_range=(0, 9999)):
    sents, labels = [], []
    while len(sents) < num_samples:
        a = random.randint(*operand_range)
        b = random.randint(*operand_range)
        op_str = random.choice(['+', '-', '×'])
        
        try:
            if op_str == '×': res = a * b
            elif op_str == '-':
                if a < b: continue
                res = a - b
            else: res = a + b
        except OverflowError: continue

        problem = [TOKENIZER[d] for d in str(a)] + [TOKENIZER[op_str]] + [TOKENIZER[d] for d in str(b)] + [TOKENIZER['=']]
        full_seq = [TOKENIZER['BOS']] + problem + [TOKENIZER[d] for d in str(res)] + [TOKENIZER['EOS']]
        
        if len(full_seq) >= seq_len: continue

        x, y = full_seq[:-1], full_seq[1:]
        x.extend([TOKENIZER['PAD']] * (seq_len - len(x)))
        y.extend([TOKENIZER['PAD']] * (seq_len - len(y)))
        
        sents.append(torch.tensor(x))
        labels.append(torch.tensor(y))

    return torch.stack(sents).long(), torch.stack(labels).long()

class MoIELinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.gate_param = nn.Parameter(torch.Tensor(out_features))
        self.mu_bias = nn.Parameter(torch.Tensor(out_features))

        nn.init.kaiming_uniform_(self.mu_weight, a=math.sqrt(5))
        nn.init.normal_(self.sigma_weight, mean=0.0, std=0.02)
        nn.init.constant_(self.gate_param, 0.1)
        nn.init.zeros_(self.mu_bias)

    def forward(self, x):
        original_shape = x.shape
        x_reshaped = x.view(-1, self.in_features)

        keys = self.mu_weight * F.softplus(self.sigma_weight)
        
        scores = F.cosine_similarity(x_reshaped.unsqueeze(1), keys.unsqueeze(0), dim=-1)
        
        weights = F.relu(scores - self.gate_param.unsqueeze(0))
        
        computation_output = F.linear(x_reshaped, self.mu_weight, self.mu_bias)
        
        masked_output = computation_output * weights
        
        output = masked_output.view(*original_shape[:-1], self.out_features)

        return output, (scores, masked_output)

class MoIETransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(CONFIG["D_MODEL"])
        self.attn = nn.MultiheadAttention(CONFIG["D_MODEL"], CONFIG["NUM_HEADS"], batch_first=True)
        self.ln2 = nn.LayerNorm(CONFIG["D_MODEL"])
        self.ffn1 = MoIELinear(CONFIG["D_MODEL"], CONFIG["D_MODEL"] * 4)
        self.ffn2 = nn.Linear(CONFIG["D_MODEL"] * 4, CONFIG["D_MODEL"])

    def forward(self, x):
        attn_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        
        ffn_in = self.ln2(x)
        ffn1_out, (ffn1_scores, ffn1_masked_out) = self.ffn1(ffn_in)
        ffn1_out_activated = F.silu(ffn1_out)
        ffn2_out = self.ffn2(ffn1_out_activated)
        x = x + ffn2_out
        
        return x, (ffn1_scores,), (ffn1_masked_out,)

class PocModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(CONFIG["VOCAB_SIZE"], CONFIG["D_MODEL"])
        self.pos_embedding = nn.Embedding(CONFIG["SEQ_LEN"], CONFIG["D_MODEL"])
        self.blocks = nn.ModuleList([MoIETransformerBlock() for _ in range(CONFIG["NUM_TRANSFORMER_BLOCKS"])])
        self.lm_head = nn.Linear(CONFIG["D_MODEL"], CONFIG["VOCAB_SIZE"])

    def forward(self, x):
        tok_emb = self.embedding(x)
        pos = torch.arange(0, x.size(1), device=x.device)
        pos_emb = self.pos_embedding(pos)
        x = tok_emb + pos_emb
        
        all_scores, all_masked_outputs = [], []
        
        for block in self.blocks:
            x, scores, masked_outputs = block(x)
            all_scores.extend(scores)
            all_masked_outputs.extend(masked_outputs)

        return self.lm_head(x), all_scores, all_masked_outputs

def evaluate_ood(model, ood_dataset_x, ood_dataset_y):
    # This function remains unchanged for now.
    pass

def run_experiment():
    model = PocModel().to(device=CONFIG["DEVICE"], dtype=CONFIG["DTYPE"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LR"])
    
    start_step = 0
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step'] + 1

    dataset_x, dataset_y = generate_arithmetic_data(CONFIG["DATASET_SIZE"], CONFIG["SEQ_LEN"], operand_range=(0, 900))
    dataset_x, dataset_y = dataset_x.to(CONFIG["DEVICE"]), dataset_y.to(CONFIG["DEVICE"])

    model.train()
    for step in range(start_step, CONFIG["TRAINING_STEPS"]):
        batch_indices = torch.randint(0, dataset_x.size(0), (CONFIG["BATCH_SIZE"],))
        x, labels = dataset_x[batch_indices], dataset_y[batch_indices]
        
        optimizer.zero_grad()
        
        logits, scores, masked_outputs = model(x)
        
        loss_mask = torch.zeros_like(labels, dtype=torch.bool)
        for i, row in enumerate(x):
            equals_pos = (row == TOKENIZER['=']).nonzero(as_tuple=True)[0]
            if equals_pos.numel() > 0:
                loss_mask[i, equals_pos[0].item() + 1:] = True
        
        masked_labels = labels.masked_fill(~loss_mask, IGNORE_INDEX)

        main_loss = F.cross_entropy(logits.view(-1, CONFIG["VOCAB_SIZE"]), masked_labels.view(-1), ignore_index=IGNORE_INDEX)
        
        if not masked_outputs: continue
        
        surprise_grads = torch.autograd.grad(main_loss, masked_outputs, retain_graph=True, allow_unused=True)
        
        f_min_loss = torch.tensor(0.0, device=x.device)
        total_active_neurons = 0
        for i, grad_tensor in enumerate(surprise_grads):
            if grad_tensor is not None:
                surprise = grad_tensor.abs().sum(dim=0)
                gate_param = model.blocks[i // 1].ffn1.gate_param # Simplified mapping
                f_min_loss += F.mse_loss(gate_param, surprise)
                total_active_neurons += (masked_outputs[i] != 0).float().sum()

        weighted_f_min_loss = 0.1 * f_min_loss
        total_loss = main_loss + weighted_f_min_loss
        total_loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            with torch.no_grad():
                accuracy = (logits.argmax(-1) == labels)[loss_mask].float().mean()
                
                # --- Re-add detailed logging ---
                all_sigmas = [p for name, p in model.named_parameters() if 'sigma_weight' in name]
                all_gates = [p for name, p in model.named_parameters() if 'gate_param' in name]

                avg_sigma = torch.mean(torch.stack([s.mean() for s in all_sigmas])).item()
                avg_gate = torch.mean(torch.stack([g.mean() for g in all_gates])).item()

                total_active_elements = sum((mo != 0).float().sum() for mo in masked_outputs)
                total_elements = sum(mo.numel() for mo in masked_outputs)
                activation_rate = (total_active_elements / total_elements * 100).item()

                tau_val = torch.distributions.Categorical(logits=logits.detach()).entropy()[loss_mask].mean().item()
                
                alpha_val = activation_rate / 100.0
                
                all_mus = [p for name, p in model.named_parameters() if 'mu_weight' in name]
                
                avg_mu_sigma = torch.mean(torch.stack([(m * F.softplus(s)).mean() for m, s in zip(all_mus, all_sigmas)])).item()
                gamma_val = avg_mu_sigma - avg_gate
                
                surprise_val = f_min_loss.item() if isinstance(f_min_loss, torch.Tensor) else 0.0
                
                pi_score = math.exp(-alpha_val * ((1 - gamma_val) * (main_loss.item() / (tau_val + 1e-9)) + gamma_val * surprise_val))

                console.print(f"Step {step:5d} | Loss(m/f): {main_loss.item():.3f}/{surprise_val:.3f} | Acc: {accuracy.item():.3f} | Avg σ/g: {avg_sigma:.4f}/{avg_gate:.4f} | Act%: {activation_rate:.2f} | PI/α/γ: {pi_score:.3f}/{alpha_val:.2f}/{gamma_val:.2f}")
            
            torch.save({
                'step': step, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, CKPT_PATH)

if __name__ == "__main__":
    run_experiment()