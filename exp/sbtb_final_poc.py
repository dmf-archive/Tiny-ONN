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
            if op_str == '×':
                res = a * b
            elif op_str == '-':
                if a < b: continue # Skip negative results for subtraction
                res = a - b
            else:
                res = a + b
        except OverflowError: # Handle potential overflow for large numbers
            continue

        a_tokens = [TOKENIZER[d] for d in str(a)]
        b_tokens = [TOKENIZER[d] for d in str(b)]
        res_tokens = [TOKENIZER[d] for d in str(res)]
        
        problem = a_tokens + [TOKENIZER[op_str]] + b_tokens + [TOKENIZER['=']]
        full_seq = [TOKENIZER['BOS']] + problem + res_tokens + [TOKENIZER['EOS']]
        
        if len(full_seq) >= seq_len:
            continue

        x = full_seq[:-1]
        y = full_seq[1:]
        
        x.extend([TOKENIZER['PAD']] * (seq_len - len(x)))
        y.extend([TOKENIZER['PAD']] * (seq_len - len(y)))
        
        sents.append(torch.tensor(x))
        labels.append(torch.tensor(y))

    return torch.stack(sents).long(), torch.stack(labels).long()

class STEGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sigma, gate):
        return (sigma > gate).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_sigma = grad_output.clone()
        grad_gate = -grad_output.clone()
        return grad_sigma, grad_gate

class SparseBayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.rho_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.gate_weight = nn.Parameter(torch.Tensor(out_features, in_features))

        self.mu_bias = nn.Parameter(torch.Tensor(out_features))
        self.rho_bias = nn.Parameter(torch.Tensor(out_features))
        self.gate_bias = nn.Parameter(torch.Tensor(out_features))

        nn.init.kaiming_uniform_(self.mu_weight, a=math.sqrt(5))
        nn.init.normal_(self.rho_weight, mean=1.85, std=0.1)
        nn.init.constant_(self.gate_weight, 0.5)

        nn.init.zeros_(self.mu_bias)
        nn.init.normal_(self.rho_bias, mean=1.85, std=0.1)
        nn.init.constant_(self.gate_bias, 0.5)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        outputs = []
        all_sigmas = []
        all_masked_weights = []

        proto_keys = F.softplus(self.rho_weight)
        thresholds = self.gate_weight.mean(dim=1)

        for b in range(batch_size):
            seq_outputs = []
            for s in range(seq_len):
                token_vector = x[b, s, :]
                
                scores = torch.matmul(token_vector, proto_keys.T)
                
                neuron_mask = STEGate.apply(scores, thresholds)
                
                masked_mu_weight = self.mu_weight * neuron_mask.unsqueeze(1)
                sparse_weight = masked_mu_weight.to_sparse()
                
                output_token = torch.sparse.mm(sparse_weight, token_vector.unsqueeze(1)).squeeze(1)
                
                # Bias is not masked for simplicity in this sparse implementation
                output_token += self.mu_bias

                seq_outputs.append(output_token)

                # For SML loss, we need to collect per-token sigmas and masked_weights
                all_sigmas.append(scores)
                all_masked_weights.append(masked_mu_weight)

            outputs.append(torch.stack(seq_outputs, dim=0))
        
        output_tensor = torch.stack(outputs, dim=0)

        # Stacking sigmas and weights for the loss function
        # Note: This part needs careful handling in the main training loop
        sigma_for_loss = torch.stack(all_sigmas)
        masked_weights_for_loss = torch.stack(all_masked_weights)

        return output_tensor, (sigma_for_loss, masked_weights_for_loss)

class SparseBayesianAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = SparseBayesianLinear(CONFIG["D_MODEL"], CONFIG["D_MODEL"])
        self.k_proj = SparseBayesianLinear(CONFIG["D_MODEL"], CONFIG["D_MODEL"])
        self.v_proj = SparseBayesianLinear(CONFIG["D_MODEL"], CONFIG["D_MODEL"])
        self.out_proj = SparseBayesianLinear(CONFIG["D_MODEL"], CONFIG["D_MODEL"])
        self.num_heads = CONFIG["NUM_HEADS"]
        self.head_dim = CONFIG["D_MODEL"] // self.num_heads

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q, (q_sigma, q_w) = self.q_proj(x)
        k, (k_sigma, k_w) = self.k_proj(x)
        v, (v_sigma, v_w) = self.v_proj(x)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, CONFIG["D_MODEL"])
        
        output, (o_sigma, o_w) = self.out_proj(attn_output)

        sigmas = (q_sigma, k_sigma, v_sigma, o_sigma)
        masked_weights = (q_w, k_w, v_w, o_w)
        return output, sigmas, masked_weights

class SparseBayesianTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(CONFIG["D_MODEL"])
        self.attn = SparseBayesianAttention()
        self.ln2 = nn.LayerNorm(CONFIG["D_MODEL"])
        self.ffn1 = SparseBayesianLinear(CONFIG["D_MODEL"], CONFIG["D_MODEL"])
        self.ffn2 = SparseBayesianLinear(CONFIG["D_MODEL"], CONFIG["D_MODEL"])
        self.ffn3 = SparseBayesianLinear(CONFIG["D_MODEL"], CONFIG["D_MODEL"])

    def forward(self, x):
        attn_out, attn_sigmas, attn_weights = self.attn(self.ln1(x))
        x = x + attn_out
        
        ffn_in = self.ln2(x)
        ffn1_out, (ffn1_sigma, ffn1_w) = self.ffn1(ffn_in)
        ffn1_out = F.silu(ffn1_out)
        ffn2_out, (ffn2_sigma, ffn2_w) = self.ffn2(ffn1_out)
        ffn2_out = F.silu(ffn2_out)
        ffn3_out, (ffn3_sigma, ffn3_w) = self.ffn3(ffn2_out)
        x = x + ffn3_out
        
        all_sigmas = attn_sigmas + (ffn1_sigma, ffn2_sigma, ffn3_sigma)
        all_weights = attn_weights + (ffn1_w, ffn2_w, ffn3_w)
        return x, all_sigmas, all_weights

class PocModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(CONFIG["VOCAB_SIZE"], CONFIG["D_MODEL"])
        self.pos_embedding = nn.Embedding(CONFIG["SEQ_LEN"], CONFIG["D_MODEL"])
        self.blocks = nn.ModuleList([SparseBayesianTransformerBlock() for _ in range(CONFIG["NUM_TRANSFORMER_BLOCKS"])])
        self.lm_head = nn.Linear(CONFIG["D_MODEL"], CONFIG["VOCAB_SIZE"])

    def forward(self, x):
        batch_size, seq_len = x.shape
        tok_emb = self.embedding(x)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos)
        x = tok_emb + pos_emb
        
        all_sigmas: list[tuple[torch.Tensor, ...]] = []
        all_weights: list[tuple[torch.Tensor, ...]] = []
        
        for block in self.blocks:
            x, sigmas, weights = block(x)
            all_sigmas.extend(sigmas)
            all_weights.extend(weights)

        return self.lm_head(x), all_sigmas, all_weights

def evaluate_ood(model, ood_dataset_x, ood_dataset_y):
    model.eval()
    correct_predictions = 0
    total_samples = ood_dataset_x.size(0)
    samples_to_print = []
    indices_to_print = set(random.sample(range(total_samples), k=min(5, total_samples)))

    def decode_tokens(tokens: torch.Tensor) -> str:
        tokens_list = tokens.cpu().tolist()
        return "".join([REVERSE_TOKENIZER.get(t, '?') for t in tokens_list if t not in [TOKENIZER['PAD'], TOKENIZER['BOS'], TOKENIZER['EOS']]])

    with torch.no_grad():
        for i in track(range(total_samples), description="Evaluating OOD Generalization..."):
            x = ood_dataset_x[i].unsqueeze(0)
            y_true = ood_dataset_y[i]

            equals_pos = (x[0] == TOKENIZER['=']).nonzero(as_tuple=True)[0].item()
            problem_tokens = x[0, :equals_pos + 1]

            generated_seq = problem_tokens.unsqueeze(0)
            for _ in range(CONFIG["SEQ_LEN"] - (equals_pos + 1)):
                logits, _, _ = model(generated_seq)
                next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
                
                if next_token.item() in [TOKENIZER['PAD'], TOKENIZER['EOS']]:
                    break
                
                generated_seq = torch.cat([generated_seq, next_token], dim=1)

            generated_answer_tokens = generated_seq[0, equals_pos + 1:]
            
            true_answer_start_pos = equals_pos + 1

            true_answer_end_idx = (y_true == TOKENIZER['PAD']).nonzero(as_tuple=True)[0]
            if true_answer_end_idx.numel() == 0:
                true_answer_end_idx = (y_true == TOKENIZER['EOS']).nonzero(as_tuple=True)[0]
            
            if true_answer_end_idx.numel() > 0:
                true_answer_tokens = y_true[true_answer_start_pos : true_answer_end_idx[0].item()]
            else:
                true_answer_tokens = y_true[true_answer_start_pos:] 

            gen_answer_end = (generated_answer_tokens == TOKENIZER['EOS']).nonzero(as_tuple=True)[0]
            if len(gen_answer_end) > 0:
                generated_answer_tokens = generated_answer_tokens[:gen_answer_end[0].item()]
            
            # 在比较之前移除PAD，因为generate_arithmetic_data中用PAD填充了
            true_answer_tokens = true_answer_tokens[true_answer_tokens != TOKENIZER['PAD']]
            generated_answer_tokens = generated_answer_tokens[generated_answer_tokens != TOKENIZER['PAD']]

            if generated_answer_tokens.size() == true_answer_tokens.size() and torch.all(generated_answer_tokens.cpu() == true_answer_tokens.cpu()):
                correct_predictions += 1
            
            if i in indices_to_print:
                samples_to_print.append({
                    "problem": decode_tokens(problem_tokens),
                    "generated": decode_tokens(generated_answer_tokens),
                    "true": decode_tokens(true_answer_tokens),
                })
    
    console.print("\n[bold cyan]--- OOD Generation Samples ---[/bold cyan]")
    for sample in samples_to_print:
        console.print(f"Problem:   {sample['problem']}")
        console.print(f"Generated: {sample['generated']}")
        console.print(f"True:      {sample['true']}")
        console.print("-" * 30)

    return correct_predictions / total_samples

def run_experiment():
    model = PocModel().to(device=CONFIG["DEVICE"], dtype=CONFIG["DTYPE"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LR"])
    
    start_step = 0
    if os.path.exists(CKPT_PATH):
        console.print(f"Loading checkpoint from {CKPT_PATH}")
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step'] + 1
        console.print(f"Resuming from step {start_step}")

    console.print("Generating in-distribution training data (0-900)...")
    dataset_x, dataset_y = generate_arithmetic_data(CONFIG["DATASET_SIZE"], CONFIG["SEQ_LEN"], operand_range=(0, 900))
    dataset_x, dataset_y = dataset_x.to(CONFIG["DEVICE"]), dataset_y.to(CONFIG["DEVICE"])
    
    console.print("Generating out-of-distribution test data (900-1000)...")
    ood_dataset_x, ood_dataset_y = generate_arithmetic_data(CONFIG["DATASET_SIZE"] // 50, CONFIG["SEQ_LEN"], operand_range=(900, 1000))
    ood_dataset_x, ood_dataset_y = ood_dataset_x.to(CONFIG["DEVICE"]), ood_dataset_y.to(CONFIG["DEVICE"])

    console.print("Starting training...")
    model.train()
    last_log_time = time.monotonic()
    for step in range(start_step, CONFIG["TRAINING_STEPS"]):
        try:
            batch_indices = torch.randint(0, dataset_x.size(0), (CONFIG["BATCH_SIZE"],))
            x, labels = dataset_x[batch_indices], dataset_y[batch_indices]
            
            optimizer.zero_grad()
            
            logits, sigmas, masked_weights = model(x)
            
            equals_indices = (x == TOKENIZER['=']).nonzero(as_tuple=True)
            loss_mask = torch.zeros_like(labels, dtype=torch.bool)
            for i in range(equals_indices[0].size(0)):
                row_idx = equals_indices[0][i]
                col_idx = equals_indices[1][i]
                loss_mask[row_idx, col_idx+1:] = True
            
            masked_labels = labels.masked_fill(~loss_mask, IGNORE_INDEX)

            main_loss = F.cross_entropy(logits.view(-1, CONFIG["VOCAB_SIZE"]), masked_labels.view(-1), ignore_index=IGNORE_INDEX)
            
            # The new forward pass returns sigmas and masked_weights per token.
            # We need to flatten them before calculating gradients.
            flat_masked_weights = [w.view(-1) for w in masked_weights]
            
            # Since masked_weights is now a list of tensors, we need to handle it differently
            # We will compute surprise for each tensor in the list.
            surprise_grads = torch.autograd.grad(main_loss, flat_masked_weights, retain_graph=True)
            
            f_min_loss = 0
            # sigmas are now scores, also per token. We need to flatten them.
            flat_sigmas = [s.view(-1) for s in sigmas]

            for sigma_group, surprise_grad_group in zip(flat_sigmas, surprise_grads):
                surprise_per_param = surprise_grad_group.abs()
                f_min_loss += F.mse_loss(sigma_group, surprise_per_param.view_as(sigma_group))
                
            weighted_f_min_loss = f_min_loss / (main_loss.detach()**2 + 1e-9)
            total_loss = main_loss + weighted_f_min_loss
            total_loss.backward()
            optimizer.step()

            if step % 30 == 0:
                with torch.no_grad():
                    current_time = time.monotonic()
                    duration = current_time - last_log_time
                    last_log_time = current_time
                    it_s = 30 / duration if step > 0 else 0.0

                    preds = logits.argmax(dim=-1)
                    correct_preds = (preds == labels) & (masked_labels != IGNORE_INDEX)
                    accuracy = correct_preds.sum() / ((masked_labels != IGNORE_INDEX).sum() + 1e-9)

                    avg_sigma = torch.mean(torch.stack([s.mean() for s in sigmas])).item()
                    all_rhos = [p for name, p in model.named_parameters() if 'rho' in name]
                    avg_rho = torch.mean(torch.stack([r.mean() for r in all_rhos])).item()
                    all_gates = [p for name, p in model.named_parameters() if 'gate' in name]
                    avg_gate = torch.mean(torch.stack([g.mean() for g in all_gates])).item()
                    
                    all_sigmas_from_rho = [F.softplus(rho) for rho in all_rhos]
                    activation_rates = [(sig > gat).float().mean() for sig, gat in zip(all_sigmas_from_rho, all_gates)]
                    total_activation_rate = torch.mean(torch.stack(activation_rates)).item() * 100

                    weighted_f_min_loss_val = f_min_loss.item() / (main_loss.item()**2 + 1e-9)
                    
                    tau_val = torch.distributions.Categorical(logits=logits.detach()).entropy().mean().item()
                    gamma_val = torch.mean(torch.stack([(s - g).mean() for s, g in zip(all_sigmas_from_rho, all_gates)])).item()
                    alpha_val = total_activation_rate / 100.0
                    
                    pi_score = math.exp(-alpha_val * ((1 - gamma_val) * (main_loss.item() / (tau_val + 1e-9)) + gamma_val * f_min_loss.item()))

                    console.print(f"Step {step:5d} | Loss(m/f_w): {main_loss.item():.3f}/{weighted_f_min_loss_val:.3f} | Acc: {accuracy.item():.3f} | Avg σ/ρ/g: {avg_sigma:.4f}/{avg_rho:.4f}/{avg_gate:.4f} | Act%: {total_activation_rate:.2f} | PI/α/γ/S: {pi_score:.3f}/{alpha_val:.2f}/{gamma_val:.2f}/{f_min_loss.item():.3f} | it/s: {it_s:.2f}")
                    
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, CKPT_PATH)

        except KeyboardInterrupt:
            console.print("\nTraining interrupted by user.")
            break

    console.print("\nTraining complete.")
    
    ood_accuracy = evaluate_ood(model, ood_dataset_x, ood_dataset_y)
    console.print(f"\n[bold green]OOD Generalization Accuracy (900-1000): {ood_accuracy:.4f}[/bold green]")

if __name__ == "__main__":
    run_experiment()