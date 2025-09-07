import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from rich.console import Console
from rich.progress import track
import math
import random
import os
from collections import deque

CONFIG = {
    "BATCH_SIZE": 1, "SEQ_LEN": 64, "D_MODEL": 128, "VOCAB_SIZE": 22,
    "NUM_TRANSFORMER_BLOCKS": 3,
    "LR": 3e-3, "DEVICE": "cuda", "DTYPE": torch.bfloat16,
    "MAX_STEPS_PER_COURSE": 20000,
    "PI_GRADUATION_WINDOW": 100,
    "PI_GRADUATION_THRESHOLD": 0.9,
}
CKPT_DIR = "exp"
CKPT_PATH = os.path.join(CKPT_DIR, "thefeus.ckpt")
console = Console()
console.print(f"Using device: {CONFIG['DEVICE']}")

TOKENIZER = {
    'BOS': 0, 'EOS': 1, 'PAD': 2,
    '0': 3, '1': 4, '2': 5, '3': 6, '4': 7, '5': 8, '6': 9, '7': 10, '8': 11, '9': 12,
    '+': 13, '-': 14, '=': 16,
    '<think>': 17, '</think>': 18, '\n': 19,
    'CARRY': 20, 'BORROW': 21
}
REVERSE_TOKENIZER = {v: k for k, v in TOKENIZER.items()}
IGNORE_INDEX = -100

def str_to_tokens(s):
    tokens = []
    s = str(s)
    if s.startswith('-'):
        tokens.append(TOKENIZER['-'])
        s = s[1:]
    tokens.extend([TOKENIZER.get(c, -1) for c in s])
    return [t for t in tokens if t != -1]

def generate_arithmetic_data(num_samples, seq_len, operand_ranges, task_type):
    sents, labels = [], []

    def _get_addition_cot(a, b):
        s_a, s_b = str(a), str(b)
        max_len = max(len(s_a), len(s_b))
        s_a, s_b = s_a.zfill(max_len), s_b.zfill(max_len)
        cot_tokens = []
        carry = 0
        for i in range(max_len - 1, -1, -1):
            d1, d2 = int(s_a[i]), int(s_b[i])
            sum_val = d1 + d2 + carry
            result_digit = sum_val % 10
            new_carry = sum_val // 10
            cot_tokens.extend(str_to_tokens(s_a[i]) + [TOKENIZER['+']] + str_to_tokens(s_b[i]) + [TOKENIZER['=']] + str_to_tokens(result_digit))
            cot_tokens.extend([TOKENIZER['CARRY'], *str_to_tokens(new_carry), TOKENIZER['\n']])
            carry = new_carry
        return cot_tokens

    def _get_subtraction_cot(a, b):
        s_a, s_b = str(a), str(b)
        max_len = max(len(s_a), len(s_b))
        s_a, s_b = s_a.zfill(max_len), s_b.zfill(max_len)
        cot_tokens = []
        borrow = 0
        for i in range(max_len - 1, -1, -1):
            d1 = int(s_a[i]) - borrow
            d2 = int(s_b[i])
            if d1 < d2:
                d1 += 10
                new_borrow = 1
            else:
                new_borrow = 0
            result_digit = d1 - d2
            cot_tokens.extend(str_to_tokens(s_a[i]) + [TOKENIZER['-']] + str_to_tokens(s_b[i]) + [TOKENIZER['=']] + str_to_tokens(result_digit))
            cot_tokens.extend([TOKENIZER['BORROW'], *str_to_tokens(new_borrow), TOKENIZER['\n']])
            borrow = new_borrow
        return cot_tokens

    while len(sents) < num_samples:
        a = random.randint(*random.choice(operand_ranges))
        b = random.randint(*random.choice(operand_ranges))
        
        op_str = ''
        if task_type == 'addition':
            op_str = '+'
            cot_tokens = _get_addition_cot(a, b)
            res = a + b
        elif task_type == 'subtraction':
            op_str = '-'
            if a < b: a, b = b, a
            cot_tokens = _get_subtraction_cot(a, b)
            res = a - b
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        problem_tokens = str_to_tokens(a) + [TOKENIZER[op_str]] + str_to_tokens(b)
        answer_tokens = str_to_tokens(res)

        full_seq = ([TOKENIZER['BOS']] + problem_tokens + [TOKENIZER['=']] +
                    [TOKENIZER['<think>']] + cot_tokens + [TOKENIZER['</think>']] +
                    answer_tokens + [TOKENIZER['EOS']])
        
        if len(full_seq) >= seq_len: continue

        x, y = full_seq[:-1], full_seq[1:]
        x.extend([TOKENIZER['PAD']] * (seq_len - len(x)))
        y.extend([TOKENIZER['PAD']] * (seq_len - len(y)))
        
        sents.append(torch.tensor(x))
        labels.append(torch.tensor(y))

    return torch.stack(sents).long(), torch.stack(labels).long()

class SBL(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features, dtype=CONFIG["DTYPE"]))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features, dtype=CONFIG["DTYPE"]))
        self.gate_param = nn.Parameter(torch.empty(out_features, dtype=CONFIG["DTYPE"]))
        self.mu_bias = nn.Parameter(torch.empty(out_features, dtype=CONFIG["DTYPE"]))
        nn.init.kaiming_uniform_(self.mu_weight, a=math.sqrt(5))
        nn.init.normal_(self.sigma_weight, mean=0.0, std=0.5)
        nn.init.constant_(self.gate_param, -0.1)
        nn.init.zeros_(self.mu_bias)

    def forward(self, x):
        original_shape = x.shape
        x_reshaped = x.view(-1, self.in_features)
        
        keys = self.mu_weight * F.softplus(self.sigma_weight)
        scores = torch.matmul(x_reshaped, keys.t()) / math.sqrt(self.in_features)
        raw_weights = F.relu(scores - self.gate_param.unsqueeze(0))

        computation_output = F.linear(x_reshaped, self.mu_weight, self.mu_bias)
        masked_output = computation_output * raw_weights
        output = masked_output.view(*original_shape[:-1], self.out_features)
        return output, scores, masked_output

class DynSIHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.sbl_qkv = SBL(CONFIG["D_MODEL"], 3 * CONFIG["D_MODEL"])
        self.sbl_o = SBL(CONFIG["D_MODEL"], CONFIG["D_MODEL"])

    def forward(self, x):
        qkv, s_qkv, m_qkv = self.sbl_qkv(x)
        q, k, v = torch.split(qkv, CONFIG["D_MODEL"], dim=-1)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y, s_o, m_o = self.sbl_o(attn_out)
        return y, (s_qkv, s_o), (m_qkv, m_o)

class MoIEFFN(nn.Module):
    def __init__(self):
        super().__init__()
        d_ffn = CONFIG["D_MODEL"] * 4
        self.sbl1 = SBL(CONFIG["D_MODEL"], d_ffn)
        self.sbl2 = SBL(d_ffn, CONFIG["D_MODEL"])

    def forward(self, x):
        h, s1, m1 = self.sbl1(x)
        h_act = F.silu(h)
        y, s2, m2 = self.sbl2(h_act)
        return y, (s1, s2), (m1, m2)

class MoIETransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(CONFIG["D_MODEL"], dtype=CONFIG["DTYPE"])
        self.attn = DynSIHA()
        self.ln2 = nn.LayerNorm(CONFIG["D_MODEL"], dtype=CONFIG["DTYPE"])
        self.ffn = MoIEFFN()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        attn_in = self.ln1(x)
        attn_out, attn_scores, attn_masked = self.attn(attn_in)
        x = x + self.dropout(attn_out)
        ffn_in = self.ln2(x)
        ffn_out, ffn_scores, ffn_masked = self.ffn(ffn_in)
        x = x + self.dropout(ffn_out)
        return x, attn_scores + ffn_scores, attn_masked + ffn_masked

class PocModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(CONFIG["VOCAB_SIZE"], CONFIG["D_MODEL"], dtype=CONFIG["DTYPE"])
        self.pos_embedding = nn.Embedding(CONFIG["SEQ_LEN"], CONFIG["D_MODEL"], dtype=CONFIG["DTYPE"])
        self.blocks = nn.ModuleList([MoIETransformerBlock() for _ in range(CONFIG["NUM_TRANSFORMER_BLOCKS"])])
        self.lm_head = nn.Linear(CONFIG["D_MODEL"], CONFIG["VOCAB_SIZE"], dtype=CONFIG["DTYPE"])

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

class PIScoreCalculator:
    def __init__(self):
        self.pi_history = deque(maxlen=CONFIG["PI_GRADUATION_WINDOW"])

    @torch.no_grad()
    def update_and_check(self, model, main_loss, surprise_grads, logits, loss_mask):
        all_sigmas = [p.detach() for name, p in model.named_parameters() if 'sigma_weight' in name]
        all_gates = [p.detach() for name, p in model.named_parameters() if 'gate_param' in name]
        
        avg_sigma = torch.mean(torch.stack([F.softplus(s).mean() for s in all_sigmas])).item() if all_sigmas else 0
        avg_gate = torch.mean(torch.stack([g.mean() for g in all_gates])).item() if all_gates else 0
        
        plasticity = avg_sigma - avg_gate
        gamma = torch.sigmoid(torch.tensor(plasticity)).item()
        
        total_active = sum((mo.abs() > 1e-5).float().sum() for mo in surprise_grads if mo is not None)
        total_elements = sum(mo.numel() for mo in surprise_grads if mo is not None)
        activation_rate = (total_active / total_elements).item() if total_elements > 0 else 0
        alpha = activation_rate * 100
        
        surprise = torch.tensor(0.0)
        if surprise_grads:
            flat_grads = torch.cat([g.view(-1) for g in surprise_grads if g is not None])
            if flat_grads.numel() > 0:
                surprise = flat_grads.norm(p=2)
        
        avg_tau = torch.distributions.Categorical(logits=logits.detach()).entropy()[loss_mask].mean() if loss_mask.any() else torch.tensor(1e-9)

        normalized_error = main_loss.item() / (avg_tau.item() + 1e-9)
        complexity_cost = surprise.item() * activation_rate
        
        cost = (1 - gamma) * normalized_error + gamma * complexity_cost
        pi_score = math.exp(-alpha * cost)
        
        self.pi_history.append(pi_score)
        
        graduated = (len(self.pi_history) == CONFIG["PI_GRADUATION_WINDOW"] and
                     all(p > CONFIG["PI_GRADUATION_THRESHOLD"] for p in self.pi_history))
        
        metrics = {
            "pi_score": pi_score, "gamma": gamma, "alpha": alpha,
            "activation_rate": activation_rate, "avg_tau": avg_tau.item(),
            "surprise": surprise.item(),
            "avg_sigma": avg_sigma, "avg_gate": avg_gate
        }
        return graduated, metrics

def evaluate_task(model, task_type):
    model.eval()
    correct_count, total_count = 0, 100
    operand_ranges = [(0, 999)]
    test_x, test_y = generate_arithmetic_data(total_count, CONFIG["SEQ_LEN"], operand_ranges, task_type)
    test_x, test_y = test_x.to(CONFIG["DEVICE"]), test_y.to(CONFIG["DEVICE"])

    with torch.no_grad():
        for i in range(total_count):
            x_sample = test_x[i].unsqueeze(0)
            logits, _, _ = model(x_sample)
            preds = logits.argmax(-1)
            
            mask = (test_y[i] != TOKENIZER['PAD'])
            correct_tokens = ((preds.squeeze(0) == test_y[i])[mask]).all()
            if correct_tokens:
                correct_count += 1
    
    accuracy = correct_count / total_count
    console.print(f"[bold green]Accuracy on '{task_type}': {accuracy:.4f}[/bold green]")
    model.train()
    return accuracy

def run_experiment():
    model = PocModel().to(device=CONFIG["DEVICE"], dtype=CONFIG["DTYPE"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["LR"], betas=(0.9, 0.98))
    pi_calculator = PIScoreCalculator()

    courses = ['addition', 'subtraction']
    operand_ranges = [(0, 999)]
    
    start_step = 0
    start_course_idx = 0
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['global_step'] + 1
        start_course_idx = checkpoint['course_idx']
        console.print(f"Resuming from step {start_step} in course '{courses[start_course_idx]}'")

    global_step = start_step
    last_log_time = time.time()
    for course_idx in range(start_course_idx, len(courses)):
        course = courses[course_idx]
        console.print(f"\n--- [bold yellow]Starting Course: {course.upper()} ---[/bold yellow]")
        pi_calculator.pi_history.clear()
        
        for step in range(CONFIG["MAX_STEPS_PER_COURSE"]):
            x, labels = generate_arithmetic_data(CONFIG["BATCH_SIZE"], CONFIG["SEQ_LEN"], operand_ranges, course)
            x, labels = x.to(CONFIG["DEVICE"]), labels.to(CONFIG["DEVICE"])
            
            optimizer.zero_grad()
            logits, scores, masked_outputs = model(x)
            
            loss_mask = (labels != TOKENIZER['PAD'])
            masked_labels = labels.masked_fill(~loss_mask, IGNORE_INDEX)
            main_loss = F.cross_entropy(logits.view(-1, CONFIG["VOCAB_SIZE"]), masked_labels.view(-1), ignore_index=IGNORE_INDEX)
            
            if not masked_outputs: continue

            surprise_grads = torch.autograd.grad(main_loss, masked_outputs, retain_graph=True, allow_unused=True)
            gate_loss = torch.tensor(0.0, device=x.device)

            total_active = sum((mo.abs() > 1e-5).float().sum() for mo in masked_outputs if mo is not None)
            total_elements = sum(mo.numel() for mo in masked_outputs if mo is not None)
            activation_rate = (total_active / total_elements).item() if total_elements > 0 else 0
            
            for grad_tensor in surprise_grads:
                if grad_tensor is not None:
                    surprise_per_neuron = grad_tensor.view(-1, grad_tensor.shape[-1]).norm(p=2, dim=0)
                    active_surprise = surprise_per_neuron[surprise_per_neuron > 1e-9]
                    if active_surprise.numel() > 0:
                        weighted_surprise = active_surprise * activation_rate
                        gate_loss += (-torch.log(weighted_surprise + 1e-9) * weighted_surprise).sum()

            avg_tau_val = torch.distributions.Categorical(logits=logits.detach()).entropy()[loss_mask].mean() if loss_mask.any() else torch.tensor(0.0)
            prior_std = torch.clamp(avg_tau_val, min=1e-9)

            kl_loss = torch.tensor(0.0, device=x.device)
            num_sbl_layers = 0
            for module in model.modules():
                if isinstance(module, SBL):
                    q_w = torch.distributions.Normal(module.mu_weight, F.softplus(module.sigma_weight))
                    p_w = torch.distributions.Normal(torch.zeros_like(module.mu_weight), prior_std)
                    kl_loss += torch.distributions.kl_divergence(q_w, p_w).mean()
                    num_sbl_layers += 1
            if num_sbl_layers > 0: kl_loss /= num_sbl_layers

            w_gate = 1.0 - torch.sigmoid(main_loss.detach())
            total_loss = main_loss + w_gate * gate_loss + kl_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if global_step % 50 == 0:
                end_time = time.time()
                steps_per_sec = 50 / (end_time - last_log_time) if (end_time - last_log_time) > 0 else float('inf')
                last_log_time = end_time

                graduated, metrics = pi_calculator.update_and_check(model, main_loss, surprise_grads, logits, loss_mask)
                accuracy = (logits.argmax(-1) == labels)[loss_mask].float().mean().item()
                
                gate_loss_val = gate_loss.item() if isinstance(gate_loss, torch.Tensor) else gate_loss
                kl_loss_val = kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss

                log_str = (
                    f"Step {global_step:6d} [{course}] | Loss(m/g/k): {main_loss.item():.3f}/{gate_loss_val:.3f}/{kl_loss_val:.3f} | Acc: {accuracy:.3f} | "
                    f"PI: {metrics['pi_score']:.3f} | τ: {metrics['avg_tau']:.3f} | Surprise: {metrics['surprise']:.3f} | "
                    f"Act%: {metrics['activation_rate']*100:.2f} | Avg σ/g: {metrics['avg_sigma']:.4f}/{metrics['avg_gate']:.4f} | it/s: {steps_per_sec:.2f}"
                )
                console.print(log_str)

                torch.save({
                    'global_step': global_step,
                    'course_idx': course_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, CKPT_PATH)
                
                if graduated:
                    console.print(f"--- [bold green]Graduated from Course: {course.upper()} ---[/bold green]")
                    break
            
            global_step += 1
        
        ckpt_path = os.path.join(CKPT_DIR, f"ckpt_{course}.pt")
        torch.save(model.state_dict(), ckpt_path)
        console.print(f"Saved checkpoint to {ckpt_path}")

        if step == CONFIG["MAX_STEPS_PER_COURSE"] - 1:
            console.print(f"[yellow]Warning: Max steps reached for course '{course}' without graduation.[/yellow]")

    console.print("\n--- [bold yellow]Final Evaluation ---[/bold yellow]")
    model_final = PocModel().to(device=CONFIG["DEVICE"])
    model_final.load_state_dict(torch.load(os.path.join(CKPT_DIR, 'ckpt_subtraction.pt')))
    
    acc_A_final = evaluate_task(model_final, 'addition')
    
    model_initial_A = PocModel().to(device=CONFIG["DEVICE"])
    model_initial_A.load_state_dict(torch.load(os.path.join(CKPT_DIR, 'ckpt_addition.pt')))
    acc_A_initial = evaluate_task(model_initial_A, 'addition')

    forgetting_score = max(0, 1 - (acc_A_final / acc_A_initial)) if acc_A_initial > 0 else float('inf')
    console.print(f"\n[bold magenta]Forgetting Score: {forgetting_score:.4f}[/bold magenta]")

if __name__ == "__main__":
    run_experiment()