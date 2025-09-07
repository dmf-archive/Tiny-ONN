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
    "BATCH_SIZE": 1, "SEQ_LEN": 256, "D_MODEL": 128, "VOCAB_SIZE": 22,
    "NUM_TRANSFORMER_BLOCKS": 3,
    "LR": 3e-3, "DEVICE": "cuda", "DTYPE": torch.bfloat16,
    "TRAINING_STEPS": 100000,
    "KL_PRIOR_EPSILON": 1e-9,
}
CKPT_PATH = "latest.pt"
console = Console()
console.print(f"Using device: {CONFIG['DEVICE']}")

TOKENIZER = {
    'BOS': 0, 'EOS': 1, 'PAD': 2,
    '0': 3, '1': 4, '2': 5, '3': 6, '4': 7, '5': 8, '6': 9, '7': 10, '8': 11, '9': 12,
    '+': 13, '-': 14, '×': 15, '=': 16,
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

def generate_arithmetic_data(num_samples, seq_len, operand_ranges):
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

    def _get_multiplication_cot(a, b):
        s_b = str(b)
        cot_tokens = []
        partial_products = []
        for i, digit in enumerate(reversed(s_b)):
            multiplier = int(digit)
            partial_product = a * multiplier
            partial_products.append(partial_product * (10**i))
            
            s_a = str(a)
            carry = 0
            for j in range(len(s_a) - 1, -1, -1):
                d1 = int(s_a[j])
                prod_val = d1 * multiplier + carry
                result_digit = prod_val % 10
                new_carry = prod_val // 10
                cot_tokens.extend(str_to_tokens(s_a[j]) + [TOKENIZER['×']] + str_to_tokens(digit) + [TOKENIZER['+']] + str_to_tokens(carry) + [TOKENIZER['=']] + str_to_tokens(result_digit))
                cot_tokens.extend([TOKENIZER['CARRY'], *str_to_tokens(new_carry), TOKENIZER['\n']])
                carry = new_carry

        if len(partial_products) > 1:
            current_sum = 0
            for p in partial_products:
                cot_tokens.extend(_get_addition_cot(current_sum, p))
                current_sum += p
        return cot_tokens

    while len(sents) < num_samples:
        a_range = random.choice(operand_ranges)
        a = random.randint(*a_range)
        if len(operand_ranges) > 1 and a >= 1000:
            b_range = random.choice(operand_ranges[:-1])
        else:
            b_range = random.choice(operand_ranges)
        b = random.randint(*b_range)
        op_str = random.choice(['+', '-', '×'])
        
        a_orig, b_orig = a, b
        if op_str == '+':
            cot_tokens = _get_addition_cot(a, b)
            res = a + b
        elif op_str == '-':
            if a < b: a, b = b, a
            cot_tokens = _get_subtraction_cot(a, b)
            res = a_orig - b_orig
        else:
            cot_tokens = _get_multiplication_cot(a, b)
            res = a * b

        problem_tokens = str_to_tokens(a_orig) + [TOKENIZER[op_str]] + str_to_tokens(b_orig)
        answer_tokens = str_to_tokens(res)

        full_seq = ([TOKENIZER['BOS']] + problem_tokens + [TOKENIZER['=']] +
                    [TOKENIZER['<think>']] + cot_tokens + [TOKENIZER['</think>']] +
                    answer_tokens + [TOKENIZER['EOS']])
        
        if len(full_seq) >= CONFIG["SEQ_LEN"]: continue

        x, y = full_seq[:-1], full_seq[1:]
        x.extend([TOKENIZER['PAD']] * (CONFIG["SEQ_LEN"] - len(x)))
        y.extend([TOKENIZER['PAD']] * (CONFIG["SEQ_LEN"] - len(y)))
        
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

def evaluate_ood(model, ood_dataset_x, ood_dataset_y):
    model.eval()
    correct_count, total_count = 0, ood_dataset_x.size(0)
    sample_outputs = []
    
    console.print("\n[bold yellow]--- OOD Evaluation ---[/bold yellow]")
    
    with torch.no_grad():
        for i in track(range(total_count), description="Evaluating OOD..."):
            x_sample_list = ood_dataset_x[i].tolist()
            y_sample_list = ood_dataset_y[i].tolist()
            
            try:
                equals_pos = x_sample_list.index(TOKENIZER['='])
                prompt_tokens = x_sample_list[:equals_pos + 1]
                think_end_pos = y_sample_list.index(TOKENIZER['</think>'])
                eos_pos = y_sample_list.index(TOKENIZER['EOS'])
                true_answer_tokens = y_sample_list[think_end_pos + 1 : eos_pos]
            except ValueError: continue

            prompt = torch.tensor(prompt_tokens, device=CONFIG["DEVICE"]).unsqueeze(0)
            
            generated_tokens = []
            for _ in range(CONFIG["SEQ_LEN"] - prompt.size(1)):
                logits, _, _ = model(prompt)
                next_token = logits[0, -1, :].argmax()
                if next_token.item() == TOKENIZER['EOS']: break
                generated_tokens.append(next_token.item())
                prompt = torch.cat([prompt, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

            def decode_tokens(tokens):
                if not tokens: return ""
                s = ""
                tokens_copy = list(tokens)
                if TOKENIZER['-'] in tokens_copy:
                    s += "-"
                    tokens_copy.remove(TOKENIZER['-'])
                s += "".join([REVERSE_TOKENIZER.get(t, '?') for t in tokens_copy if t != TOKENIZER['PAD']])
                return s
            
            true_answer_str = decode_tokens(true_answer_tokens)
            gen_full_str = "".join([REVERSE_TOKENIZER.get(t, '?') for t in generated_tokens])
            think_end_tag = "</think>"
            end_idx = gen_full_str.rfind(think_end_tag)
            gen_answer_str = gen_full_str[end_idx + len(think_end_tag):] if end_idx != -1 else gen_full_str

            if true_answer_str == gen_answer_str:
                correct_count += 1

            if i < 5:
                prompt_str = "".join([REVERSE_TOKENIZER.get(t, '?') for t in prompt_tokens if t != TOKENIZER['BOS']])
                think_start_tag, think_end_tag = "<think>", "</think>"
                start_idx, end_idx = gen_full_str.find(think_start_tag), gen_full_str.rfind(think_end_tag)

                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    think_content = gen_full_str[start_idx + len(think_start_tag):end_idx]
                    think_content_formatted = "\n    ".join(think_content.split("\\n"))
                    result_content = gen_full_str[end_idx + len(think_end_tag):]
                else:
                    think_content_formatted, result_content = "[Could not parse think block]", gen_full_str

                status = '✅' if true_answer_str == result_content else '❌'
                output = (f"--- Sample {i+1} ---\n[bold cyan]Prompt  :[/bold cyan] {prompt_str}\n"
                          f"[bold yellow]Think   :[/bold yellow]\n    {think_content_formatted}\n"
                          f"[bold magenta]Result  :[/bold magenta] {result_content}\n"
                          f"[bold green]Expected:[/bold green] {true_answer_str}\n"
                          f"[bold]Status  :[/bold] {status}")
                sample_outputs.append(output)

    if sample_outputs:
        console.print("\n" + "\n\n".join(sample_outputs))
    accuracy = correct_count / total_count if total_count > 0 else 0
    console.print(f"[bold green]OOD Exact Match Accuracy: {accuracy:.4f}[/bold green]")
    console.print("[bold yellow]----------------------[/bold yellow]\n")
    model.train()

def run_experiment():
    model = PocModel().to(device=CONFIG["DEVICE"], dtype=CONFIG["DTYPE"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["LR"], betas=(0.9, 0.98))
    
    start_step = 0
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step'] + 1
    
    run_experiment.last_log_time = time.time()
    training_ranges = [(0, 9), (10, 99), (100, 999), (1000, 9999)]
    
    model.train()
    for step in range(start_step, CONFIG["TRAINING_STEPS"]):
        x, labels = generate_arithmetic_data(CONFIG["BATCH_SIZE"], CONFIG["SEQ_LEN"], operand_ranges=training_ranges)
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

        avg_tau = torch.distributions.Categorical(logits=logits.detach()).entropy()[loss_mask].mean() if loss_mask.any() else torch.tensor(0.0)
        prior_std = torch.clamp(avg_tau, min=CONFIG["KL_PRIOR_EPSILON"])
        
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
        
        if step % 50 == 0:
            if not hasattr(run_experiment, 'last_log_time'): run_experiment.last_log_time = time.time()
            end_time = time.time()
            steps_per_sec = 50 / (end_time - run_experiment.last_log_time) if (end_time - run_experiment.last_log_time) > 0 else float('inf')
            run_experiment.last_log_time = end_time

            with torch.no_grad():
                accuracy = (logits.argmax(-1) == labels)[loss_mask].float().mean() if loss_mask.any() else torch.tensor(0.0)
                all_sigmas = [p for name, p in model.named_parameters() if 'sigma_weight' in name]
                all_gates = [p for name, p in model.named_parameters() if 'gate_param' in name]
                avg_sigma = torch.mean(torch.stack([F.softplus(s).mean() for s in all_sigmas])).item() if all_sigmas else 0
                avg_gate = torch.mean(torch.stack([g.mean() for g in all_gates])).item() if all_gates else 0
                total_active = sum((mo.abs() > 1e-5).float().sum() for mo in masked_outputs)
                total_elements = sum(mo.numel() for mo in masked_outputs)
                activation_rate = (total_active / total_elements * 100).item() if total_elements > 0 else 0
                gate_loss_val = gate_loss.item() if isinstance(gate_loss, torch.Tensor) else gate_loss
                kl_loss_val = kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
                
                console.print(f"Step {step:5d} | Loss(m/g/k): {main_loss.item():.3f}/{gate_loss_val:.3f}/{kl_loss_val:.3f} | Acc: {accuracy.item():.3f} | Avg σ/g: {avg_sigma:.4f}/{avg_gate:.4f} | Act%: {activation_rate:.2f} | τ/p_std: {avg_tau.item():.3f}/{prior_std.item():.3f} | it/s: {steps_per_sec:.2f}")
            
            if step > 0 and step % 500 == 0:
                ood_ranges = [(0, 20000)]
                ood_x, ood_y = generate_arithmetic_data(96, CONFIG["SEQ_LEN"], operand_ranges=ood_ranges)
                evaluate_ood(model, ood_x.to(CONFIG["DEVICE"]), ood_y.to(CONFIG["DEVICE"]))

            torch.save({ 'step': step, 'model_state_dict': model.state_dict() }, CKPT_PATH)

if __name__ == "__main__":
    run_experiment()