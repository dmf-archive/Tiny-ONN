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
    "BATCH_SIZE": 16, "SEQ_LEN": 256, "D_MODEL": 64, "VOCAB_SIZE": 25,
    "NUM_HEADS": 4, "NUM_TRANSFORMER_BLOCKS": 2, 
    "LR": 3e-3, "DEVICE": "cuda", "DTYPE": torch.bfloat16,
    "TRAINING_STEPS": 50000, "DATASET_SIZE": 16384
}
CKPT_PATH = "latest.pt"
console = Console()
console.print(f"Using device: {CONFIG['DEVICE']}")

TOKENIZER = {
    'BOS': 0, 'EOS': 1, 'PAD': 2,
    '0': 3, '1': 4, '2': 5, '3': 6, '4': 7, '5': 8, '6': 9, '7': 10, '8': 11, '9': 12,
    '+': 13, '-': 14, '×': 15, '=': 16,
    '<think>': 17, '</think>': 18, '\n': 19, 'RESULT_': 20,
    'CARRY_0': 21, 'CARRY_1': 22,
    'BORROW_0': 23, 'BORROW_1': 24
}
REVERSE_TOKENIZER = {v: k for k, v in TOKENIZER.items()}
IGNORE_INDEX = -100

def str_to_tokens(s):
    tokens = []
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
        cot_tokens.extend(str_to_tokens(s_a) + [TOKENIZER['+']] + str_to_tokens(s_b) + [TOKENIZER['\n']])
        
        carry = 0
        res_digits = []
        for i in range(max_len - 1, -1, -1):
            d1, d2 = int(s_a[i]), int(s_b[i])
            sum_val = d1 + d2 + carry
            res_digits.insert(0, str(sum_val % 10))
            new_carry = sum_val // 10
            cot_tokens.extend(str_to_tokens(s_a[i]) + [TOKENIZER['+']] + str_to_tokens(s_b[i]) + [TOKENIZER['=']] + str_to_tokens(str(sum_val % 10)))
            cot_tokens.extend([TOKENIZER[f'CARRY_{new_carry}'], TOKENIZER['\n']])
            carry = new_carry
        if carry > 0:
            res_digits.insert(0, str(carry))

        res_str = "".join(res_digits)
        cot_tokens.extend([TOKENIZER['RESULT_']] + str_to_tokens(res_str))
        return cot_tokens, res_str

    def _get_subtraction_cot(a, b):
        res = a - b
        res_str = str(res)
        a_orig, b_orig = a, b

        cot_tokens = []
        cot_tokens.extend(str_to_tokens(str(a_orig)) + [TOKENIZER['-']] + str_to_tokens(str(b_orig)) + [TOKENIZER['\n']])
        
        is_neg = a < b
        if is_neg: a, b = b, a
        s_a, s_b = str(a), str(b)
        max_len = max(len(s_a), len(s_b))
        s_a, s_b = s_a.zfill(max_len), s_b.zfill(max_len)

        borrow = 0
        for i in range(max_len - 1, -1, -1):
            d1 = int(s_a[i]) - borrow
            d2 = int(s_b[i])
            
            if d1 < d2:
                d1 += 10
                borrow = 1
            else:
                borrow = 0
            
            cot_tokens.extend(str_to_tokens(s_a[i]) + [TOKENIZER['-']] + str_to_tokens(s_b[i]))
            cot_tokens.extend([TOKENIZER[f'BORROW_{borrow}'], TOKENIZER['\n']])

        cot_tokens.extend([TOKENIZER['RESULT_']] + str_to_tokens(res_str))
        return cot_tokens, res_str

    def _get_multiplication_cot(a, b):
        cot_tokens = []
        s_a, s_b = str(a), str(b)
        cot_tokens.extend(str_to_tokens(s_a) + [TOKENIZER['×']] + str_to_tokens(s_b) + [TOKENIZER['\n']])
        
        partial_products = []
        for i, digit in enumerate(reversed(s_b)):
            multiplier = int(digit) * (10**i)
            partial_product = a * multiplier
            partial_products.append(partial_product)
            cot_tokens.extend(str_to_tokens(s_a) + [TOKENIZER['×']] + str_to_tokens(str(multiplier)) + [TOKENIZER['=']] + str_to_tokens(str(partial_product)) + [TOKENIZER['\n']])
        
        if len(partial_products) > 1:
            current_sum = 0
            for p in partial_products:
                new_sum = current_sum + p
                cot_tokens.extend(str_to_tokens(str(current_sum)) + [TOKENIZER['+']] + str_to_tokens(str(p)) + [TOKENIZER['=']] + str_to_tokens(str(new_sum)) + [TOKENIZER['\n']])
                current_sum = new_sum
        
        res = a * b
        res_str = str(res)
        cot_tokens.extend([TOKENIZER['RESULT_']] + str_to_tokens(res_str))
        return cot_tokens, res_str

    while len(sents) < num_samples:
        operand_range = random.choice(operand_ranges)
        a = random.randint(*operand_range)
        b = random.randint(*operand_range)
        op_str = random.choice(['+', '-', '×'])
        
        if op_str == '+': cot_tokens, res_str = _get_addition_cot(a, b)
        elif op_str == '-': cot_tokens, res_str = _get_subtraction_cot(a, b)
        else: cot_tokens, res_str = _get_multiplication_cot(a, b)

        problem_tokens = str_to_tokens(str(a)) + [TOKENIZER[op_str]] + str_to_tokens(str(b))
        answer_tokens = str_to_tokens(res_str)

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
        computation_output = F.linear(x_reshaped, self.mu_weight * F.softplus(self.sigma_weight), self.mu_bias)
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
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask, is_causal=True, need_weights=False)
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
    model.eval()
    correct_count = 0
    total_count = ood_dataset_x.size(0)
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
            except ValueError:
                continue

            prompt = torch.tensor(prompt_tokens, device=CONFIG["DEVICE"]).unsqueeze(0)
            
            generated_tokens = []
            max_new_tokens = 50 
            for _ in range(max_new_tokens):
                logits, _, _ = model(prompt)
                next_token = logits[0, -1, :].argmax()
                if next_token.item() == TOKENIZER['EOS']: break
                generated_tokens.append(next_token.item())
                prompt = torch.cat([prompt, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                if prompt.size(1) >= CONFIG["SEQ_LEN"]: break

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
            
            if end_idx != -1:
                gen_answer_str = gen_full_str[end_idx + len(think_end_tag):]
            else:
                gen_answer_str = gen_full_str

            if true_answer_str == gen_answer_str:
                correct_count += 1

            if i < 5:
                prompt_str = "".join([REVERSE_TOKENIZER.get(t, '?') for t in prompt_tokens if t != TOKENIZER['BOS']])
                full_generated_str = "".join([REVERSE_TOKENIZER.get(t, '?') for t in generated_tokens])

                think_start_tag = "<think>"
                think_end_tag = "</think>"
                start_idx = full_generated_str.find(think_start_tag)
                end_idx = full_generated_str.rfind(think_end_tag)

                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    think_content = full_generated_str[start_idx + len(think_start_tag):end_idx]
                    think_content_formatted = "\n    ".join(think_content.split("\\n"))
                    result_content = full_generated_str[end_idx + len(think_end_tag):]
                else:
                    think_content_formatted = "[Could not parse think block]"
                    result_content = full_generated_str
                
                status = '✅' if true_answer_str == result_content else '❌'

                output = (
                    f"--- Sample {i+1} ---\n"
                    f"[bold cyan]Prompt  :[/bold cyan] {prompt_str}\n"
                    f"[bold yellow]Think   :[/bold yellow]\n    {think_content_formatted}\n"
                    f"[bold magenta]Result  :[/bold magenta] {result_content}\n"
                    f"[bold green]Expected:[/bold green] {true_answer_str}\n"
                    f"[bold]Status  :[/bold] {status}"
                )
                sample_outputs.append(output)

    if sample_outputs:
        console.print("\n" + "\n\n".join(sample_outputs))

    accuracy = correct_count / total_count if total_count > 0 else 0
    console.print(f"[bold green]OOD Exact Match Accuracy: {accuracy:.4f}[/bold green]")
    console.print("[bold yellow]----------------------[/bold yellow]\n")
    model.train()

def run_experiment():
    model = PocModel().to(device=CONFIG["DEVICE"], dtype=CONFIG["DTYPE"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LR"])
    
    start_step = 0
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step'] + 1
    
    run_experiment.last_log_time = time.time()

    dataset_x, dataset_y = generate_arithmetic_data(CONFIG["DATASET_SIZE"], CONFIG["SEQ_LEN"], operand_ranges=[(0, 9999)])
    dataset_x, dataset_y = dataset_x.to(CONFIG["DEVICE"]), dataset_y.to(CONFIG["DEVICE"])
    
    ood_dataset_x, ood_dataset_y = generate_arithmetic_data(96, CONFIG["SEQ_LEN"], operand_ranges=[(9000, 9999), (10000, 20000)])
    ood_dataset_x, ood_dataset_y = ood_dataset_x.to(CONFIG["DEVICE"]), ood_dataset_y.to(CONFIG["DEVICE"])

    model.train()
    for step in range(start_step, CONFIG["TRAINING_STEPS"]):
        batch_indices = torch.randint(0, dataset_x.size(0), (CONFIG["BATCH_SIZE"],))
        x, labels = dataset_x[batch_indices], dataset_y[batch_indices]
        
        optimizer.zero_grad()
        
        logits, scores, masked_outputs = model(x)
        
        loss_mask = torch.zeros_like(labels, dtype=torch.bool)
        for i, row in enumerate(x):
            equals_pos_list = (row == TOKENIZER['=']).nonzero(as_tuple=True)[0]
            if equals_pos_list.numel() > 0:
                loss_mask[i, equals_pos_list[0].item():] = True
        
        masked_labels = labels.masked_fill(~loss_mask, IGNORE_INDEX)

        main_loss = F.cross_entropy(logits.view(-1, CONFIG["VOCAB_SIZE"]), masked_labels.view(-1), ignore_index=IGNORE_INDEX)
        
        if not masked_outputs: continue
        
        surprise_grads = torch.autograd.grad(main_loss, masked_outputs, retain_graph=True, allow_unused=True)
        
        gate_loss = torch.tensor(0.0, device=x.device)
        for i, grad_tensor in enumerate(surprise_grads):
            if grad_tensor is not None:
                surprise_per_neuron = grad_tensor.view(-1, grad_tensor.shape[-1]).norm(p=2, dim=0)
                active_mask = (masked_outputs[i].abs().sum(dim=(0, 1)) > 0)
                active_surprise = surprise_per_neuron[active_mask]
                if active_surprise.numel() > 0:
                    gate_loss += (-torch.log(active_surprise + 1e-9) * active_surprise).sum()

        avg_tau = torch.distributions.Categorical(logits=logits.detach()).entropy()[loss_mask].mean() if loss_mask.any() else torch.tensor(0.0)
        prior_std = F.softplus(avg_tau)
        
        kl_loss = torch.tensor(0.0, device=x.device)
        for block in model.blocks:
            layer = block.ffn1
            q_w = torch.distributions.Normal(layer.mu_weight, F.softplus(layer.sigma_weight))
            p_w = torch.distributions.Normal(torch.zeros_like(layer.mu_weight), prior_std)
            kl_loss += torch.distributions.kl_divergence(q_w, p_w).mean()
        if model.blocks: kl_loss /= len(model.blocks)

        w_gate = 1.0 - torch.sigmoid(main_loss.detach())
        total_loss = main_loss + w_gate * gate_loss + kl_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 50 == 0:
            end_time = time.time()
            steps_per_sec = 50 / (end_time - getattr(run_experiment, "last_log_time", end_time))
            run_experiment.last_log_time = end_time

            with torch.no_grad():
                accuracy = (logits.argmax(-1) == labels)[loss_mask].float().mean() if loss_mask.any() else torch.tensor(0.0)
                all_sigmas = [p for name, p in model.named_parameters() if 'sigma_weight' in name]
                all_gates = [p for name, p in model.named_parameters() if 'gate_param' in name]
                avg_sigma = torch.mean(torch.stack([F.softplus(s).mean() for s in all_sigmas])).item()
                avg_gate = torch.mean(torch.stack([g.mean() for g in all_gates])).item()
                total_active_elements = sum((mo != 0).float().sum() for mo in masked_outputs)
                total_elements = sum(mo.numel() for mo in masked_outputs)
                activation_rate = (total_active_elements / total_elements * 100).item() if total_elements > 0 else 0
                gate_loss_val = gate_loss.item()
                kl_loss_val = kl_loss.item()
                
                console.print(f"Step {step:5d} | Loss(m/g/k): {main_loss.item():.3f}/{gate_loss_val:.3f}/{kl_loss_val:.3f} | Acc: {accuracy.item():.3f} | Avg σ/g: {avg_sigma:.4f}/{avg_gate:.4f} | Act%: {activation_rate:.2f} | τ/p_std: {avg_tau.item():.3f}/{prior_std.item():.3f} | it/s: {steps_per_sec:.2f}")
            
            if step > 0 and step % 500 == 0:
                evaluate_ood(model, ood_dataset_x, ood_dataset_y)

            torch.save({
                'step': step, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, CKPT_PATH)

if __name__ == "__main__":
    run_experiment()