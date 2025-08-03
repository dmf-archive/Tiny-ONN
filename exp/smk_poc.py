import json
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        return (scores > 0).to(scores.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

@dataclass
class Config(PretrainedConfig):
    model_type = "unified_hybrid_loss_poc"
    hidden_size: int = 32
    intermediate_size: int = 128
    num_experts: int = 32
    vocab_size: int = 50257
    num_attention_heads: int = 4
    max_seq_len: int = 256
    learning_rate: float = 1e-3
    epochs: int = 30
    w_aux: float = 1.0
    w_ce: float = 1.0
    w_kl: float = 1.0

class Expert(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))

class DynamicGate(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.sim_matrix = nn.Parameter(torch.randn(config.hidden_size, config.num_experts))
        self.gates = nn.Parameter(torch.zeros(config.num_experts))

    def forward(self, x: torch.Tensor):
        logits = torch.matmul(F.normalize(x, dim=-1), F.normalize(self.sim_matrix, dim=0))
        gate_thresholds = torch.sigmoid(self.gates)
        pre_activation_logits = logits - gate_thresholds
        
        gated_logits = F.relu(pre_activation_logits)
        activation_mask = STEFunction.apply(gated_logits)

        num_active_experts = torch.sum(activation_mask, dim=1)
        inactive_tokens_mask = num_active_experts == 0
        if inactive_tokens_mask.any():
            k_fallback = self.config.num_experts // 2
            topk_expert_indices = torch.topk(logits[inactive_tokens_mask], k=k_fallback, dim=1).indices
            for i, idx in enumerate(torch.where(inactive_tokens_mask)[0]):
                activation_mask[idx, topk_expert_indices[i]] = 1.0
        
        gated_logits_masked = torch.where(activation_mask > 0, gated_logits, -torch.finfo(DTYPE).max)
        active_expert_probs = F.softmax(gated_logits_masked, dim=-1)
        
        return active_expert_probs, pre_activation_logits, activation_mask

class DynamicMoELayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.experts = nn.ModuleList([Expert(config) for _ in range(self.num_experts)])
        self.gate = DynamicGate(config)

    def forward(self, hidden_states: torch.Tensor):
        num_tokens, C = hidden_states.shape
        routing_weights, pre_act_logits, activation_mask = self.gate(hidden_states)
        token_indices, expert_indices = torch.where(activation_mask > 0)
        
        full_expert_outputs = torch.zeros(num_tokens, self.num_experts, C, device=DEVICE, dtype=DTYPE)

        if token_indices.numel() > 0:
            flat_expert_inputs = hidden_states[token_indices]
            dispatched_outputs = torch.zeros_like(flat_expert_inputs)
            for i in range(self.num_experts):
                mask = expert_indices == i
                if mask.any():
                    dispatched_outputs[mask] = self.experts[i](flat_expert_inputs[mask])
            full_expert_outputs.index_put_((token_indices, expert_indices), dispatched_outputs)

        final_output = torch.einsum('te,tec->tc', routing_weights, full_expert_outputs)
        return final_output, full_expert_outputs, pre_act_logits, activation_mask

class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        B, T, C = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(attn_output)

class DynamicMoEModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.attn = Attention(config)
        self.moe = DynamicMoELayer(config)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        B, T = input_ids.shape
        hidden_states = self.embedding(input_ids)
        attn_output = self.attn(self.ln1(hidden_states))
        hidden_states = hidden_states + attn_output
        hidden_states_for_moe = self.ln2(hidden_states)
        flat_hs_for_moe = hidden_states_for_moe.view(-1, self.config.hidden_size)
        moe_output, full_expert_outputs, pre_act_logits, activation_mask = self.moe(flat_hs_for_moe)
        moe_output = moe_output.view(B, T, -1)
        hidden_states = hidden_states + moe_output
        logits = self.lm_head(hidden_states)
        return logits, full_expert_outputs, pre_act_logits, activation_mask

def generate_text(model, tokenizer, prompt, max_new_tokens=30):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        for _ in range(max_new_tokens):
            outputs, _, _, _ = model(input_ids)
            next_token_logits = outputs[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
    model.train()
    return tokenizer.decode(input_ids[0, len(tokenizer.encode(prompt)):] , skip_special_tokens=True)

def get_hybrid_gating_loss(main_loss, full_expert_outputs, pre_act_logits, config):
    grad_matrix, = torch.autograd.grad(main_loss, full_expert_outputs, retain_graph=True, allow_unused=True)
    if grad_matrix is None:
        return torch.tensor(0.0, device=DEVICE), torch.zeros_like(pre_act_logits)
        
    surprise_matrix = torch.linalg.norm(grad_matrix.float(), dim=-1)
    
    # Cross Entropy (Hard Target)
    with torch.no_grad():
        target_indices = torch.argmin(surprise_matrix, dim=-1)
    ce_loss = F.cross_entropy(pre_act_logits, target_indices)
    
    # KL Divergence (Soft Target)
    log_target_dist = F.log_softmax(-surprise_matrix, dim=-1)
    log_gate_dist = F.log_softmax(pre_act_logits, dim=-1)
    kl_loss = F.kl_div(log_gate_dist, log_target_dist, reduction='batchmean', log_target=True)
    
    combined_loss = config.w_ce * ce_loss + config.w_kl * kl_loss
    
    return combined_loss, surprise_matrix

def apply_gradient_filtering(model: nn.Module):
    expert_grads = []
    for param in model.moe.experts.parameters():
        if param.grad is not None:
            expert_grads.append(param.grad.view(-1))
    
    if not expert_grads:
        return

    all_grads = torch.cat(expert_grads)
    mean = all_grads.mean()
    std = all_grads.std()
    threshold = mean + 2 * std

    for param in model.moe.experts.parameters():
        if param.grad is not None:
            param.grad.data[param.grad.data > threshold] = 0.0

def main():
    config = Config()
    model = DynamicMoEModel(config).to(DEVICE, dtype=DTYPE)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", cache_dir="./weights", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    config.vocab_size = len(tokenizer)
    model.embedding = nn.Embedding(config.vocab_size, config.hidden_size).to(DEVICE, dtype=DTYPE)
    model.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(DEVICE, dtype=DTYPE)

    with open("data/dummy_chat_data.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        
    for epoch in range(config.epochs):
        pbar = tqdm(data, desc=f"Epoch {epoch+1}/{config.epochs}")
        total_main_loss, total_gating_loss, total_main_acc, total_avg_k, total_gate_acc = 0.0, 0.0, 0.0, 0.0, 0.0
        
        for item in pbar:
            optimizer.zero_grad()
            
            user_content = next((msg['content'] for msg in item['messages'] if msg['role'] == 'user'), "")
            assistant_content = next((msg['content'] for msg in item['messages'] if msg['role'] == 'assistant'), "")
            prompt = f"user: {user_content}\nassistant: "
            full_text = f"{prompt}{assistant_content}{tokenizer.eos_token}"
            inputs = tokenizer(full_text, return_tensors="pt", max_length=config.max_seq_len, truncation=True, padding="max_length")
            input_ids = inputs.input_ids.to(DEVICE)
            labels = input_ids.clone()
            prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
            if prompt_len < config.max_seq_len:
                labels[:, :prompt_len] = -100
            labels[labels == tokenizer.pad_token_id] = -100

            logits, full_expert_outputs, pre_act_logits, act_mask = model(input_ids)
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            main_loss = F.cross_entropy(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1), ignore_index=-100)
            if torch.isnan(main_loss): continue
            
            gating_loss, surprise_matrix = get_hybrid_gating_loss(main_loss, full_expert_outputs, pre_act_logits, config)
            
            combined_loss = main_loss + config.w_aux * gating_loss
            combined_loss.backward()
            
            apply_gradient_filtering(model)

            optimizer.step()

            with torch.no_grad():
                main_acc = (shift_logits.argmax(dim=-1) == shift_labels).float().mean()
                avg_k = torch.sum(act_mask) / act_mask.shape[0]
                gate_acc = (pre_act_logits.argmax(dim=-1) == surprise_matrix.argmin(dim=-1)).float().mean()
            
            total_main_loss += main_loss.item()
            total_gating_loss += gating_loss.item()
            total_main_acc += main_acc.item()
            total_avg_k += avg_k.item()
            total_gate_acc += gate_acc.item()

            pbar.set_postfix({
                "Main Acc": f"{main_acc.item():.2f}",
                "Gate Acc": f"{gate_acc.item():.2f}",
                "Avg K": f"{avg_k.item():.2f}",
                "Gate Loss": f"{gating_loss.item():.3f}"
            })

        avg_main_loss = total_main_loss / len(data)
        avg_gating_loss = total_gating_loss / len(data)
        avg_main_acc = total_main_acc / len(data)
        avg_k = total_avg_k / len(data)
        avg_gate_acc = total_gate_acc / len(data)
        print(f"Epoch Summary: Avg Main Loss: {avg_main_loss:.3f}, Avg Gate Loss: {avg_gating_loss:.3f}, Avg Main Acc: {avg_main_acc:.2f}, Avg Gate Acc: {avg_gate_acc:.2f}, Avg K: {avg_k:.2f}")
        
        print(f"Gate thresholds (sigmoid): {torch.sigmoid(model.moe.gate.gates).data.float().cpu().numpy()}")
        
        sample_item = random.choice(data)
        user_content = next((msg['content'] for msg in sample_item['messages'] if msg['role'] == 'user'), "")
        prompt = f"user: {user_content}\nassistant: "
        generated_text = generate_text(model, tokenizer, prompt)
        print(f"--- Sample Generation ---\n{prompt}{generated_text}\n-------------------------")

if __name__ == "__main__":
    main()
