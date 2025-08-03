import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from dataclasses import dataclass
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16

@dataclass
class Config(PretrainedConfig):
    model_type = "kv_moe_poc"
    hidden_size: int = 64
    intermediate_size: int = 256
    num_attention_heads: int = 4
    vocab_size: int = 50257
    max_seq_len: int = 256
    learning_rate: float = 1e-3
    epochs: int = 30
    w_gate: float = 1.0
    w_ce: float = 1.0
    w_kl: float = 1.0
    
    # KV MoE specific
    num_kv_experts: int = 16
    kv_block_size: int = 32

class DynamicGate(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.router = nn.Linear(config.hidden_size, config.num_kv_experts, bias=False)

    def forward(self, hidden_states):
        return self.router(hidden_states)

class DynamicKVProvider(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.gate = DynamicGate(config)
        
        self.k_experts = nn.Parameter(torch.randn(config.num_kv_experts, config.kv_block_size, config.hidden_size))
        self.v_experts = nn.Parameter(torch.randn(config.num_kv_experts, config.kv_block_size, config.hidden_size))

    def forward(self, hidden_states):
        B, T, C = hidden_states.shape
        flat_hs = hidden_states.view(-1, C)

        gate_logits = self.gate(flat_hs)
        routing_weights = F.softmax(gate_logits, dim=-1) # (B*T, num_kv_experts)

        dynamic_k = torch.einsum("te,ebc->tbc", routing_weights, self.k_experts)
        dynamic_v = torch.einsum("te,ebc->tbc", routing_weights, self.v_experts)

        return dynamic_k.view(B, T, self.config.kv_block_size, C), \
               dynamic_v.view(B, T, self.config.kv_block_size, C), \
               gate_logits.view(B, T, self.config.num_kv_experts)

class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, dynamic_k, dynamic_v):
        q = self.q_proj(hidden_states)
        # We attend from each token's query to its dynamically generated KV context
        # This requires careful reshaping to work with scaled_dot_product_attention
        B, T, C = hidden_states.shape
        q_reshaped = q.view(B * T, 1, C)
        k_reshaped = dynamic_k.view(B * T, self.config.kv_block_size, C)
        v_reshaped = dynamic_v.view(B * T, self.config.kv_block_size, C)

        attn_output = F.scaled_dot_product_attention(q_reshaped, k_reshaped, v_reshaped)
        attn_output = attn_output.view(B, T, C)
        return self.o_proj(attn_output)

class KVMoEModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.kv_provider = DynamicKVProvider(config)
        self.attn = Attention(config)
        self.mlp = nn.Linear(config.hidden_size, config.hidden_size)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Static KV Cache and Gate
        self.static_k = nn.Parameter(torch.randn(1, 1, config.kv_block_size, config.hidden_size))
        self.static_v = nn.Parameter(torch.randn(1, 1, config.kv_block_size, config.hidden_size))
        self.kv_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.embedding(input_ids)
        x_ln = self.ln1(x)
        
        dynamic_k, dynamic_v, gate_logits = self.kv_provider(x_ln)
        
        g = torch.sigmoid(self.kv_gate)
        k_final = g * dynamic_k + (1 - g) * self.static_k.expand(B, T, -1, -1)
        v_final = g * dynamic_v + (1 - g) * self.static_v.expand(B, T, -1, -1)

        attn_out = self.attn(x_ln, k_final, v_final)
        
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        logits = self.lm_head(x)

        grad_target = (self.kv_provider.k_experts, self.kv_provider.v_experts)
        return logits, grad_target, gate_logits

def get_gating_loss(main_loss, grad_target, gate_logits, config):
    k_experts, v_experts = grad_target
    flat_logits = gate_logits.view(-1, config.num_kv_experts)

    if not k_experts.requires_grad:
        return torch.tensor(0.0, device=DEVICE), torch.zeros_like(flat_logits)

    k_grads, v_grads = torch.autograd.grad(main_loss, [k_experts, v_experts], retain_graph=True, allow_unused=True)
    
    if k_grads is None or v_grads is None:
        return torch.tensor(0.0, device=DEVICE), torch.zeros_like(flat_logits).mean(-1)

    surprise_k = torch.linalg.norm(k_grads.float(), dim=(-1, -2))
    surprise_v = torch.linalg.norm(v_grads.float(), dim=(-1, -2))
    surprise = surprise_k + surprise_v

    with torch.no_grad():
        target_indices = torch.argmin(surprise).expand(flat_logits.size(0))
        
    ce_loss = F.cross_entropy(flat_logits, target_indices)
    
    log_target_dist = F.log_softmax(-surprise, dim=-1)
    log_gate_dist = F.log_softmax(flat_logits, dim=-1).mean(dim=0)
    kl_loss = F.kl_div(log_gate_dist, log_target_dist, reduction='batchmean', log_target=True)

    return config.w_ce * ce_loss + config.w_kl * kl_loss, surprise

def generate_text(model, tokenizer, prompt, max_new_tokens=30):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        for _ in range(max_new_tokens):
            logits, _, _ = model(input_ids)
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
    model.train()
    return tokenizer.decode(input_ids[0, len(tokenizer.encode(prompt)):] , skip_special_tokens=True)

def main():
    config = Config()
    model = KVMoEModel(config).to(DEVICE, dtype=DTYPE)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", cache_dir="./weights", trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    config.vocab_size = len(tokenizer)
    model.embedding = nn.Embedding(config.vocab_size, config.hidden_size).to(DEVICE, dtype=DTYPE)
    model.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(DEVICE, dtype=DTYPE)

    with open("data/dummy_chat_data.jsonl", "r", encoding="utf-8") as f: data = [json.loads(line) for line in f][:50]

    for epoch in range(config.epochs):
        total_main_loss, total_gating_loss, total_main_acc, total_gate_entropy, total_gate_acc = 0.0, 0.0, 0.0, 0.0, 0.0
        pbar = tqdm(data, desc=f"Epoch {epoch+1}/{config.epochs}")
        for item in pbar:
            optimizer.zero_grad()
            
            user_content = next((msg['content'] for msg in item['messages'] if msg['role'] == 'user'), "")
            assistant_content = next((msg['content'] for msg in item['messages'] if msg['role'] == 'assistant'), "")

            prompt = f"user: {user_content}\nassistant: "
            full_text = f"{prompt}{assistant_content}{tokenizer.eos_token}"
            
            inputs = tokenizer(full_text, return_tensors="pt", max_length=config.max_seq_len, truncation=True, padding="max_length")
            input_ids = inputs.input_ids.to(DEVICE)
            labels = input_ids.clone()
            
            prompt_len = len(tokenizer.encode(prompt))
            labels[:, :prompt_len] = -100
            labels[labels == tokenizer.pad_token_id] = -100

            logits, grad_target, gate_logits = model(input_ids)
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            main_loss = F.cross_entropy(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1), ignore_index=-100)
            if torch.isnan(main_loss) or torch.isinf(main_loss): continue
            
            gating_loss, surprise = get_gating_loss(main_loss, grad_target, gate_logits, config)
            
            loss = main_loss + config.w_gate * gating_loss
            if torch.isnan(loss): continue
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                active_tokens_mask = shift_labels.view(-1) != -100
                main_acc = (shift_logits.argmax(dim=-1).view(-1)[active_tokens_mask] == shift_labels.view(-1)[active_tokens_mask]).float().mean()
                
                routing_probs = F.softmax(gate_logits, dim=-1)
                gate_entropy = (-torch.sum(routing_probs * torch.log(routing_probs + 1e-9), dim=-1)).mean()
                gate_acc = (gate_logits.argmax(dim=-1).view(-1) == surprise.argmin().expand_as(gate_logits.argmax(dim=-1).view(-1))).float().mean()

            total_main_loss += main_loss.item()
            total_gating_loss += gating_loss.item()
            total_main_acc += main_acc.item()
            total_gate_entropy += gate_entropy.item()
            total_gate_acc += gate_acc.item()

            pbar.set_postfix({
                "Loss": f"{main_loss.item():.2f}",
                "Acc": f"{main_acc.item():.2f}",
                "GateLoss": f"{gating_loss.item():.2f}",
                "GateAcc": f"{gate_acc.item():.2f}",
                "Entropy": f"{gate_entropy.item():.2f}"
            })
        
        avg_main_loss = total_main_loss / len(data)
        avg_gating_loss = total_gating_loss / len(data)
        avg_main_acc = total_main_acc / len(data)
        avg_gate_entropy = total_gate_entropy / len(data)
        avg_gate_acc = total_gate_acc / len(data)
        print(f"Epoch Summary: Loss={avg_main_loss:.3f}, Acc={avg_main_acc:.3f}, GateLoss={avg_gating_loss:.3f}, GateAcc={avg_gate_acc:.3f}, AvgEntropy={avg_gate_entropy:.3f}")
        
        sample_item = random.choice(data)
        user_content = next((msg['content'] for msg in sample_item['messages'] if msg['role'] == 'user'), "")
        prompt = f"user: {user_content}\nassistant: "
        generated_text = generate_text(model, tokenizer, prompt)
        print(f"--- Sample Generation ---\nPrompt: {prompt}\nGenerated: {generated_text}\n-------------------------")


if __name__ == "__main__":
    main()