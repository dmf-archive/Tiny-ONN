import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from dataclasses import dataclass
from tqdm import tqdm
import json
from transformers import AutoTokenizer, PretrainedConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16

@dataclass
class ModelConfig(PretrainedConfig):
    model_type: str = "smk_poc"
    hidden_size: int = 32
    intermediate_size: int = 128
    num_experts: int = 32
    top_k: int = 2
    vocab_size: int = 50257
    num_attention_heads: int = 4
    max_seq_len: int = 256
    learning_rate: float = 5e-4
    gate_learning_rate: float = 5e-3
    epochs: int = 10
    batch_size: int = 8

class Expert(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))

class MoELayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.experts = nn.ModuleList([Expert(config) for _ in range(self.num_experts)])
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        num_tokens, C = hidden_states.shape
        router_logits = self.gate(hidden_states)
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float).to(DTYPE)

        full_expert_outputs = torch.zeros(
            num_tokens, self.num_experts, C, device=DEVICE, dtype=DTYPE
        )
        
        flat_expert_indices = selected_experts.flatten()
        flat_token_indices = (
            torch.arange(num_tokens, device=DEVICE)
            .unsqueeze(1)
            .expand(-1, self.top_k)
            .flatten()
        )
        
        flat_hidden_states = hidden_states.repeat_interleave(self.top_k, dim=0)
        expert_inputs = torch.zeros_like(flat_hidden_states)
        
        for i in range(self.num_experts):
            mask = flat_expert_indices == i
            if mask.any():
                expert_inputs[mask] = self.experts[i](flat_hidden_states[mask])
        
        full_expert_outputs.index_put_(
            (flat_token_indices, flat_expert_indices), expert_inputs
        )

        bmm_routing_weights = routing_weights.unsqueeze(1)
        
        gathered_expert_outputs = torch.gather(
            full_expert_outputs, 
            1, 
            selected_experts.unsqueeze(-1).expand(-1, -1, C)
        )
        
        final_output = torch.bmm(bmm_routing_weights, gathered_expert_outputs).squeeze(1)
        return final_output, full_expert_outputs, router_logits

class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
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

class SimpleTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.attn = Attention(config)
        self.moe = MoELayer(config)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        B, T = input_ids.shape
        hidden_states = self.embedding(input_ids)
        attn_output = self.attn(self.ln1(hidden_states))
        hidden_states = hidden_states + attn_output
        
        hidden_states_for_moe = self.ln2(hidden_states)
        moe_output, full_expert_outputs, router_logits = self.moe(hidden_states_for_moe.view(-1, self.moe.gate.in_features))
        moe_output = moe_output.view(B, T, -1)
        
        hidden_states = hidden_states + moe_output
        logits = self.lm_head(hidden_states)
        return logits, full_expert_outputs, router_logits, hidden_states_for_moe

def get_surprise_matrix(loss: torch.Tensor, full_expert_outputs: torch.Tensor) -> torch.Tensor:
    grad_matrix, = torch.autograd.grad(
        loss, full_expert_outputs, retain_graph=True, allow_unused=False
    )
    return torch.linalg.norm(grad_matrix.float(), dim=-1)

def main():
    config = ModelConfig()
    model = SimpleTransformer(config).to(DEVICE, dtype=DTYPE)

    main_params = [p for n, p in model.named_parameters() if 'gate' not in n]
    gate_params = model.moe.gate.parameters()
    optimizer_main = AdamW(main_params, lr=config.learning_rate)
    optimizer_gate = AdamW(gate_params, lr=config.gate_learning_rate)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", cache_dir="./weights", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    config.vocab_size = len(tokenizer)
    model.embedding = nn.Embedding(config.vocab_size, config.hidden_size).to(DEVICE, dtype=DTYPE)
    model.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(DEVICE, dtype=DTYPE)

    try:
        with open("data/dummy_chat_data.jsonl", "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("Creating dummy data...")
        dummy_data = [
            {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]},
            {"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I'm doing great, thanks!"}]}
        ]
        with open("data/dummy_chat_data.jsonl", "w", encoding="utf-8") as f:
            for item in dummy_data:
                f.write(json.dumps(item) + "\n")
        data = dummy_data
        
    for epoch in range(config.epochs):
        print(f"\n--- Epoch {epoch+1}/{config.epochs} ---")
        pbar = tqdm(data)
        total_main_loss = 0.0
        total_gate_loss = 0.0
        
        for item in pbar:
            optimizer_main.zero_grad()
            optimizer_gate.zero_grad()

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

            logits, full_expert_outputs, _, hidden_states_for_moe = model(input_ids)
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            main_loss = F.cross_entropy(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1), ignore_index=-100)
            if torch.isnan(main_loss): continue

            surprise_matrix = get_surprise_matrix(main_loss, full_expert_outputs)
            main_loss.backward()
            
            for param in model.moe.gate.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            optimizer_main.step()

            detached_hidden_states = hidden_states_for_moe.detach().view(-1, config.hidden_size)
            recomputed_router_logits = model.moe.gate(detached_hidden_states)

            target_indices = torch.argmin(surprise_matrix, dim=-1)
            
            gating_loss = F.cross_entropy(recomputed_router_logits, target_indices)
            gating_loss.backward()
            optimizer_gate.step()

            total_main_loss += main_loss.item()
            total_gate_loss += gating_loss.item()
            pbar.set_description(f"Main Loss: {main_loss.item():.4f} | Gate Loss: {gating_loss.item():.4f}")

        avg_main_loss = total_main_loss / len(data)
        avg_gate_loss = total_gate_loss / len(data)
        print(f"Epoch Summary: Avg Main Loss: {avg_main_loss:.4f}, Avg Gate Loss: {avg_gate_loss:.4f}")

if __name__ == "__main__":
    main()