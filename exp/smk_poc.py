import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput
import json
from tqdm import tqdm
import os
from pathlib import Path
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16

class STEGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, mask):
        return mask.to(scores.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.expand_as(grad_output.squeeze(-1)), None

class Config(PretrainedConfig):
    model_type = "smk_poc"
    hidden_size = 32
    intermediate_size = 32
    num_experts = 32
    num_attention_heads = 4
    max_seq_len = 256
    learning_rate = 1e-3
    epochs = 100
    gate_lr_scale = 1.0
    max_grad_norm = 1.0
    w_smk = 1.0
    w_diversity = 0.5

class TinyExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = F.gelu

    def forward(self, hidden_states):
        return self.w2(self.act(self.w1(hidden_states)))

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        B, T, C = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(attn_output)

class DynamicTopKGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sim_matrix = nn.Parameter(torch.randn(config.hidden_size, config.num_experts))
        self.threshold = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states):
        scores = torch.matmul(F.normalize(hidden_states, dim=-1), F.normalize(self.sim_matrix, dim=0))
        activated_mask = (scores > self.threshold).bool()
        k_per_token = activated_mask.sum(dim=-1)
        masked_scores = torch.where(activated_mask, scores, torch.tensor(-1e9, device=scores.device, dtype=scores.dtype))
        routing_weights = F.softmax(masked_scores, dim=-1)
        return routing_weights, scores, k_per_token, activated_mask

    def diversity_loss(self):
        norm_sim_matrix = F.normalize(self.sim_matrix, dim=0)
        cos_sim = torch.matmul(norm_sim_matrix.T, norm_sim_matrix)
        identity = torch.eye(self.config.num_experts, device=cos_sim.device)
        return torch.norm(cos_sim - identity)

class SMKMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = DynamicTopKGate(config)
        self.experts = nn.ModuleList([TinyExpert(config) for _ in range(self.config.num_experts)])

    def forward(self, hidden_states):
        B, T, C = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, C)
        routing_weights, router_scores, k_per_token, activated_mask = self.gate(hidden_states_flat)
        
        expert_outputs_full = torch.zeros(hidden_states_flat.shape[0], self.config.num_experts, C, device=DEVICE, dtype=DTYPE)
        
        activated_indices = torch.nonzero(activated_mask, as_tuple=True)
        if activated_indices[0].numel() > 0:
            token_indices, expert_indices = activated_indices
            expert_inputs = hidden_states_flat[token_indices]
            expert_outputs_sparse = torch.empty_like(expert_inputs)
            for i in range(self.config.num_experts):
                expert_mask = expert_indices == i
                if expert_mask.any():
                    expert_outputs_sparse[expert_mask] = self.experts[i](expert_inputs[expert_mask])
            expert_outputs_full.index_put_(activated_indices, expert_outputs_sparse)
        
        ste_mask = STEGate.apply(router_scores, activated_mask)
        weighted_expert_outputs = expert_outputs_full * ste_mask.unsqueeze(-1)
        final_hidden_states = weighted_expert_outputs.sum(dim=1)
        
        return final_hidden_states.view(B, T, C), router_scores, expert_outputs_full, k_per_token

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        self.moe = SMKMoE(config)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states):
        attn_output = self.attn(self.ln1(hidden_states))
        hidden_states = hidden_states + attn_output
        moe_output, router_scores, expert_outputs, k_per_token = self.moe(self.ln2(hidden_states))
        hidden_states = hidden_states + moe_output
        return hidden_states, router_scores, expert_outputs, k_per_token

class SMKPoCModel(PreTrainedModel, GenerationMixin):
    config_class = Config

    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.block = TransformerBlock(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, labels=None, **kwargs):
        hidden_states = self.embedding(input_ids)
        hidden_states, router_scores, expert_outputs, k_per_token = self.block(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1), ignore_index=-100)
            
        outputs = CausalLMOutput(logits=logits, loss=loss)

        if not self.training:
            return outputs
        
        return outputs, router_scores, expert_outputs, k_per_token

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}
    
def main():
    checkpoint_dir = Path("output/smk_poc_checkpoint")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    config = Config(vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id)
    model = SMKPoCModel(config).to(DEVICE, dtype=DTYPE)
    
    optimizer_main = AdamW([p for n, p in model.named_parameters() if 'gate' not in n], lr=config.learning_rate)
    optimizer_gate = AdamW(model.block.moe.gate.parameters(), lr=config.learning_rate * config.gate_lr_scale)

    with open("data/dummy_chat_data.jsonl", "r", encoding="utf-8") as f: data = [json.loads(line) for line in f]

    for epoch in range(config.epochs):
        pbar = tqdm(data, desc=f"Epoch {epoch+1}")
        for item in pbar:
            model.train()
            
            user_content = next(msg['content'] for msg in item['messages'] if msg['role'] == 'user')
            assistant_content = next(msg['content'] for msg in item['messages'] if msg['role'] == 'assistant')
            
            user_prompt = f"user: {user_content} assistant: "
            full_text = f"{user_prompt}{assistant_content}"
            
            full_ids = tokenizer(full_text, return_tensors="pt", max_length=config.max_seq_len, truncation=True, padding="max_length").input_ids.to(DEVICE)
            user_len = tokenizer(user_prompt, return_tensors="pt").input_ids.shape[1]

            labels = full_ids.clone()
            if user_len >= config.max_seq_len: continue
            labels[:, :user_len] = -100

            # --- 1f2b2o ---
            optimizer_main.zero_grad()
            optimizer_gate.zero_grad()

            outputs, router_scores, expert_outputs, k_per_token = model(full_ids, labels=labels)
            main_loss = outputs.loss
            if main_loss is None or torch.isnan(main_loss): continue

            with torch.no_grad():
                expert_grads, = torch.autograd.grad(main_loss, expert_outputs, retain_graph=True)
                m_a = expert_outputs.detach().sum(dim=-1)
                m_g = expert_grads.sum(dim=-1)
                m_dsc_scores = m_a - m_g
                target_indices = torch.argmax(m_dsc_scores, dim=-1)

            smk_loss = F.cross_entropy(router_scores, target_indices)
            div_loss = model.block.moe.gate.diversity_loss()
            gate_loss = (config.w_smk * smk_loss) + (config.w_diversity * div_loss)
            
            total_loss = main_loss + gate_loss
            total_loss.backward()

            optimizer_main.step()
            optimizer_gate.step()

            with torch.no_grad():
                main_acc = (outputs.logits.argmax(-1)[labels != -100] == labels[labels != -100]).float().mean() if (labels != -100).any() else torch.tensor(0.0)
                smk_acc = (router_scores.argmax(-1) == target_indices).float().mean()

            pbar.set_postfix_str(f"main:{main_loss.item():.2f} acc:{main_acc.item():.2f}|gate:{gate_loss.item():.2f} acc:{smk_acc.item():.2f} k:{k_per_token.float().mean().item():.2f}")

        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        
        model.eval()
        with torch.no_grad():
            val_item = random.choice(data)
            val_user_content = next((msg['content'] for msg in val_item['messages'] if msg['role'] == 'user'), "")
            val_text = f"user: {val_user_content} assistant: "
            val_input_ids = tokenizer(val_text, return_tensors='pt').input_ids.to(DEVICE)
            
            output_ids = model.generate(val_input_ids, max_new_tokens=40, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"\n--- VAL (Epoch {epoch+1}) ---\nInput: {val_text}\nModel: {response}\n")

if __name__ == "__main__":
    main()