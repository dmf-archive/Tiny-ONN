import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput
import math
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16

class Config(PretrainedConfig):
    model_type = "smk_poc"
    hidden_size = 32
    intermediate_size = 32
    num_experts = 8
    num_attention_heads = 4
    max_seq_len = 128

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
        norm_hidden_states = F.normalize(hidden_states, dim=-1)
        norm_sim_matrix = F.normalize(self.sim_matrix, dim=0)
        scores = torch.matmul(norm_hidden_states, norm_sim_matrix)
        activated_mask = (scores > self.threshold).bool()
        k_per_token = activated_mask.sum(dim=-1)
        masked_scores = torch.where(activated_mask, scores, torch.tensor(-1e9, device=scores.device, dtype=scores.dtype))
        routing_weights = F.softmax(masked_scores, dim=-1)
        return routing_weights, scores, k_per_token

class SMKMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = DynamicTopKGate(config)
        self.experts = nn.ModuleList([TinyExpert(config) for _ in range(self.config.num_experts)])

    def forward(self, hidden_states):
        B, T, C = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, C)
        routing_weights, router_scores, k_per_token = self.gate(hidden_states_flat)
        final_hidden_states = torch.zeros_like(hidden_states_flat)
        expert_outputs = torch.stack([expert(hidden_states_flat) for expert in self.experts], dim=1)
        weighted_expert_outputs = expert_outputs * routing_weights.unsqueeze(-1)
        final_hidden_states = weighted_expert_outputs.sum(dim=1)
        return final_hidden_states.view(B, T, C), router_scores, None, k_per_token

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
        moe_output, _, _, _ = self.moe(self.ln2(hidden_states))
        hidden_states = hidden_states + moe_output
        return hidden_states

class SMKPoCModel(PreTrainedModel, GenerationMixin):
    config_class = Config

    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.block = TransformerBlock(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, **kwargs):
        hidden_states = self.embedding(input_ids)
        hidden_states = self.block(hidden_states)
        logits = self.lm_head(hidden_states)
        return CausalLMOutput(logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

def main():
    checkpoint_dir = Path("output/smk_poc_checkpoint")
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    model = SMKPoCModel.from_pretrained(str(checkpoint_dir)).to(DEVICE, dtype=DTYPE)
    model.eval()

    print("--- Interactive Chat ---")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            messages = [{"role": "user", "content": user_input}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)

            with torch.no_grad():
                output_ids = model.generate(input_ids, max_new_tokens=50, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"Model: {response}")

        except KeyboardInterrupt:
            print("\nExiting chat.")
            break

if __name__ == "__main__":
    main()