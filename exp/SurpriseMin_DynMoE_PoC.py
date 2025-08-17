import json
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, PretrainedConfig

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
    model_type = "SurpriseMin_DynMoE"
    hidden_size: int = 512
    intermediate_size: int = 64
    num_experts: int = 32
    vocab_size: int = 5000
    num_attention_heads: int = 16
    max_seq_len: int = 256
    learning_rate: float = 3e-3
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
            fallback_logits = logits[inactive_tokens_mask]
            topk_expert_indices = torch.topk(fallback_logits, k=k_fallback, dim=1).indices
            inactive_indices = torch.where(inactive_tokens_mask)[0]
            expanded_inactive_indices = inactive_indices.unsqueeze(1).expand(-1, k_fallback)
            activation_mask[expanded_inactive_indices, topk_expert_indices] = 1.0
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
        full_expert_outputs = torch.zeros(num_tokens, self.num_experts, C, device=DEVICE, dtype=DTYPE, requires_grad=True)
        if token_indices.numel() > 0:
            sorted_expert_indices, sorting_indices = torch.sort(expert_indices)
            undo_sorting_indices = torch.argsort(sorting_indices)
            dispatched_inputs = hidden_states[token_indices][sorting_indices]
            expert_boundaries = torch.searchsorted(sorted_expert_indices, torch.arange(self.num_experts + 1, device=DEVICE))
            output_chunks = []
            for i in range(self.num_experts):
                start, end = expert_boundaries[i], expert_boundaries[i+1]
                if start < end:
                    output_chunks.append(self.experts[i](dispatched_inputs[start:end]))
            if output_chunks:
                sorted_outputs = torch.cat(output_chunks, dim=0)
                dispatched_outputs = sorted_outputs[undo_sorting_indices]
                temp_outputs = torch.zeros_like(full_expert_outputs)
                temp_outputs.index_put_((token_indices, expert_indices), dispatched_outputs)
                full_expert_outputs = full_expert_outputs + temp_outputs
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

def generate_text(model, tokenizer, prompt, max_new_tokens=30, top_p=0.9):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        for _ in range(max_new_tokens):
            outputs, _, _, _ = model(input_ids)
            next_token_logits = outputs[:, -1, :]
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[:, indices_to_remove] = -float("Inf")
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
    model.train()
    return tokenizer.decode(input_ids[0, len(tokenizer(prompt).input_ids):] , skip_special_tokens=True)

def get_hybrid_gating_loss(main_loss, full_expert_outputs, pre_act_logits, config):
    grad_matrix, = torch.autograd.grad(main_loss, full_expert_outputs, create_graph=True, allow_unused=True)
    if grad_matrix is None:
        return torch.tensor(0.0, device=DEVICE), torch.zeros_like(pre_act_logits)
    surprise_matrix = torch.linalg.norm(grad_matrix.float(), dim=-1)
    with torch.no_grad():
        target_indices = torch.argmin(surprise_matrix, dim=-1)
    ce_loss = F.cross_entropy(pre_act_logits, target_indices)
    log_target_dist = F.log_softmax(-surprise_matrix, dim=-1)
    log_gate_dist = F.log_softmax(pre_act_logits, dim=-1)
    kl_loss = F.kl_div(log_gate_dist, log_target_dist, reduction='batchmean', log_target=True)
    combined_loss = config.w_ce * ce_loss + config.w_kl * kl_loss
    return combined_loss, surprise_matrix.detach()

def main():
    config = Config()
    
    with open("data/dummy_chat_data.jsonl", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    def get_training_corpus():
        return (
            f"user: {next((msg['content'] for msg in item['messages'] if msg['role'] == 'user'), '')}\n"
            f"assistant: {next((msg['content'] for msg in item['messages'] if msg['role'] == 'assistant'), '')}"
            for item in data
        )
    
    base_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    base_tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[EOS]"], vocab_size=config.vocab_size)
    base_tokenizer.train_from_iterator(get_training_corpus(), trainer)
    
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=base_tokenizer,
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    
    config.vocab_size = base_tokenizer.get_vocab_size()

    model = DynamicMoEModel(config).to(DEVICE, dtype=DTYPE)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    prompts, full_texts = [], []
    for item in data:
        user_content = next((msg['content'] for msg in item['messages'] if msg['role'] == 'user'), "")
        assistant_content = next((msg['content'] for msg in item['messages'] if msg['role'] == 'assistant'), "")
        prompt = f"user: {user_content}\nassistant: "
        prompts.append(prompt)
        full_texts.append(f"{prompt}{assistant_content}{tokenizer.eos_token}")

    inputs = tokenizer(full_texts, return_tensors="pt", max_length=config.max_seq_len, truncation=True, padding="max_length")
    prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=config.max_seq_len)
    
    all_input_ids = inputs.input_ids
    all_labels = all_input_ids.clone()
    prompt_lengths = prompt_inputs.attention_mask.sum(dim=1)

    for i, prompt_len in enumerate(prompt_lengths):
        all_labels[i, :prompt_len] = -100
    all_labels[all_labels == tokenizer.pad_token_id] = -100

    dataset_size = len(data)
    all_input_ids = all_input_ids.to(DEVICE, non_blocking=True)
    all_labels = all_labels.to(DEVICE, non_blocking=True)

    for epoch in range(config.epochs):
        totals = {
            "main_loss": torch.tensor(0.0, device=DEVICE), "gating_loss": torch.tensor(0.0, device=DEVICE),
            "main_acc": torch.tensor(0.0, device=DEVICE), "avg_k": torch.tensor(0.0, device=DEVICE),
            "gate_acc": torch.tensor(0.0, device=DEVICE),
        }
        print(f"--- Starting Epoch {epoch+1}/{config.epochs} ---")
        for i in range(dataset_size):
            optimizer.zero_grad(set_to_none=True)
            input_ids, labels = all_input_ids[i:i+1], all_labels[i:i+1]
            logits, full_expert_outputs, pre_act_logits, act_mask = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            main_loss = F.cross_entropy(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1), ignore_index=-100)
            if torch.isnan(main_loss): continue
            gating_loss, surprise_matrix = get_hybrid_gating_loss(main_loss, full_expert_outputs, pre_act_logits, config)
            combined_loss = main_loss + config.w_aux * gating_loss
            combined_loss.backward()
            optimizer.step()
            with torch.no_grad():
                totals["main_loss"] += main_loss
                totals["gating_loss"] += gating_loss
                totals["main_acc"] += (shift_logits.argmax(dim=-1) == shift_labels).float().mean()
                totals["avg_k"] += torch.sum(act_mask) / act_mask.shape[0]
                totals["gate_acc"] += (pre_act_logits.argmax(dim=-1) == surprise_matrix.argmin(dim=-1)).float().mean()
            current_step = i + 1
            if current_step % 20 == 0 or current_step == dataset_size:
                avg_main_loss_step = (totals['main_loss'] / current_step).item()
                avg_gate_loss_step = (totals['gating_loss'] / current_step).item()
                avg_main_acc_step = (totals['main_acc'] / current_step).item()
                avg_gate_acc_step = (totals['gate_acc'] / current_step).item()
                avg_k_step = (totals['avg_k'] / current_step).item()
                print(f"  Step {current_step}/{dataset_size}: "
                      f"Main Loss: {avg_main_loss_step:.3f}, Gate Loss: {avg_gate_loss_step:.3f}, "
                      f"Main Acc: {avg_main_acc_step:.2f}, Gate Acc: {avg_gate_acc_step:.2f}, Avg K: {avg_k_step:.2f}")
        avg_metrics = {k: (v / dataset_size).item() for k, v in totals.items()}
        print(f"Epoch Summary: Avg Main Loss: {avg_metrics['main_loss']:.3f}, Avg Gate Loss: {avg_metrics['gating_loss']:.3f}, "
              f"Avg Main Acc: {avg_metrics['main_acc']:.2f}, Avg Gate Acc: {avg_metrics['gate_acc']:.2f}, Avg K: {avg_metrics['avg_k']:.2f}")
        sample_item = random.choice(data)
        user_content = next((msg['content'] for msg in item['messages'] if msg['role'] == 'user'), "")
        prompt = f"user: {user_content}\nassistant: "
        generated_text = generate_text(model, tokenizer, prompt)
        print(f"--- Sample Generation ---\n{prompt}{generated_text}\n-------------------------")

if __name__ == "__main__":
    main()
