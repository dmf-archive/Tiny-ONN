import json
import os
import random

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from .config import DEVICE, DTYPE, DynSMHAConfig
from .model import TinyOnnModel


class ChatDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizerFast, max_length: int, data: list):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data
        self.prompts, self.full_texts = [], []
        for item in self.data:
            user_content = next((msg['content'] for msg in item['messages'] if msg['role'] == 'user'), "")
            assistant_content = next((msg['content'] for msg in item['messages'] if msg['role'] == 'assistant'), "")
            prompt = f"user: {user_content}\nassistant: "
            self.prompts.append(prompt)
            self.full_texts.append(f"{prompt}{assistant_content}{tokenizer.eos_token}")

        self.inputs = self.tokenizer(self.full_texts, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        self.prompt_inputs = self.tokenizer(self.prompts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = self.inputs.input_ids[idx]
        labels = input_ids.clone()
        prompt_len = self.prompt_inputs.attention_mask[idx].sum()
        labels[:prompt_len] = -100
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        return input_ids, labels

def get_hybrid_gating_loss(main_loss: torch.Tensor, forward_cache: dict, config: DynSMHAConfig) -> tuple[torch.Tensor, torch.Tensor]:
    all_sha_outputs = forward_cache["all_sha_outputs"]
    gate_cache = forward_cache["gate_cache"]
    logits = gate_cache["logits"] 
    
    B, T, E, H = all_sha_outputs.shape
    
    grad_matrix, = torch.autograd.grad(main_loss, all_sha_outputs, create_graph=True, allow_unused=True)

    surprise = torch.zeros(B, T, E, device=DEVICE, dtype=DTYPE)
    if grad_matrix is not None:
        surprise = torch.sum(grad_matrix.to(DTYPE).float().abs(), dim=3)

    flat_logits = logits.view(-1, E)
    flat_surprise = surprise.view(-1, E)

    with torch.no_grad():
        target_indices = torch.argmin(flat_surprise, dim=-1)
        gating_acc = (flat_logits.argmax(dim=-1) == target_indices).float().mean()
    
    ce_loss = F.cross_entropy(flat_logits, target_indices)

    log_target_dist = F.log_softmax(-flat_surprise, dim=-1)
    log_gate_dist = F.log_softmax(flat_logits, dim=-1)
    kl_loss = F.kl_div(log_gate_dist, log_target_dist, reduction='batchmean', log_target=True)

    combined_loss = config.w_ce * ce_loss + config.w_kl * kl_loss
    return combined_loss, gating_acc

def generate_text(model, tokenizer, prompt, max_new_tokens=30, top_p=0.9):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        for _ in range(max_new_tokens):
            outputs, _, _ = model(input_ids)
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
            if tokenizer.eos_token_id is not None and next_token_id.item() == tokenizer.eos_token_id:
                break
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
    model.train()
    return tokenizer.decode(input_ids[0, len(tokenizer(prompt).input_ids):], skip_special_tokens=True)

def main():
    config = DynSMHAConfig()
    
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

    model = TinyOnnModel(config).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    dataset = ChatDataset("data/dummy_chat_data.jsonl", tokenizer, config.max_seq_len, data)
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    
    for epoch in range(config.epochs):
        totals = {
            "main_loss": torch.tensor(0.0, device=DEVICE), "gating_loss": torch.tensor(0.0, device=DEVICE),
            "main_acc": torch.tensor(0.0, device=DEVICE), "avg_k": torch.tensor(0.0, device=DEVICE),
            "gate_acc": torch.tensor(0.0, device=DEVICE),
        }
        
        print(f"--- Starting Epoch {epoch+1}/{config.epochs} ---")
        for i, (input_ids, labels) in enumerate(data_loader):
            input_ids, labels = input_ids.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            final_logits, _, forward_cache = model(input_ids)

            main_loss = F.cross_entropy(
                rearrange(final_logits, 'b t d -> (b t) d'),
                rearrange(labels, 'b t -> (b t)'),
                ignore_index=-100
            )

            if torch.isnan(main_loss): continue

            gating_loss, gating_acc = get_hybrid_gating_loss(main_loss, forward_cache, config)
            total_loss = main_loss + config.w_aux * gating_loss
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                totals["main_loss"] += main_loss
                totals["gating_loss"] += gating_loss
                totals["main_acc"] += (final_logits.argmax(-1) == labels).float().mean()
                totals["avg_k"] += forward_cache["num_active_experts"]
                totals["gate_acc"] += gating_acc

            current_step = i + 1
            if current_step % 20 == 0 or current_step == len(data_loader):
                avg_main_loss_step = (totals['main_loss'] / current_step).item()
                avg_gate_loss_step = (totals['gating_loss'] / current_step).item()
                avg_main_acc_step = (totals['main_acc'] / current_step).item()
                avg_gate_acc_step = (totals['gate_acc'] / current_step).item()
                avg_k_step = (totals['avg_k'] / current_step).item()
                print(f"  Step {current_step}/{len(data_loader)}: "
                      f"Main Loss: {avg_main_loss_step:.3f}, Gate Loss: {avg_gate_loss_step:.3f}, "
                      f"Main Acc: {avg_main_acc_step:.2f}, Gate Acc: {avg_gate_acc_step:.2f}, Avg K: {avg_k_step:.2f}")

        avg_metrics = {k: (v / len(data_loader)).item() for k, v in totals.items()}
        print(f"Epoch Summary: Avg Main Loss: {avg_metrics['main_loss']:.3f}, Avg Gate Loss: {avg_metrics['gating_loss']:.3f}, "
              f"Avg Main Acc: {avg_metrics['main_acc']:.2f}, Avg Gate Acc: {avg_metrics['gate_acc']:.2f}, Avg K: {avg_metrics['avg_k']:.2f}")
        
        sample_item = random.choice(dataset.data)
        prompt = f"user: {sample_item['messages'][0]['content']}\nassistant: "
        generated_text = generate_text(model, tokenizer, prompt)
        print(f"--- Sample Generation ---\n{prompt}{generated_text}\n-------------------------")

if __name__ == "__main__":
    main()
