import json
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Dict, List
from einops import rearrange
from torch.utils.data import DataLoader, Dataset

from .config import DenseConfig, DynNSAConfig, DEVICE, DTYPE
from .model import DenseModel, DynNSAv2Model

class ChatDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = f"user: {item['messages'][0]['content']}\nassistant: "
        full_text = f"{prompt}{item['messages'][1]['content']}{self.tokenizer.eos_token}"
        
        inputs = self.tokenizer(full_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = inputs.input_ids.squeeze(0)
        
        labels = input_ids.clone()
        prompt_len = len(self.tokenizer.encode(prompt))
        labels[:prompt_len] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return input_ids, labels

class Visualizer:
    def __init__(self, output_dir="output/dyn_nsa_poc_v2"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def update_plot(self, history: Dict[str, List[float]]):
        epochs = range(1, len(history['dyn_nsa_loss']) + 1)
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        axs[0, 0].plot(epochs, history['dyn_nsa_loss'], 'o-', label='DynNSA Loss')
        axs[0, 0].plot(epochs, history['dense_loss'], 'o-', label='Dense Loss')
        axs[0, 0].set_title('Main Loss per Epoch')
        axs[0, 0].legend(); axs[0, 0].grid(True)

        axs[0, 1].plot(epochs, history['dyn_nsa_acc'], 'o-', label='DynNSA Accuracy')
        axs[0, 1].plot(epochs, history['dense_acc'], 'o-', label='Dense Accuracy')
        axs[0, 1].set_title('Accuracy per Epoch')
        axs[0, 1].legend(); axs[0, 1].grid(True)

        axs[1, 0].plot(epochs, history['gating_acc'], 'o-', label='Gating Accuracy')
        axs[1, 0].set_title('Gating Accuracy (DynNSA)')
        axs[1, 0].grid(True)
        
        axs[1, 1].plot(epochs, history['avg_heads'], 'o-', label='Average Active Heads')
        axs[1, 1].set_title('Average Active Heads (DynNSA)')
        axs[1, 1].grid(True)

        fig.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metrics_latest.png'))
        plt.close(fig)

def get_hybrid_gating_loss(main_loss, all_head_outputs, pre_act_logits_seq, config, step):
    if all_head_outputs is None or all_head_outputs.numel() == 0:
        return torch.tensor(0.0, device=DEVICE), torch.zeros_like(pre_act_logits_seq)
    
    grad_matrix, = torch.autograd.grad(main_loss, all_head_outputs, retain_graph=True, allow_unused=False)
    
    B, H, T, D = grad_matrix.shape
    grad_matrix_flat = rearrange(grad_matrix, 'b h t d -> (b t) h d')
    surprise_matrix_flat = torch.linalg.norm(grad_matrix_flat.float(), dim=-1)
    
    surprise_matrix_seq = rearrange(surprise_matrix_flat, '(b t) h -> b h t', b=B, t=T).mean(dim=2)

    if step % 20 == 0:
        print(f"\n--- Surprise (Step {step}) | Mean: {surprise_matrix_seq.mean():.4f}, Std: {surprise_matrix_seq.std():.4f} ---")

    with torch.no_grad():
        target_indices = torch.argmin(surprise_matrix_seq, dim=-1)
    ce_loss = F.cross_entropy(pre_act_logits_seq, target_indices)

    log_target_dist = F.log_softmax(-surprise_matrix_seq, dim=-1)
    log_gate_dist = F.log_softmax(pre_act_logits_seq, dim=-1)
    kl_loss = F.kl_div(log_gate_dist, log_target_dist, reduction='batchmean', log_target=True)

    return config.w_ce * ce_loss + config.w_kl * kl_loss, surprise_matrix_seq

def main():
    dense_config = DenseConfig()
    dyn_nsa_config = DynNSAConfig()
    
    visualizer = Visualizer()

    dense_model = DenseModel(dense_config).to(DEVICE, dtype=DTYPE)
    dyn_nsa_model = DynNSAv2Model(dyn_nsa_config).to(DEVICE, dtype=DTYPE)
    
    dense_optimizer = AdamW(dense_model.parameters(), lr=dense_config.learning_rate)
    dyn_nsa_optimizer = AdamW(dyn_nsa_model.parameters(), lr=dyn_nsa_config.learning_rate)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", cache_dir="./weights", trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    for config in [dense_config, dyn_nsa_config]: config.vocab_size = len(tokenizer)
    
    dense_model.embedding = nn.Embedding(dense_config.vocab_size, dense_config.hidden_size).to(DEVICE, dtype=DTYPE)
    dense_model.lm_head = nn.Linear(dense_config.hidden_size, dense_config.vocab_size, bias=False).to(DEVICE, dtype=DTYPE)
    dyn_nsa_model.embedding = nn.Embedding(dyn_nsa_config.vocab_size, dyn_nsa_config.hidden_size).to(DEVICE, dtype=DTYPE)
    dyn_nsa_model.lm_head = nn.Linear(dyn_nsa_config.hidden_size, dyn_nsa_config.vocab_size, bias=False).to(DEVICE, dtype=DTYPE)

    with open("data/dummy_chat_data.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    dataset = ChatDataset(data, tokenizer, dense_config.max_seq_len)
    data_loader = DataLoader(dataset, batch_size=dyn_nsa_config.gradient_accumulation_steps)

    history = {k: [] for k in ['dense_loss', 'dense_acc', 'dyn_nsa_loss', 'dyn_nsa_acc', 'gating_loss', 'gating_acc', 'avg_heads']}
    
    global_step = 0
    for epoch in range(dense_config.epochs):
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{dense_config.epochs}")
        epoch_metrics = {k: 0.0 for k in history.keys()}
        
        for input_ids, labels in pbar:
            input_ids, labels = input_ids.to(DEVICE), labels.to(DEVICE)
            shift_labels = labels[..., 1:].contiguous().view(-1)
            
            # --- Dense Model ---
            dense_optimizer.zero_grad()
            dense_logits, _ = dense_model(input_ids)
            shift_logits = dense_logits[..., :-1, :].contiguous()
            dense_loss = F.cross_entropy(shift_logits.view(-1, dense_config.vocab_size), shift_labels, ignore_index=-100)
            if not torch.isnan(dense_loss):
                dense_loss.backward()
                dense_optimizer.step()
                epoch_metrics['dense_loss'] += dense_loss.item()
                with torch.no_grad():
                    epoch_metrics['dense_acc'] += (shift_logits.argmax(dim=-1).view(-1) == shift_labels).float().mean().item()

            # --- DynNSA Model ---
            dyn_nsa_optimizer.zero_grad()
            nsa_logits, all_head_outputs, pre_act_logits_seq, act_mask_seq = dyn_nsa_model(input_ids)
            shift_logits_nsa = nsa_logits[..., :-1, :].contiguous()
            main_loss = F.cross_entropy(shift_logits_nsa.view(-1, dyn_nsa_config.vocab_size), shift_labels, ignore_index=-100)
            
            if not torch.isnan(main_loss):
                gating_loss, surprise_seq = get_hybrid_gating_loss(main_loss, all_head_outputs, pre_act_logits_seq, dyn_nsa_config, global_step)
                total_loss = main_loss + dyn_nsa_config.w_aux * gating_loss
                total_loss.backward()
                dyn_nsa_optimizer.step()

                epoch_metrics['dyn_nsa_loss'] += main_loss.item()
                epoch_metrics['gating_loss'] += gating_loss.item()
                with torch.no_grad():
                    epoch_metrics['dyn_nsa_acc'] += (shift_logits_nsa.argmax(dim=-1).view(-1) == shift_labels).float().mean().item()
                    epoch_metrics['gating_acc'] += (pre_act_logits_seq.argmax(dim=-1) == surprise_seq.argmin(dim=-1)).float().mean().item()
                    epoch_metrics['avg_heads'] += act_mask_seq.sum(dim=1).float().mean().item()
            
            global_step += 1
            pbar.set_postfix({
                "G_Loss": f"{epoch_metrics['gating_loss'] / (pbar.n + 1):.3f}",
                "G_Acc": f"{epoch_metrics['gating_acc'] / (pbar.n + 1):.2f}",
                "Avg_H": f"{epoch_metrics['avg_heads'] / (pbar.n + 1):.2f}"
            })

        num_batches = len(data_loader)
        for k in history.keys():
            history[k].append(epoch_metrics[k] / num_batches if num_batches > 0 else 0)
        
        visualizer.update_plot(history)
        print(f"Epoch Summary: Dense L/A: {history['dense_loss'][-1]:.3f}/{history['dense_acc'][-1]:.2f} | DynNSA L/A: {history['dyn_nsa_loss'][-1]:.3f}/{history['dyn_nsa_acc'][-1]:.2f} | Gate L/A: {history['gating_loss'][-1]:.3f}/{history['gating_acc'][-1]:.2f} | Avg Heads: {history['avg_heads'][-1]:.2f}")

if __name__ == "__main__":
    main()