import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import json
from transformers import AutoTokenizer
import os
from einops import rearrange

from .config import DynSMHAConfig, DEVICE, DTYPE
from .model import TinyOnnModel

class ChatDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: AutoTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        prompt = f"user: {item['messages'][0]['content']}\nassistant: "
        full_text = f"{prompt}{item['messages'][1]['content']}{self.tokenizer.eos_token}"
        inputs = self.tokenizer(full_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = inputs.input_ids.squeeze(0)
        labels = input_ids.clone()
        prompt_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        labels[:prompt_len] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        return input_ids, labels

def get_gating_loss(surprise: torch.Tensor, pre_act_logits: torch.Tensor, config: DynSMHAConfig) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        target_indices = torch.argmin(surprise, dim=-1)
    ce_loss = F.cross_entropy(pre_act_logits, target_indices)

    log_target_dist = F.log_softmax(-surprise, dim=-1)
    log_gate_dist = F.log_softmax(pre_act_logits, dim=-1)
    kl_loss = F.kl_div(log_gate_dist, log_target_dist, reduction='batchmean', log_target=True)

    combined_loss = config.w_ce * ce_loss + config.w_kl * kl_loss
    
    return combined_loss, ce_loss

def main():
    config = DynSMHAConfig()
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", cache_dir="./weights", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    config.vocab_size = len(tokenizer)

    model = TinyOnnModel(config).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
    dataset = ChatDataset("data/dummy_chat_data.jsonl", tokenizer, config.max_seq_len)
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    for epoch in range(config.epochs):
        for i, (input_ids, labels) in enumerate(data_loader):
            input_ids, labels = input_ids.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            final_logits, all_sha_outputs, pre_act_logits, act_mask, fallback_count, raw_affinity_score = model(input_ids)
            
            main_loss = F.cross_entropy(
                rearrange(final_logits, 'b t d -> (b t) d'),
                rearrange(labels, 'b t -> (b t)'),
                ignore_index=-100
            )
            
            if torch.isnan(main_loss):
                continue
            
            all_sha_outputs.requires_grad_(True)
            grad_matrix, = torch.autograd.grad(main_loss, all_sha_outputs, retain_graph=True, allow_unused=True)
            surprise = torch.linalg.norm(grad_matrix.to(DTYPE).float(), dim=(-1, 1)) if grad_matrix is not None else torch.zeros_like(pre_act_logits)

            # Disable gating loss for this experiment
            total_loss = main_loss
            total_loss.backward()

            optimizer.step()
            
            if (i + 1) % 5 == 0:
                with torch.no_grad():
                    main_acc = (final_logits.argmax(-1) == labels).float().mean().item()
                    
                    # Reshape to (B, T, M) and average over sequence length T
                    surprise_per_seq = surprise.view(config.batch_size, config.max_seq_len, -1).mean(dim=1)
                    affinity_per_seq = raw_affinity_score.view(config.batch_size, config.max_seq_len, -1).mean(dim=1)
                    
                    # Flatten to (B*M) for correlation calculation
                    flat_surprise = surprise_per_seq.flatten()
                    flat_affinity = affinity_per_seq.flatten()
                    
                    # We expect affinity (-entropy) to be correlated with -surprise
                    corr_matrix = torch.corrcoef(torch.stack([-flat_surprise, flat_affinity]))
                    correlation = corr_matrix[0, 1].item() if not torch.isnan(corr_matrix).any() else 0.0
                    
                    log_str = (
                        f"Epoch {epoch+1}/{config.epochs} | Step {i+1}/{len(data_loader)} | "
                        f"Loss: {main_loss.item():.3f} | Acc: {main_acc:.2%} | "
                        f"Corr: {correlation:.3f}"
                    )
                    print(log_str)

if __name__ == "__main__":
    main()