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
from Dense_model import DenseModel, Config, DEVICE, DTYPE
from DynNSA_model import DynNSAModel

class Visualizer:
    def __init__(self, output_dir="output/dyn_nsa_poc"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def update_plot(self, history: Dict[str, List[float]]):
        epochs = range(1, len(history['sparse_loss']) + 1)
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        axs[0, 0].plot(epochs, history['sparse_loss'], 'o-', label='Sparse Loss')
        axs[0, 0].plot(epochs, history['dense_loss'], 'o-', label='Dense Loss')
        axs[0, 0].set_title('Loss per Epoch')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        axs[0, 1].plot(epochs, history['sparse_acc'], 'o-', label='Sparse Accuracy')
        axs[0, 1].plot(epochs, history['dense_acc'], 'o-', label='Dense Accuracy')
        axs[0, 1].set_title('Accuracy per Epoch')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        axs[1, 0].plot(epochs, history['sparse_pi'], 'o-', label='Sparse PI Score')
        axs[1, 0].plot(epochs, history['dense_pi'], 'o-', label='Dense PI Score')
        axs[1, 0].set_title('Predictive Integrity (PI) Score')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        
        ax2 = axs[1, 1].twinx()
        axs[1, 1].plot(epochs, history['sparse_surprise'], 'o-r', label='Sparse Surprise')
        axs[1, 1].plot(epochs, history['dense_surprise'], 'o--r', label='Dense Surprise')
        ax2.plot(epochs, history['sparse_tau'], 'o-b', label='Sparse Tau')
        ax2.plot(epochs, history['dense_tau'], 'o--b', label='Dense Tau')
        axs[1, 1].set_title('Surprise and Tau')
        axs[1, 1].set_ylabel('Surprise (L2 Norm)', color='r')
        ax2.set_ylabel('Tau (Entropy)', color='b')
        axs[1, 1].legend(loc='upper left')
        ax2.legend(loc='upper right')

        fig.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metrics_latest.png'))
        plt.close(fig)

def main():
    config = Config()
    visualizer = Visualizer()
    sparse_model = DynNSAModel(config).to(DEVICE, dtype=DTYPE)
    dense_model = DenseModel(config).to(DEVICE, dtype=DTYPE)
    
    dense_model.embedding = sparse_model.embedding
    dense_model.lm_head = sparse_model.lm_head
    dense_model.ffn = sparse_model.ffn
    dense_model.ln1 = sparse_model.ln1
    dense_model.ln2 = sparse_model.ln2

    sparse_optimizer = AdamW(sparse_model.parameters(), lr=config.learning_rate)
    dense_optimizer = AdamW(dense_model.parameters(), lr=config.learning_rate)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", cache_dir="../../weights", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    config.vocab_size = len(tokenizer)
    
    sparse_model.embedding = nn.Embedding(config.vocab_size, config.hidden_size).to(DEVICE, dtype=DTYPE)
    sparse_model.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(DEVICE, dtype=DTYPE)
    
    dense_model.embedding = sparse_model.embedding
    dense_model.lm_head = sparse_model.lm_head


    with open("../../data/dummy_chat_data.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    history = {k: [] for k in ['sparse_loss', 'dense_loss', 'sparse_acc', 'dense_acc', 'sparse_surprise', 'dense_surprise', 'sparse_tau', 'dense_tau', 'sparse_pi', 'dense_pi']}
        
    for epoch in range(config.epochs):
        pbar = tqdm(data, desc=f"Epoch {epoch+1}/{config.epochs}")
        epoch_metrics = {k: 0.0 for k in history.keys()}
        epoch_metrics['avg_k'] = 0.0

        for item in pbar:
            prompt = f"user: {item['messages'][0]['content']}\nassistant: "
            full_text = f"{prompt}{item['messages'][1]['content']}{tokenizer.eos_token}"
            inputs = tokenizer(full_text, return_tensors="pt", max_length=config.max_seq_len, truncation=True, padding="max_length")
            
            input_ids = inputs.input_ids.to(DEVICE)
            labels = input_ids.clone()
            labels[:, :len(tokenizer.encode(prompt))] = -100
            labels[labels == tokenizer.pad_token_id] = -100
            shift_labels = labels[..., 1:].contiguous()

            def process_model(model, optimizer):
                optimizer.zero_grad()
                outputs = model(input_ids)
                
                is_sparse = isinstance(model, DynNSAModel)
                sparsity_loss_val = 0.0
                if is_sparse:
                    logits, avg_k, sparsity_loss = outputs
                    sparsity_loss_val = sparsity_loss.item()
                else:
                    logits, avg_k = outputs
                    sparsity_loss = torch.tensor(0.0, device=DEVICE)

                shift_logits = logits[..., :-1, :].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1), ignore_index=-100)
                
                if torch.isnan(loss):
                    return (None,) * 7

                total_loss = loss + config.w_aux * sparsity_loss
                total_loss.backward()
                
                surprise = torch.linalg.norm(torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]))
                optimizer.step()
                
                acc = (shift_logits.argmax(dim=-1) == shift_labels).float().mean()
                tau = torch.distributions.Categorical(logits=shift_logits.detach()).entropy().mean()
                pi_score = torch.exp(- (loss.detach() / (tau + 1e-9) + 0.1 * surprise))

                return loss.item(), acc.item(), surprise.item(), tau.item(), pi_score.item(), avg_k.item(), sparsity_loss_val

            s_loss, s_acc, s_surprise, s_tau, s_pi, s_avg_k, s_aux_loss = process_model(sparse_model, sparse_optimizer)
            d_loss, d_acc, d_surprise, d_tau, d_pi, _, __ = process_model(dense_model, dense_optimizer)

            if s_loss is not None:
                metrics_to_update = {'sparse_loss': s_loss, 'sparse_acc': s_acc, 'sparse_surprise': s_surprise, 'sparse_tau': s_tau, 'sparse_pi': s_pi, 'avg_k': s_avg_k}
                for k,v in metrics_to_update.items(): 
                    if isinstance(v, torch.Tensor): v = v.item()
                    epoch_metrics[k] += v

            if d_loss is not None:
                metrics_to_update = {'dense_loss': d_loss, 'dense_acc': d_acc, 'dense_surprise': d_surprise, 'dense_tau': d_tau, 'dense_pi': d_pi}
                for k,v in metrics_to_update.items():
                    if isinstance(v, torch.Tensor): v = v.item()
                    epoch_metrics[k] += v

            pbar.set_postfix({"S_Loss": f"{s_loss:.2f}", "D_Loss": f"{d_loss:.2f}", "S_Aux": f"{s_aux_loss:.3f}", "Avg K": f"{s_avg_k:.2f}"})
        
        num_items = len(data)
        for k in history.keys():
            history[k].append(epoch_metrics[k] / num_items)
        
        visualizer.update_plot(history)
        print(f"Epoch Summary: Sparse L/A: {history['sparse_loss'][-1]:.3f}/{history['sparse_acc'][-1]:.2f} | Dense L/A: {history['dense_loss'][-1]:.3f}/{history['dense_acc'][-1]:.2f} | Avg K: {(epoch_metrics['avg_k'] / num_items):.2f}")

if __name__ == "__main__":
    main()