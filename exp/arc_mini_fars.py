import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.functional as F
import json
import time
import random
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from rich.console import Console
from src.models.dynsiha.recursive.modeling_recursive_dynsiha import RecursiveDynSIHAForCausalLM
from src.models.dynsiha.recursive.configuration_recursive_dynsiha import RecursiveDynSIHAConfig
from src.optimizers.ars2_neo import SingleDeviceARS2Neo
from exp.arc.tokenizer import ArcColorTokenizer
from exp.arc.data import GridSerializer, ArcCollator, GridDeserializer

console = Console()

class MiniArcDataset(Dataset):
    def __init__(self, data_path: str, max_size: int = 10, split: str = "train", max_tasks: int = None):
        self.data_path = Path(data_path)
        self.tasks = []
        all_tasks = []
        
        for path in sorted(self.data_path.glob("*.json")):
            with open(path) as f:
                task = json.load(f)
            all_grids = []
            for pair in task['train'] + task['test']:
                all_grids.append(pair['input'])
                if 'output' in pair:
                    all_grids.append(pair['output'])
            if all(len(g) <= max_size and len(g[0]) <= max_size for g in all_grids if g):
                t_in = task['test'][0]['input']
                t_out = task['test'][0]['output']
                h1, w1 = len(t_in), len(t_in[0])
                h2, w2 = len(t_out), len(t_out[0])
                num_diff = h2 * w2 if (h1 != h2 or w1 != w2) else sum(t_in[i][j] != t_out[i][j] for i in range(h1) for j in range(w1))
                task['adl_info'] = {"num_diff": num_diff, "task_id": path.stem}
                all_tasks.append(task)
        
        # 只取10x10任务中的半数作为训练集，剩下一半中的15条作为验证集
        if max_tasks is None:
            max_tasks = len(all_tasks)
        
        if split == "train":
            # 取前半部分作为训练集
            self.tasks = all_tasks[:len(all_tasks)//2][:max_tasks]
        elif split == "val":
            # 取后半部分中的前15条作为验证集
            val_start = len(all_tasks)//2
            self.tasks = all_tasks[val_start:val_start+15]
        else:
            self.tasks = all_tasks[:max_tasks]
            
        console.print(f"[cyan]Loaded {len(self.tasks)} {split} tasks (≤{max_size}×{max_size})[/cyan]")

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]

def evaluate_mini_arc(model, dataset, tokenizer, device, num_samples=3):
    model.eval()
    serializer, deserializer = GridSerializer(tokenizer), GridDeserializer(tokenizer)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    correct = 0
    console.print("\n[bold magenta]=== RDS-ACT DEEP INSPECTION ===[/bold magenta]")
    with torch.no_grad():
        for idx in indices:
            task = dataset[idx]
            prompt_ids, _ = serializer.serialize_for_inference(task)
            input_tensor = torch.tensor([prompt_ids], device=device)
            outputs = model(input_tensor, return_dict=True)
            prompt_steps = outputs.exit_steps.item() if hasattr(outputs, 'exit_steps') else 0
            output_ids = model.generate(
                input_tensor,
                max_new_tokens=64,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                use_cache=True
            )
            gen_tokens = output_ids[0, input_tensor.shape[1]:].tolist()
            true_output = task['test'][0]['output']
            true_tokens, _ = serializer._serialize_grid(true_output)
            console.print(f"\n[Task {task['adl_info']['task_id']}] Prompt Steps: {prompt_steps}")
            console.print(f"Target: {true_tokens}")
            console.print(f"Gen:    {gen_tokens}")
            try:
                pred_grid = deserializer.deserialize(gen_tokens)
                true_grid = torch.tensor(true_output, device='cpu')
                if pred_grid.shape == true_grid.shape and torch.all(pred_grid == true_grid):
                    correct += 1
                    console.print("Result: [bold green]PASS ✨[/bold green]")
                else:
                    unique = torch.unique(pred_grid).tolist()
                    console.print(f"Result: [bold red]FAIL[/bold red] | Colors: {unique}")
            except Exception as e:
                console.print(f"Result: [bold red]ERROR {e}[/bold red]")
    accuracy = correct / len(indices)
    console.print(f"\n[bold]Eval EM Accuracy: {accuracy:.2%}[/bold]")
    model.train()
    return accuracy

def calculate_itjd_rmi(history):
    if len(history) < 2: return 0.0, 0.0
    task_ids = [h[0] for h in history]
    probs = torch.stack([h[1] for h in history])
    unique = list(set(task_ids))
    if len(unique) < 2: return 0.0, 0.0
    centroids = []
    for tid in unique:
        mask = torch.tensor([t == tid for t in task_ids], device=probs.device)
        centroids.append(probs[mask].mean(dim=0))
    centroids = F.normalize(torch.stack(centroids), p=2, dim=-1)
    sim = torch.mm(centroids, centroids.t())
    itjd = 1.0 - (sim.sum() - len(unique)) / (len(unique) * (len(unique) - 1))
    p_e = probs.mean(dim=0)
    h_e = -torch.sum(p_e * torch.log(p_e + 1e-10))
    h_e_t = 0
    for tid in unique:
        mask = torch.tensor([t == tid for t in task_ids], device=probs.device)
        p_e_t = probs[mask].mean(dim=0)
        h_e_t += -torch.sum(p_e_t * torch.log(p_e_t + 1e-10)) * (mask.sum().item() / len(task_ids))
    rmi = h_e - h_e_t
    return itjd.item(), rmi.item()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ArcColorTokenizer()
    train_dataset = MiniArcDataset("data/ARC-AGI/data/training", max_size=10, split="train")
    val_dataset = MiniArcDataset("data/ARC-AGI/data/training", max_size=10, split="val")
    collator = ArcCollator(tokenizer, max_len=1024)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collator)
    config = RecursiveDynSIHAConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        max_refinement_steps=8,
        max_inference_steps=16,
        num_hidden_layers=16,
        num_heads=8,
        num_experts=32,
        top_k=32,
        max_position_embeddings=1024
    )
    model = RecursiveDynSIHAForCausalLM(config).to(device)
    model.gradient_checkpointing_enable()
    
    param_groups = [
        {'params': [p for p in model.parameters() if p.ndim >= 2], 'is_rmsuon_group': True},
        {'params': [p for p in model.parameters() if p.ndim < 2], 'is_rmsuon_group': False}
    ]
    optimizer = SingleDeviceARS2Neo(param_groups, lr=5e-3, rho=0.1)
    routing_history = []
    
    console.print("[bold green]Mini-RDS Training[/bold green] | 32 Experts × 8 Heads | 8→16 Steps")
    console.print("step | task_loss=main_loss+adl_loss | adl_loss | FARS_loss | ITJD | RMI | Eff_K | Eff_L")
    step = 0
    while step < 1000:
        for batch in train_loader:
            if not batch or step >= 1000: break
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            diff_mask = batch['diff_mask'].to(device)
            task_data = batch['task_data']
            metrics = {}
            
            def closure():
                outputs = model(input_ids=input_ids, labels=labels, return_dict=True)
                
                # 1. ADL (Adaptive Differential Loss) - Token Level Masking
                main_loss = outputs.loss
                
                # Calculate diff_loss: cross entropy only on diff tokens
                logits = outputs.logits
                shifted_logits = logits[..., :-1, :].contiguous()
                shifted_labels = labels[..., 1:].contiguous()
                shifted_diff_mask = diff_mask[..., 1:].contiguous()
                
                if shifted_diff_mask.any():
                    # reduction='none' to apply mask
                    raw_loss = F.cross_entropy(
                        shifted_logits.view(-1, config.vocab_size),
                        shifted_labels.view(-1),
                        reduction='none'
                    ).view(shifted_labels.shape)
                    
                    diff_loss = (raw_loss * shifted_diff_mask).sum() / (shifted_diff_mask.sum() + 1e-8)
                    
                    # ADL: Dynamic weighting based on diff density
                    ans_tokens = (labels != -100).sum().item()
                    diff_tokens = shifted_diff_mask.sum().item()
                    adl_coeff = diff_tokens / (ans_tokens + 1e-8)
                    adl_loss = diff_loss * adl_coeff
                else:
                    adl_loss = torch.tensor(0.0, device=device)
                
                # 2. FARS (Fisher-Aware Routing Shaping) - Path Aligned
                all_routing = getattr(outputs, 'routing_info', [])
                best_step_mask = getattr(outputs, 'best_step_mask', None) # [steps, B]
                fars_list, effk_list = [], []
                
                if all_routing and best_step_mask is not None:
                    for step_idx, step_routing in enumerate(all_routing):
                        # step_mask: [B]
                        step_mask = best_step_mask[step_idx]
                        if not step_mask.any(): continue
                        
                        for key, logits in step_routing.items():
                            # 对齐模型内部的贝叶斯竞争逻辑 (Softmax)
                            probs = F.softmax(logits, dim=-1)
                            
                            # 过滤活跃样本
                            active_probs = probs[step_mask] # [num_active, T, num_experts]
                            if active_probs.numel() == 0: continue
                            
                            entropy = -torch.sum(active_probs * torch.log(active_probs + 1e-10), dim=-1).mean()
                            effk_list.append(torch.exp(entropy))
                            
                            target = None
                            if "mlp" in key: target = model.block.mlp.experts.w1
                            elif "q" in key: target = model.block.attn.q_experts.w1
                            elif "k" in key: target = model.block.attn.k_experts.w1
                            elif "v" in key: target = model.block.attn.v_experts.w1
                            
                            if target is not None and target in optimizer.state:
                                v_t = optimizer.state[target].get('exp_avg_sq', None)
                                if v_t is not None:
                                    # Fisher 信息近似：二阶矩的平方根
                                    costs = torch.sqrt(v_t).mean(dim=(1, 2)) # [num_experts]
                                    # 归一化修：使用全局平均值归一化，保持绝对代价量纲
                                    costs = costs / (costs.mean() + 1e-8)
                                    
                                    # 路由概率与代价的对偶性
                                    p_flat = active_probs.view(-1, config.num_experts)
                                    fars_list.append((p_flat * costs).sum(dim=-1).mean())
                                    
                                    if "mlp" in key and step % 10 == 0:
                                        for b_idx, is_active in enumerate(step_mask):
                                            if is_active:
                                                routing_history.append((task_data[b_idx]['adl_info']['task_id'], probs[b_idx].mean(dim=0).detach()))
                
                fars_loss = torch.stack(fars_list).mean() if fars_list else torch.tensor(0.0, device=device)
                eff_k = torch.stack(effk_list).mean() if effk_list else torch.tensor(0.0)
                eff_l = outputs.exit_steps.float().mean() if hasattr(outputs, 'exit_steps') else torch.tensor(0.0)
                
                total_loss = main_loss + adl_loss + 0.1 * fars_loss
                metrics.update({
                    "main": main_loss.item(),
                    "adl": adl_loss.item(),
                    "fars": fars_loss.item(),
                    "effk": eff_k.item(),
                    "effl": eff_l.item()
                })
                
                return total_loss
            
            optimizer.zero_grad()
            loss_val = optimizer.step(closure)
            if step % 10 == 0:
                itjd, rmi = calculate_itjd_rmi(routing_history[-200:])
                task_loss = metrics['main'] + metrics['adl']
                console.print(f"{step:4d} | {task_loss:.4f}={metrics['main']:.4f}+{metrics['adl']:.4f} | {metrics['adl']:.4f} | {metrics['fars']:.4f} | {itjd:.3f} | {rmi:.3f} | {metrics['effk']:.2f} | {metrics['effl']:.1f}")
            if step % 1000 == 0 and step > 0: evaluate_mini_arc(model, val_dataset, tokenizer, device)
            step += 1
    evaluate_mini_arc(model, val_dataset, tokenizer, device, num_samples=10)
    console.print("[bold green]Training completed[/bold green]")

if __name__ == "__main__":
    train()
