import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from rich.progress import track
import numpy as np
from collections import Counter

# -----------------------------------------------------------------------------
# 0. 环境配置与依赖注入
# -----------------------------------------------------------------------------
sys.path.append(os.getcwd())

from src.models.dynsiha.flat.configuration_flat_dynsiha import FlatDynSIHAConfig
from src.models.dynsiha.flat.modeling_flat_dynsiha import FlatDynSIHAForCausalLM
from exp.arc.tokenizer import ArcColorTokenizer
from exp.arc.data import InMemoryArcDataset, ArcCollator

sys.path.append(os.path.join(os.getcwd(), "ref", "ARS"))
try:
    from optimizer.ars2_neo import SingleDeviceARS2Neo
except ImportError:
    print("Error: Could not import SingleDeviceARS2Neo. Please ensure ref/ARS exists.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# 1. 生产级 FARS 塑造器 (RoutingShaper)
# -----------------------------------------------------------------------------

class RoutingShaper:
    def __init__(self, optimizer, lambda_fars=0.01):
        self.optimizer = optimizer
        self.lambda_fars = lambda_fars

    def compute_fars_loss(self, model_output):
        if not hasattr(model_output, "routing_weights") or model_output.routing_weights is None:
            return torch.tensor(0.0, device=model_output.logits.device)

        total_fars_loss = 0.0
        for layer_idx, weights_dict in enumerate(model_output.routing_weights):
            for key in ["q", "k", "v", "mlp"]:
                w = weights_dict[key]
                num_experts = w.shape[-1]
                
                # 模拟 Fisher Cost：基础税率场
                costs = torch.linspace(0.1, 1.0, num_experts, device=w.device)
                
                # 路径一致性奖励：惩罚熵，迫使模型产生确定的选择
                # 增加 Polarization 奖励：1 - sum(w^2)，当 w 趋向 one-hot 时该值最小
                polarization = 1.0 - (w**2).sum(dim=-1).mean()
                entropy = -(w * torch.log(w + 1e-8)).sum(dim=-1).mean()
                
                # 𝒢 = Belief * Cost + λ_entropy * Entropy + λ_pol * Polarization
                # 降低基础税率影响，强化极化约束
                layer_fars = (w * costs).sum(dim=-1).mean() * 0.1 + 0.5 * entropy + 1.0 * polarization
                total_fars_loss += layer_fars

        return total_fars_loss * self.lambda_fars

# -----------------------------------------------------------------------------
# 2. 指标计算：RMI & ITJD
# -----------------------------------------------------------------------------

class MetricsTracker:
    def __init__(self, num_experts):
        self.num_experts = num_experts
        self.history = [] 
        self.task_names = []

    def update(self, task_ids, routing_weights):
        # 使用 MLP 路由权重，形状为 [batch, seq, num_experts]
        last_mlp_w = routing_weights[-1]["mlp"]
        # 对序列维度取平均，得到每个样本的专家激活分布
        avg_w = last_mlp_w.mean(dim=1).detach().cpu().numpy()
        
        for tid, w in zip(task_ids, avg_w):
            self.history.append((tid, w))
            if tid not in self.task_names:
                self.task_names.append(tid)

    def compute_rmi(self, threshold=0.5):
        if not self.history: return 0.0
        total = len(self.history)
        task_counts = Counter([h[0] for h in self.history])
        states = []
        for name, w in self.history:
            mask = (w > threshold).astype(int)
            state_int = sum(m * (2**i) for i, m in enumerate(mask))
            states.append((name, state_int))
        state_counts = Counter([s[1] for s in states])
        joint_counts = Counter(states)
        h_r = -sum((c/total) * np.log2(c/total + 1e-10) for c in state_counts.values())
        h_r_t = 0.0
        for t, t_c in task_counts.items():
            p_t = t_c / total
            sub_h = 0.0
            t_states = [s[1] for s in states if s[0] == t]
            t_state_counts = Counter(t_states)
            for s_c in t_state_counts.values():
                p_s_t = s_c / t_c
                sub_h -= p_s_t * np.log2(p_s_t + 1e-10)
            h_r_t += p_t * sub_h
        return h_r - h_r_t

    def compute_itjd(self):
        if not self.history: return 0.0
        task_avg = {}
        for name in self.task_names:
            ws = [h[1] for h in self.history if h[0] == name]
            task_avg[name] = np.mean(ws, axis=0)
        dists = []
        names = list(task_avg.keys())
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                a, b = task_avg[names[i]], task_avg[names[j]]
                inter = np.minimum(a, b).sum()
                union = np.maximum(a, b).sum()
                dists.append(1.0 - (inter / (union + 1e-8)))
        return np.mean(dists) if dists else 0.0

# -----------------------------------------------------------------------------
# 3. 实验主程序
# -----------------------------------------------------------------------------

def run_mini_arc():
    console = Console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = FlatDynSIHAConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_heads=4,
        num_experts=8,
        top_k=8, 
        vocab_size=1000,
        max_position_embeddings=512
    )
    
    model = FlatDynSIHAForCausalLM(config).to(device)
    tokenizer = ArcColorTokenizer()
    dataset = InMemoryArcDataset(data_path="data/ARC-AGI-2/data", tokenizer=tokenizer, split="training")
    dataset.tasks = dataset.tasks[:20]
    collator = ArcCollator(tokenizer, max_len=512)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collator, shuffle=True)
    
    # 降低学习率，增加 rho 强化平坦度约束，迫使路由器寻找更稳健的路径
    optimizer = SingleDeviceARS2Neo(model.parameters(), lr=1e-3, rho=0.05)
    # 动态 λ_fars：初始极低，随 epoch 增加
    shaper = RoutingShaper(optimizer, lambda_fars=0.001)
    tracker = MetricsTracker(config.num_experts)
    
    console.print(f"[bold green]Starting Mini-ARC FARS v3 Experiment on {device}...[/bold green]")
    
    for epoch in range(10):
        for batch_idx, batch in enumerate(track(loader, description=f"Epoch {epoch}")):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            # 关键修复：从 batch 中提取真实的 task_id
            task_ids = batch["task_id"]
            optimizer.zero_grad()
            def closure():
                output = model(input_ids=input_ids, labels=labels, return_dict=True)
                main_loss = output.loss
                # 动态调整惩罚强度
                shaper.lambda_fars = 0.001 * (1 + epoch)
                fars_loss = shaper.compute_fars_loss(output)
                total_loss = main_loss + fars_loss
                return total_loss
            loss = optimizer.step(closure)
            with torch.no_grad():
                output = model(input_ids=input_ids, return_dict=True)
                tracker.update(task_ids, output.routing_weights)
                sparsity = (output.routing_weights[-1]["mlp"] < 0.05).float().mean().item()
        rmi = tracker.compute_rmi()
        itjd = tracker.compute_itjd()
        console.print(f"Epoch {epoch} | Loss: {loss:.4f} | Sparsity: {sparsity:.4f} | RMI: {rmi:.4f} | ITJD: {itjd:.4f}")

    table = Table(title="Mini-ARC FARS v3 Final Report")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("RMI (Routing Mutual Info)", f"{tracker.compute_rmi():.4f}")
    table.add_row("ITJD (Path Isolation)", f"{tracker.compute_itjd():.4f}")
    table.add_row("Final Sparsity", f"{sparsity:.4f}")
    console.print(table)

if __name__ == "__main__":
    run_mini_arc()
