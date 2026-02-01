"""
ARC Depth Scaling Experiment Report
===================================

Experiment Conclusion:
1. Inverse Scaling Paradox: Contrary to expectations, increasing layers from 4 to 8
   resulted in HIGHER loss (1.98 -> 2.23) and DEGRADED RMI (2.03 -> 1.74).
2. Optimization Bottleneck: The deeper models (L6, L8) struggled to converge within
   20 epochs, likely due to the vanishing gradient in the router or the increased
   complexity of the Fisher cost field across multiple layers.
3. RMI Stability: The 4-layer model achieved the best semantic alignment (RMI 2.03),
   suggesting that for small ARC tasks (<=10x10), a shallower but well-differentiated
   architecture is more efficient.
4. Sparsity Consistency: All depths maintained ~85% sparsity, proving FARS is
   robust to depth scaling, even if the main task optimization becomes harder.

Final Metrics (20 Epochs):
| Layers | Loss   | RMI    | ITJD   | Sparsity |
| :----- | :----- | :----- | :----- | :------- |
| 4      | 1.9854 | 2.0327 | 0.4338 | 0.8607   |
| 6      | 2.1118 | 1.2579 | 0.2942 | 0.8393   |
| 8      | 2.2293 | 1.7418 | 0.4026 | 0.8381   |

Summary:
Depth alone is not the silver bullet for ARC loss reduction in FDS. The coupling
between depth and FARS taxing needs careful scheduling, or a transition to
Recursive DynSIHA (RDS) where depth is dynamic and shared.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
from collections import Counter

# -----------------------------------------------------------------------------
# 0. 依赖注入
# -----------------------------------------------------------------------------
sys.path.append(os.getcwd())
from src.models.dynsiha.flat.configuration_flat_dynsiha import FlatDynSIHAConfig
from src.models.dynsiha.flat.modeling_flat_dynsiha import FlatDynSIHAForCausalLM
from exp.arc.tokenizer import ArcColorTokenizer
from exp.arc.data import InMemoryArcDataset, ArcCollator

sys.path.append(os.path.join(os.getcwd(), "ref", "ARS"))
from optimizer.ars2_neo import SingleDeviceARS2Neo

# -----------------------------------------------------------------------------
# 1. FARS 塑造器
# -----------------------------------------------------------------------------
class FARSShaper:
    def __init__(self, optimizer, lambda_fars=0.05):
        self.optimizer = optimizer
        self.lambda_fars = lambda_fars

    def compute_fars_loss(self, model, model_output):
        if not hasattr(model_output, "routing_weights") or model_output.routing_weights is None:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        total_fars_loss = 0.0
        for layer_idx, weights_dict in enumerate(model_output.routing_weights):
            w = weights_dict["mlp"]
            expert_param = model.layers[layer_idx].mlp.experts.w1
            if expert_param in self.optimizer.state:
                state = self.optimizer.state[expert_param]
                if "exp_avg_sq" in state:
                    v_t = state["exp_avg_sq"]
                    fisher_cost = torch.sqrt(v_t.mean(dim=(1, 2)) + 1e-8).detach()
                    fisher_cost = fisher_cost / (fisher_cost.mean() + 1e-8)
                    total_fars_loss += (w * fisher_cost).sum(dim=-1).mean()
        return total_fars_loss * self.lambda_fars

# -----------------------------------------------------------------------------
# 2. 指标计算
# -----------------------------------------------------------------------------
class MetricsTracker:
    def __init__(self):
        self.history = [] 

    def update(self, task_ids, routing_weights):
        last_mlp_w = routing_weights[-1]["mlp"] 
        avg_w = last_mlp_w.mean(dim=1).detach().cpu().numpy() 
        for tid, w in zip(task_ids, avg_w):
            self.history.append((tid, w))

    def compute_rmi(self, threshold=0.1):
        if not self.history: return 0.0
        total = len(self.history)
        task_counts = Counter([h[0] for h in self.history])
        states = []
        for tid, w in self.history:
            mask = (w > threshold).astype(int)
            state_int = sum(m * (2**i) for i, m in enumerate(mask))
            states.append((tid, state_int))
        state_counts = Counter([s[1] for s in states])
        h_r = -sum((c/total) * np.log2(c/total + 1e-10) for c in state_counts.values())
        h_r_t = 0.0
        for t, t_c in task_counts.items():
            p_t = t_c / total
            t_states = [s[1] for s in states if s[0] == t]
            t_state_counts = Counter(t_states)
            sub_h = 0.0
            for s_c in t_state_counts.values():
                p_s_t = s_c / t_c
                sub_h -= p_s_t * np.log2(p_s_t + 1e-10)
            h_r_t += p_t * sub_h
        return h_r - h_r_t

    def compute_itjd(self):
        if not self.history: return 0.0
        task_profiles = {}
        for tid, w in self.history:
            if tid not in task_profiles: task_profiles[tid] = []
            task_profiles[tid].append(w > 0.1)
        avg_profiles = {tid: np.mean(masks, axis=0) > 0.5 for tid, masks in task_profiles.items()}
        distances = []
        tids = list(avg_profiles.keys())
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                m1, m2 = avg_profiles[tids[i]], avg_profiles[tids[j]]
                intersection = np.logical_and(m1, m2).sum()
                union = np.logical_or(m1, m2).sum()
                distances.append(1.0 - (intersection / union) if union > 0 else 0.0)
        return np.mean(distances) if distances else 0.0

# -----------------------------------------------------------------------------
# 3. 核心实验函数
# -----------------------------------------------------------------------------
def run_depth_scaling(num_layers, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ArcColorTokenizer()
    dataset = InMemoryArcDataset(data_path="data/ARC-AGI-2/data", tokenizer=tokenizer, split="training")
    
    small_tasks = []
    for task in dataset.tasks:
        max_h = max([len(p["input"]) for p in task["train"]] + [len(p["input"]) for p in task["test"]])
        max_w = max([len(p["input"][0]) for p in task["train"]] + [len(p["input"][0]) for p in task["test"]])
        if max_h <= 10 and max_w <= 10: small_tasks.append(task)
    dataset.tasks = small_tasks[:30]
    
    config = FlatDynSIHAConfig(
        hidden_size=128, num_hidden_layers=num_layers, num_heads=4,
        num_experts=8, top_k=8, vocab_size=tokenizer.vocab_size, max_position_embeddings=512
    )
    model = FlatDynSIHAForCausalLM(config).to(device)
    collator = ArcCollator(tokenizer, max_len=512)
    loader = DataLoader(dataset, batch_size=8, collate_fn=collator, shuffle=True)
    optimizer = SingleDeviceARS2Neo(model.parameters(), lr=1e-3, rho=0.05)
    shaper = FARSShaper(optimizer)
    tracker = MetricsTracker()

    print(f"Starting Scaling Experiment: Layers={num_layers}")
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            input_ids, labels, task_ids = batch["input_ids"].to(device), batch["labels"].to(device), batch["task_id"]
            optimizer.zero_grad()
            def closure():
                output = model(input_ids=input_ids, labels=labels, return_dict=True)
                fars_loss = shaper.compute_fars_loss(model, output)
                return output.loss + fars_loss
            loss = optimizer.step(closure)
            total_loss += loss.item()
            with torch.no_grad():
                output = model(input_ids=input_ids, return_dict=True)
                tracker.update(task_ids, output.routing_weights)
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            rmi = tracker.compute_rmi()
            itjd = tracker.compute_itjd()
            last_w = output.routing_weights[-1]["mlp"]
            sparsity = (last_w < 0.05).float().mean().item()
            print(f"L{num_layers} | Ep {epoch} | Loss: {total_loss/len(loader):.4f} | RMI: {rmi:.4f} | ITJD: {itjd:.4f} | Sparsity: {sparsity:.4f}")
    
    return {"loss": total_loss/len(loader), "rmi": rmi, "itjd": itjd, "sparsity": sparsity}

if __name__ == "__main__":
    results = {}
    for L in [4, 6, 8]:
        results[L] = run_depth_scaling(L)
    
    print("\n" + "="*60)
    print("ARC DEPTH SCALING FINAL REPORT")
    print("="*60)
    print(f"{'Layers':<10} | {'Loss':<10} | {'RMI':<10} | {'ITJD':<10} | {'Sparsity':<10}")
    print("-" * 60)
    for L, res in results.items():
        print(f"{L:<10} | {res['loss']:<10.4f} | {res['rmi']:<10.4f} | {res['itjd']:<10.4f} | {res['sparsity']:<10.4f}")
    print("="*60)
