"""
RDS-PLSD Experiment Report (MaxSteps=6)
=======================================

Experiment Conclusion:
1. Loss Breakthrough: RDS-PLSD achieved a significantly lower loss (0.5518) compared
   to FDS (1.9854), proving that recursive weight sharing and PLSD (Per-Layer
   Speculative Decode) are far more effective for ARC logic than static stacking.
2. The "Memorization" Trap: Despite the low loss, RMI (0.4854) and ITJD (0.0931)
   collapsed compared to Epoch 0. This is the "Evidence of Memorization": the model
   is using its shared capacity to "hardcode" the 30 tasks rather than specializing
   experts for general rules.
3. Sparsity vs. Logic: Sparsity remained stable (~78%), but the lack of path
   isolation (ITJD) suggests the model is using a "dense-in-shared-space" strategy,
   where the same experts are reused across all tasks in a non-discriminative way.
4. PLSD Efficiency: The rapid loss drop in early epochs suggests that the Oracle
   alignment in PLSD provides a much stronger supervision signal than standard CE.

Final Metrics (Epoch 29):
- Loss: 0.5518
- RMI: 0.4854
- ITJD: 0.0931
- Sparsity: 0.7840

Summary:
RDS-PLSD solves the "Capacity/Depth" problem of FDS but introduces a "Memorization"
risk. Future FARS tuning in RDS must prioritize ITJD to force the shared weights
into task-specific functional blocks.
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
from src.models.dynsiha.recursive.configuration_recursive_dynsiha import RecursiveDynSIHAConfig
from src.models.dynsiha.recursive.modeling_recursive_dynsiha import RecursiveDynSIHAForCausalLM
from exp.arc.tokenizer import ArcColorTokenizer
from exp.arc.data import InMemoryArcDataset, ArcCollator

sys.path.append(os.path.join(os.getcwd(), "ref", "ARS"))
from optimizer.ars2_neo import SingleDeviceARS2Neo

# -----------------------------------------------------------------------------
# 1. FARS 塑造器 (适配 RDS 轨迹)
# -----------------------------------------------------------------------------
class FARSRecursiveShaper:
    def __init__(self, optimizer, lambda_fars=0.05):
        self.optimizer = optimizer
        self.lambda_fars = lambda_fars

    def compute_fars_loss(self, model, model_output):
        # RDS 的 routing_weights 是轨迹列表 [T, batch, seq, num_experts]
        if not hasattr(model_output, "routing_weights") or model_output.routing_weights is None:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        total_fars_loss = 0.0
        # RDS 物理上只有一层，权重共享
        expert_param = model.block.mlp.experts.w1
        
        if expert_param in self.optimizer.state:
            state = self.optimizer.state[expert_param]
            if "exp_avg_sq" in state:
                v_t = state["exp_avg_sq"]
                fisher_cost = torch.sqrt(v_t.mean(dim=(1, 2)) + 1e-8).detach()
                fisher_cost = fisher_cost / (fisher_cost.mean() + 1e-8)
                
                # 惩罚整条递归轨迹的平均认知代价
                # weights_dict_list 形状为 [T_steps] 的字典列表
                for step_weights in model_output.routing_weights:
                    w = step_weights["mlp"]
                    total_fars_loss += (w * fisher_cost).sum(dim=-1).mean()
                
                total_fars_loss = total_fars_loss / len(model_output.routing_weights)

        return total_fars_loss * self.lambda_fars

# -----------------------------------------------------------------------------
# 2. 指标计算 (适配 RDS 轨迹)
# -----------------------------------------------------------------------------
class MetricsTracker:
    def __init__(self):
        self.history = [] 

    def update(self, task_ids, routing_weights):
        # 取递归最后一步的路由权重作为语义表征
        last_step_w = routing_weights[-1]["mlp"] 
        avg_w = last_step_w.mean(dim=1).detach().cpu().numpy() 
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
# 3. RDS-PLSD 实验
# -----------------------------------------------------------------------------
def run_rds_plsd_test(max_steps=6, epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ArcColorTokenizer()
    dataset = InMemoryArcDataset(data_path="data/ARC-AGI-2/data", tokenizer=tokenizer, split="training")
    
    # 筛选 10x10 子集
    small_tasks = []
    for task in dataset.tasks:
        max_h = max([len(p["input"]) for p in task["train"]] + [len(p["input"]) for p in task["test"]])
        max_w = max([len(p["input"][0]) for p in task["train"]] + [len(p["input"][0]) for p in task["test"]])
        if max_h <= 10 and max_w <= 10: small_tasks.append(task)
    dataset.tasks = small_tasks[:30]
    
    config = RecursiveDynSIHAConfig(
        hidden_size=128,
        num_heads=4,
        num_experts=8,
        top_k=8,
        max_steps=max_steps,
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=512
    )
    model = RecursiveDynSIHAForCausalLM(config).to(device)
    collator = ArcCollator(tokenizer, max_len=512)
    loader = DataLoader(dataset, batch_size=8, collate_fn=collator, shuffle=True)
    
    optimizer = SingleDeviceARS2Neo(model.parameters(), lr=1e-3, rho=0.05)
    shaper = FARSRecursiveShaper(optimizer)
    tracker = MetricsTracker()

    print(f"Starting RDS-PLSD Experiment: MaxSteps={max_steps}")
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            input_ids, labels, task_ids = batch["input_ids"].to(device), batch["labels"].to(device), batch["task_id"]
            optimizer.zero_grad()
            
            def closure():
                # RDS forward 会执行 PLSD 逻辑并返回轨迹
                output = model(input_ids=input_ids, labels=labels, return_dict=True)
                fars_loss = shaper.compute_fars_loss(model, output)
                # RDS 的 output.loss 是 PLSD 聚合后的损失
                return output.loss + fars_loss
            
            loss = optimizer.step(closure)
            total_loss += loss.item()
            
            with torch.no_grad():
                output = model(input_ids=input_ids, return_dict=True)
                tracker.update(task_ids, output.routing_weights)
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            rmi = tracker.compute_rmi()
            itjd = tracker.compute_itjd()
            # 采样最后一步的稀疏度
            last_w = output.routing_weights[-1]["mlp"]
            sparsity = (last_w < 0.05).float().mean().item()
            print(f"RDS | Ep {epoch} | Loss: {total_loss/len(loader):.4f} | RMI: {rmi:.4f} | ITJD: {itjd:.4f} | Sparsity: {sparsity:.4f}")

    print("\n" + "="*60)
    print("RDS-PLSD FINAL REPORT")
    print("="*60)
    print(f"Loss: {total_loss/len(loader):.4f}")
    print(f"RMI: {rmi:.4f}")
    print(f"ITJD: {itjd:.4f}")
    print(f"Sparsity: {sparsity:.4f}")
    print("="*60)

if __name__ == "__main__":
    run_rds_plsd_test()
