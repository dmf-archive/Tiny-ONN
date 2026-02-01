import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table
from rich.progress import track
import sys
import os
import numpy as np
from collections import Counter

# -----------------------------------------------------------------------------
# 0. 依赖注入
# -----------------------------------------------------------------------------
sys.path.append(os.getcwd())
from src.models.dynsiha.shared.router import MLPRouter, VectorizedExpertMLP

sys.path.append(os.path.join(os.getcwd(), "ref", "ARS"))
try:
    from optimizer.ars2_neo import SingleDeviceARS2Neo
except ImportError:
    print("Error: Could not import SingleDeviceARS2Neo. Please ensure ref/ARS exists.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# 1. 增强型规则模拟器：Orthogonal Bit-Pattern Classification
# -----------------------------------------------------------------------------
class BitPatternDataset:
    def __init__(self, num_samples=10000, dim=16):
        self.x = torch.randint(0, 2, (num_samples, dim)).float()
        self.y = self._generate_labels(self.x)
        
    def _generate_labels(self, x):
        num_samples = x.shape[0]
        labels = torch.zeros(num_samples, dtype=torch.long)
        
        # 定义 10 个相对正交的逻辑规则
        for i in range(num_samples):
            row = x[i]
            s = row.sum().item()
            # 规则优先级排序
            if s % 2 == 0 and row[0] == 1: labels[i] = 0
            elif s % 2 != 0 and row[-1] == 1: labels[i] = 1
            elif row[:4].sum() > 2: labels[i] = 2
            elif row[4:8].sum() > 2: labels[i] = 3
            elif row[8:12].sum() > 2: labels[i] = 4
            elif (row[1:] * row[:-1]).sum() > 2: labels[i] = 5 # 连续1
            elif s > 8: labels[i] = 6
            elif row[0] == row[-1]: labels[i] = 7
            elif row[1::2].sum() > row[0::2].sum(): labels[i] = 8
            else: labels[i] = 9
        return labels

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

# -----------------------------------------------------------------------------
# 2. 生产级 FARS 塑造器 (Real Fisher Shaper)
# -----------------------------------------------------------------------------
class FARSShaper:
    def __init__(self, optimizer, lambda_fars=0.02):
        self.optimizer = optimizer
        self.lambda_fars = lambda_fars

    def compute_fars_loss(self, router_weights, expert_module):
        if expert_module.w1 not in self.optimizer.state:
            return torch.tensor(0.0, device=router_weights.device)
            
        state = self.optimizer.state[expert_module.w1]
        if "exp_avg_sq" not in state:
            return torch.tensor(0.0, device=router_weights.device)
            
        # 提取 Fisher 信息近似
        v_t = state["exp_avg_sq"]
        fisher_cost = torch.sqrt(v_t.mean(dim=(1, 2)) + 1e-8).detach()
        
        # 归一化成本场：使其均值为 1，将绝对惩罚转为相对竞争压力
        fisher_cost = fisher_cost / (fisher_cost.mean() + 1e-8)
        
        # 𝒢 = Σ (Belief * Fisher_Cost)
        fars_term = (router_weights * fisher_cost).sum(dim=-1).mean()
        
        # 极化约束：1 - sum(w^2)，鼓励 one-hot 分布
        polarization = 1.0 - (router_weights**2).sum(dim=-1).mean()
        
        # 熵约束：防止过早坍塌到单一专家
        entropy = -(router_weights * torch.log(router_weights + 1e-8)).sum(dim=-1).mean()
        
        return (fars_term + 0.5 * polarization + 0.1 * entropy) * self.lambda_fars

# -----------------------------------------------------------------------------
# 3. 指标计算：RMI & ITJD
# -----------------------------------------------------------------------------
class MetricsTracker:
    def __init__(self, num_experts):
        self.num_experts = num_experts
        self.history = [] 

    def update(self, labels, routing_weights):
        avg_w = routing_weights.detach().cpu().numpy() 
        for tid, w in zip(labels.tolist(), avg_w):
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
        
        avg_profiles = {}
        for tid, masks in task_profiles.items():
            avg_profiles[tid] = np.mean(masks, axis=0) > 0.5
            
        distances = []
        tids = list(avg_profiles.keys())
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                m1, m2 = avg_profiles[tids[i]], avg_profiles[tids[j]]
                intersection = np.logical_and(m1, m2).sum()
                union = np.logical_or(m1, m2).sum()
                if union == 0: distances.append(0.0)
                else: distances.append(1.0 - (intersection / union))
        return np.mean(distances) if distances else 0.0

# -----------------------------------------------------------------------------
# 4. 实验：FARS 逻辑验证
# -----------------------------------------------------------------------------
def run_fars_logic_test():
    console = Console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dim, num_experts, hidden_size = 16, 10, 32
    
    router = MLPRouter(dim, num_experts, top_k=num_experts).to(device)
    experts = VectorizedExpertMLP(num_experts, dim, hidden_size).to(device)
    classifier = nn.Linear(dim, 10).to(device) 
    
    dataset = BitPatternDataset(num_samples=10000)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    params = list(router.parameters()) + list(experts.parameters()) + list(classifier.parameters())
    optimizer = SingleDeviceARS2Neo(params, lr=1e-3, rho=0.05)
    shaper = FARSShaper(optimizer, lambda_fars=0.02)
    tracker = MetricsTracker(num_experts)
    
    console.print(f"[bold green]Starting FARS Logic Test (Bit-Pattern 10-Class) on {device}...[/bold green]")
    
    for epoch in range(15):
        total_loss, correct, total_samples = 0, 0, 0
        
        for x, y in track(loader, description=f"Epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            def closure():
                weights, selected, _ = router(x)
                expert_out = experts(x, weights, selected)
                logits = classifier(expert_out)
                ce_loss = F.cross_entropy(logits, y)
                fars_loss = shaper.compute_fars_loss(weights, experts)
                return ce_loss + fars_loss
            
            loss = optimizer.step(closure)
            total_loss += loss.item()
            
            with torch.no_grad():
                weights, selected, _ = router(x)
                pred = classifier(experts(x, weights, selected)).argmax(dim=-1)
                correct += (pred == y).sum().item()
                total_samples += y.size(0)
                tracker.update(y, weights)
        
        avg_acc = correct / total_samples
        rmi = tracker.compute_rmi()
        itjd = tracker.compute_itjd()
        
        with torch.no_grad():
            test_x = dataset.x[:100].to(device)
            test_w, _, _ = router(test_x)
            sparsity = (test_w < 0.05).float().mean().item()
        
        console.print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f} | Acc: {avg_acc:.4f} | RMI: {rmi:.4f} | ITJD: {itjd:.4f} | Sparsity: {sparsity:.4f}")

if __name__ == "__main__":
    run_fars_logic_test()
