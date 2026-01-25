import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict
from src.optimizers.ars2_neo import SingleDeviceARS2Neo

# ==========================================
# 1. 任务设计: 带 Task ID 的 RPC
# ==========================================

class TaskRPCDataset:
    def __init__(self, vocab_size=64, memory_size=16, max_steps=8, num_tasks=100, samples_per_task=10):
        self.vocab_size = vocab_size
        self.memory_size = memory_size
        self.max_steps = max_steps
        self.num_tasks = num_tasks
        self.samples_per_task = samples_per_task
        
    def generate_data(self):
        inputs, memories, targets, steps_req, task_ids = [], [], [], [], []
        
        for t_id in range(self.num_tasks):
            # 每个 Task 共享一个 Memory 库
            keys = torch.randperm(self.vocab_size - 2)[:self.memory_size] + 2 
            values = torch.zeros_like(keys)
            
            # 预生成一个主链条
            chain_len = np.random.randint(2, self.max_steps + 1)
            chain_indices = torch.randperm(self.memory_size)[:chain_len]
            for i in range(chain_len - 1):
                values[chain_indices[i]] = keys[chain_indices[i+1]]
            
            target_val = torch.randint(2, self.vocab_size, (1,))
            values[chain_indices[-1]] = target_val
            
            # 填充其余内存
            mask = torch.ones(self.memory_size, dtype=torch.bool)
            mask[chain_indices] = False
            values[mask] = torch.randint(2, self.vocab_size, (mask.sum(),))
            
            # 为该 Task 生成多个样本（不同起点或微小扰动）
            for _ in range(self.samples_per_task):
                inputs.append(keys[chain_indices[0]])
                memories.append(torch.stack([keys, values], dim=1))
                targets.append(target_val)
                steps_req.append(chain_len)
                task_ids.append(t_id)
                
        return (torch.stack(inputs), 
                torch.stack(memories), 
                torch.stack(targets).squeeze(), 
                torch.tensor(steps_req),
                torch.tensor(task_ids))

# ==========================================
# 2. 模型架构: 8*16 Top-Any 递归机
# ==========================================

class TopAnyRouter(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_experts)
        )
        
    def forward(self, x):
        logits = self.net(x)
        # Top-Any: 不做硬截断，由 FARS 驱动稀疏
        weights = F.softmax(logits, dim=-1)
        return weights, logits

class AdaptiveRecursiveRPC(nn.Module):
    def __init__(self, vocab_size=64, d_model=16, num_experts=8):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.embed = nn.Embedding(vocab_size, d_model)
        self.router = TopAnyRouter(d_model, num_experts)
        
        # 专家库: 8个 16->16 的线性层
        self.experts = nn.Parameter(torch.randn(num_experts, d_model, d_model) * 0.02)
        
        self.halt_head = nn.Linear(d_model, 2)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.norm = nn.LayerNorm(d_model)

    def forward_step(self, h):
        weights, logits = self.router(h) # [B, E]
        
        # 批量矩阵乘法实现专家组合
        # h: [B, D], experts: [E, D, D] -> out: [B, D]
        # h: [B, D], experts: [E, D, D] -> out: [B, E, D]
        expert_outputs = torch.einsum('bd, edk -> bek', h, self.experts)
        combined_output = torch.sum(weights.unsqueeze(-1) * expert_outputs, dim=1)
        
        h_next = self.norm(h + combined_output)
        return h_next, weights, logits

# ==========================================
# 3. 核心逻辑: FARS + Consistency
# ==========================================

def train_study(mode="fars_only", epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TaskRPCDataset(num_tasks=50, samples_per_task=10)
    data = [d.to(device) for d in dataset.generate_data()]
    inputs, memories, targets, steps_req, task_ids = data
    
    model = AdaptiveRecursiveRPC(d_model=16, num_experts=8).to(device)
    # 使用 SingleDeviceARS2Neo 优化器
    param_groups = [
        {'params': [p for p in model.parameters() if p.ndim >= 2], 'is_rmsuon_group': True},
        {'params': [p for p in model.parameters() if p.ndim < 2], 'is_rmsuon_group': False}
    ]
    optimizer = SingleDeviceARS2Neo(param_groups, lr=1e-3, rho=0.02)
    
    for epoch in range(epochs):
        epoch_metrics = {}
        def closure():
            optimizer.zero_grad()
            h = model.norm(model.embed(inputs))
            
            all_logits, all_weights, all_h = [], [], []
            loss_seq = []
            
            # 递归展开
            for t in range(8):
                h, weights, logits = model.forward_step(h)
                all_logits.append(logits)
                all_weights.append(weights)
                all_h.append(h)
                
                # PLSD 损失捕获
                step_logits = model.lm_head(h)
                l_step = F.cross_entropy(step_logits, targets, reduction='none')
                loss_seq.append(l_step)
            
            loss_seq_stack = torch.stack(loss_seq, dim=1) # [B, 8]
            weights_stack = torch.stack(all_weights, dim=1) # [B, 8, E]
            
            # 计算 Effective K (自发稀疏度)
            # 使用 Shannon 熵的指数作为有效专家数
            entropy = -torch.sum(weights_stack * torch.log(weights_stack + 1e-10), dim=-1)
            eff_k = torch.exp(entropy).mean().item()
            epoch_metrics['eff_k'] = eff_k

            oracle_steps = torch.argmin(loss_seq_stack.detach(), dim=1)
            
            # 1. 主任务损失 (PLSD Oracle 对齐)
            main_loss = loss_seq_stack[torch.arange(inputs.size(0)), oracle_steps].mean()
            epoch_metrics['main_loss'] = main_loss.item()
            
            # 2. FARS 塑造 (Fisher-Aware Cost)
            fars_loss = 0
            # 从优化器状态中提取专家的 Fisher 信息 (v_t)
            expert_params = model.experts
            if expert_params in optimizer.state:
                v_t = optimizer.state[expert_params].get('exp_avg_sq', None)
                if v_t is not None:
                    # 计算每个专家的平均 Fisher 强度 [E]
                    expert_costs = torch.sqrt(v_t).mean(dim=(1, 2))
                    expert_costs = expert_costs / (expert_costs.norm() + 1e-8)
                    
                    # 路由权重与成本的内积
                    weights_stack = torch.stack(all_weights, dim=1)
                    fars_loss = (weights_stack * expert_costs).sum(dim=-1).mean()
            
            # 3. Consistency 损失 (Task-ID Based)
            cons_loss_diag = 0
            weights_stack = torch.stack(all_weights, dim=1) # [B, 8, E]
            for t_id in range(50):
                mask = (task_ids == t_id)
                if mask.any():
                    task_avg = weights_stack[mask].mean(dim=0, keepdim=True).detach()
                    cons_loss_diag += F.kl_div((weights_stack[mask] + 1e-10).log(), task_avg.expand_as(weights_stack[mask]), reduction='batchmean')
            
            epoch_metrics['cons_loss'] = cons_loss_diag.item() if torch.is_tensor(cons_loss_diag) else cons_loss_diag
            cons_loss = cons_loss_diag if mode == "fars_consistency" else 0
            
            # 4. 组合损失
            total_loss = main_loss + 0.05 * fars_loss + 0.1 * cons_loss
            
            # 5. 统计项: 任务-专家互信息 (RMI)
            # 衡量路由分布对 Task ID 的依赖程度
            with torch.no_grad():
                # [B, E] - 样本平均路由
                sample_routing = weights_stack.mean(dim=1)
                task_p = []
                for t_id in range(50):
                    mask = (task_ids == t_id)
                    if mask.any():
                        task_p.append(sample_routing[mask].mean(dim=0))
                
                if task_p:
                    task_p = torch.stack(task_p) # [NumTasks, E]
                    p_global = task_p.mean(dim=0)
                    h_global = -torch.sum(p_global * torch.log(p_global + 1e-10))
                    h_local = -torch.sum(task_p * torch.log(task_p + 1e-10), dim=-1).mean()
                    rmi = h_global - h_local
                    epoch_metrics['rmi'] = rmi.item()
                    
                    # 6. 统计项: 任务间路由差分率 (ITJD - Inter-Task Jaccard Distance)
                    # 衡量不同任务路由分布的非重叠程度
                    if task_p.size(0) > 1:
                        # 计算成对 Jaccard 距离: 1 - sum(min(p1, p2)) / sum(max(p1, p2))
                        p1 = task_p.unsqueeze(1) # [N, 1, E]
                        p2 = task_p.unsqueeze(0) # [1, N, E]
                        intersection = torch.min(p1, p2).sum(dim=-1)
                        union = torch.max(p1, p2).sum(dim=-1)
                        jaccard_dist = 1 - intersection / (union + 1e-10)
                        
                        # 取上三角平均值（排除自比较）
                        mask_tri = torch.triu(torch.ones_like(jaccard_dist), diagonal=1).bool()
                        itjd = jaccard_dist[mask_tri].mean()
                        epoch_metrics['itjd'] = itjd.item()

            return total_loss

        optimizer.step(closure)
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Mode: {mode} | Main_Loss: {epoch_metrics.get('main_loss', 0):.4f} | RMI: {epoch_metrics.get('rmi', 0):.4f} | ITJD: {epoch_metrics.get('itjd', 0):.4f} | Eff_K: {epoch_metrics.get('eff_k', 0):.2f}")

if __name__ == "__main__":
    print("Starting FARS-Consistency Study...")
    train_study(mode="fars_only", epochs=1000)
    train_study(mode="fars_consistency", epochs=1000)
