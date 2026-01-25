import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Tuple, List, Dict

"""
ACT Comparison Experiment v3: ADL-Enhanced & Better PLSD
========================================================

本实验引入 ARC-Theory 中的 ADL (H2) 思想与成熟的早停机制：
1. ADL (Adaptive Differential Loss): 聚焦变换 (Transformation)，抑制复制 (Copying)。
2. Differential Reward: Q-learning 奖励基于状态改进量与复制惩罚。
3. Better PLSD: 引入阈值 lambda 控制早停，模拟 AGA 的自适应同步逻辑。
4. Fallback Mechanism: 设定最大步数作为兜底。

任务：稀疏序列变换。
"""

# --- 1. Core Recursive Module ---

class RecursiveCore(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.SiLU(),
            nn.Linear(d * 2, d)
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, h):
        return self.norm(h + self.net(h))

# --- 2. ACT Mechanisms ---

class QLearningACT(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_head = nn.Linear(d, 2)

    def forward(self, h):
        return self.q_head(h)

class PLSDACT(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.halt_head = nn.Linear(d, 1)

    def forward(self, h):
        return self.halt_head(h)

# --- 3. Task Generator (ADL-Style Sparse Transformation) ---

def generate_adl_data(b, d, max_steps):
    torch.manual_seed(42)
    x = torch.randn(b, d)
    
    # 复杂度决定了需要多少次“有效变换”
    complexity = torch.randint(1, max_steps, (b,))
    
    # 构造稀疏变换：每一步只改变 20% 的维度
    targets = [x]
    curr = x.clone()
    
    for _ in range(max_steps):
        mask = (torch.rand(b, d) < 0.2).float()
        delta = torch.randn(b, d) * 0.5
        curr = curr + mask * delta
        targets.append(curr)
    
    y = torch.stack([targets[c.item()][i] for i, c in enumerate(complexity)])
    
    # 计算 Diff Mask: 最终输出与输入的差异点
    diff_mask = (torch.abs(y - x) > 1e-5).float()
    
    return x, y, complexity, diff_mask

# --- 4. Training Logic ---

def train_q_learning_adl(core, act, x, y, complexity, max_steps, diff_mask):
    optimizer = torch.optim.AdamW(list(core.parameters()) + list(act.parameters()), lr=1e-3)
    
    for epoch in range(500):
        optimizer.zero_grad()
        h = x
        q_values = []
        states = []
        
        for t in range(max_steps):
            h = core(h)
            q = act(h)
            q_values.append(q)
            states.append(h)
            
        # ADL Differential Reward
        rewards = []
        prev_mse = torch.sum((x - y)**2, dim=-1)
        
        for s in states:
            curr_mse = torch.sum((s - y)**2, dim=-1)
            improvement = prev_mse - curr_mse
            # 惩罚对非变化区域的修改
            copy_penalty = torch.sum(((s - x) * (1 - diff_mask))**2, dim=-1)
            reward = improvement - 0.5 * copy_penalty
            rewards.append(reward)
            prev_mse = curr_mse
            
        rewards = torch.stack(rewards)
        
        q_loss = 0
        for t in range(max_steps - 1):
            target_q_cont = rewards[t+1] + 0.9 * q_values[t+1].max(dim=-1)[0].detach()
            q_loss += F.mse_loss(q_values[t][:, 1], target_q_cont)
            q_loss += F.mse_loss(q_values[t][:, 0], rewards[t])
            
        # ADL Weighted Task Loss
        final_loss = (torch.sum((states[-1] - y)**2 * (1 + 5 * diff_mask), dim=-1)).mean()
        
        (final_loss + 0.1 * q_loss).backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Q-ACT (ADL) Epoch {epoch} | Loss: {final_loss.item():.4f}")

def train_plsd_better(core, act, x, y, complexity, max_steps, diff_mask):
    optimizer = torch.optim.AdamW(list(core.parameters()) + list(act.parameters()), lr=1e-3)
    
    for epoch in range(500):
        optimizer.zero_grad()
        h = x
        step_losses = []
        halt_logits = []
        
        for t in range(max_steps):
            h = core(h)
            # ADL Weighted Step Loss
            step_loss = torch.sum((h - y)**2 * (1 + 5 * diff_mask), dim=-1)
            step_losses.append(step_loss)
            
            halt_logit = act(h)
            halt_logits.append(halt_logit.squeeze(-1))
            
        step_losses_tensor = torch.stack(step_losses)
        halt_logits_tensor = torch.stack(halt_logits)
        
        # Oracle Step: 认知饱和点
        best_losses, oracle_steps = step_losses_tensor.min(dim=0)
        loss_task = best_losses.mean()
        
        # PLSD Alignment: 拟合阶跃函数
        targets = torch.zeros_like(halt_logits_tensor)
        for b in range(x.shape[0]):
            targets[oracle_steps[b]:, b] = 1.0
            
        loss_act = F.binary_cross_entropy_with_logits(halt_logits_tensor, targets)
        
        (loss_task + 0.1 * loss_act).backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Better-PLSD Epoch {epoch} | Loss: {loss_task.item():.4f} | Oracle Step: {oracle_steps.float().mean().item():.2f}")

# --- 5. Evaluation with Threshold (Lambda) ---

def evaluate_better(name, core, act, act_type, x, y, target_steps, max_steps, diff_mask, lambda_val=0.5):
    core.eval()
    act.eval()
    
    h = x
    exit_steps = torch.full((x.shape[0],), max_steps - 1, dtype=torch.long)
    finished = torch.zeros(x.shape[0], dtype=torch.bool)
    
    with torch.no_grad():
        for t in range(max_steps):
            h = core(h)
            if act_type == "q":
                q = act(h)
                # Q-learning: 比较 Q 值，但也可以引入阈值偏置
                halt = (q[:, 0] - q[:, 1]) > (1.0 - lambda_val)
            else:
                # PLSD: 基于阈值 lambda 的早停
                halt = torch.sigmoid(act(h)).squeeze(-1) > lambda_val
            
            new_halt = halt & (~finished)
            exit_steps[new_halt] = t
            finished |= halt
            
            if finished.all(): break
            
    mse_diff = torch.mean(((h - y)**2 * diff_mask)).item() / (diff_mask.mean().item() + 1e-9)
    mse_copy = torch.mean(((h - x)**2 * (1 - diff_mask))).item() / ((1 - diff_mask).mean().item() + 1e-9)
    
    step_diff = torch.abs(exit_steps - target_steps).float().mean().item()
    savings = 1.0 - (exit_steps.float().mean().item() / (max_steps - 1))
    
    print(f"\nResults for {name} (lambda={lambda_val}):")
    print(f"  Diff MSE: {mse_diff:.4f}")
    print(f"  Copy MSE: {mse_copy:.4f}")
    print(f"  Step Error: {step_diff:.2f}")
    print(f"  Savings: {savings:.2%}")

if __name__ == "__main__":
    B, D, MAX_STEPS = 1024, 32, 10
    x, y, complexity, diff_mask = generate_adl_data(B, D, MAX_STEPS)
    
    print("--- Training QLearning-ACT (ADL-Enhanced) ---")
    q_core = RecursiveCore(D)
    q_act = QLearningACT(D)
    train_q_learning_adl(q_core, q_act, x, y, complexity, MAX_STEPS, diff_mask)
    
    print("\n--- Training Better-PLSD (ADL-Enhanced) ---")
    p_core = RecursiveCore(D)
    p_act = PLSDACT(D)
    train_plsd_better(p_core, p_act, x, y, complexity, MAX_STEPS, diff_mask)
    
    # Compare with different lambda thresholds
    for l in [0.3, 0.7, 0.9]:
        evaluate_better("QLearning-ACT", q_core, q_act, "q", x, y, complexity, MAX_STEPS, diff_mask, lambda_val=l)
        evaluate_better("Better-PLSD", p_core, p_act, "p", x, y, complexity, MAX_STEPS, diff_mask, lambda_val=l)
