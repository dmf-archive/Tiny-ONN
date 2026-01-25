import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from typing import Tuple, List
from src.optimizers.ars2_neo import SingleDeviceARS2Neo

# ==========================================
# 1. 任务设计: 递归指针追逐 (RPC) - 增强版
# ==========================================

class RPCDataset:
    def __init__(self, vocab_size=64, memory_size=16, max_steps=8, num_samples=1000, overthink_noise=False):
        self.vocab_size = vocab_size
        self.memory_size = memory_size
        self.max_steps = max_steps
        self.num_samples = num_samples
        self.overthink_noise = overthink_noise
        
    def generate_data(self, override_max_steps=None):
        max_steps = override_max_steps or self.max_steps
        inputs = []
        memories = []
        targets = []
        steps_required = []
        
        for _ in range(self.num_samples):
            # 随机生成内存库 (Key-Value pairs)
            keys = torch.randperm(self.vocab_size - 2)[:self.memory_size] + 2 
            values = torch.zeros_like(keys)
            
            chain_len = np.random.randint(1, max_steps + 1)
            chain_indices = torch.randperm(self.memory_size)[:chain_len]
            
            # 构建链条: K[i] -> K[i+1]
            for i in range(chain_len - 1):
                values[chain_indices[i]] = keys[chain_indices[i+1]]
            
            # 最后一个 key 指向 target_val
            while True:
                target_val = torch.randint(2, self.vocab_size, (1,))
                if target_val not in keys:
                    break
            
            values[chain_indices[-1]] = target_val
            
            # 填充其余内存
            mask = torch.ones(self.memory_size, dtype=torch.bool)
            mask[chain_indices] = False
            remaining_vals = torch.randint(2, self.vocab_size, (mask.sum(),))
            values[mask] = remaining_vals
            
            start_key = keys[chain_indices[0]]
            
            inputs.append(start_key)
            memories.append(torch.stack([keys, values], dim=1)) 
            targets.append(target_val)
            steps_required.append(chain_len)
            
        return (torch.stack(inputs), 
                torch.stack(memories), 
                torch.stack(targets).squeeze(), 
                torch.tensor(steps_required))

# ==========================================
# 2. 模型架构: 递归离散推断机 (Attention-based Memory)
# ==========================================

class RecursiveRPC(nn.Module):
    def __init__(self, vocab_size=64, d_model=256, memory_size=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        
        # Memory Lookup Projections
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        
        self.recursive_block = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # ACT Heads
        self.q_head = nn.Linear(d_model, 2) # [Halt, Continue]
        self.halt_head = nn.Linear(d_model, 2) # PLSD [Halt, Continue]
        
    def get_memory_embeddings(self, memory):
        # memory: [B, M, 2]
        m_keys_emb = self.embed(memory[:, :, 0])
        m_vals_emb = self.embed(memory[:, :, 1])
        return m_keys_emb, m_vals_emb

    def forward_step(self, h, m_keys_emb, m_vals_emb):
        # 1. Attention-based Lookup
        q = self.query_proj(h).unsqueeze(1) # [B, 1, D]
        k = self.key_proj(m_keys_emb)       # [B, M, D]
        v = self.value_proj(m_vals_emb)     # [B, M, D]
        
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * (h.size(-1) ** -0.5) # [B, 1, M]
        attn_probs = F.softmax(attn_weights, dim=-1)
        context = torch.bmm(attn_probs, v).squeeze(1) # [B, D]
        
        # 2. State Update
        combined = torch.cat([h, context], dim=-1)
        h_next = h + self.recursive_block(combined) 
        logits = self.lm_head(h_next)
        return h_next, logits

# ==========================================
# 3. 训练逻辑
# ==========================================

def get_ars2_groups(model):
    rmsuon_params = []
    adamw_params = []
    for name, p in model.named_parameters():
        if p.ndim >= 2:
            rmsuon_params.append(p)
        else:
            adamw_params.append(p)
    return [
        {'params': rmsuon_params, 'is_rmsuon_group': True},
        {'params': adamw_params, 'is_rmsuon_group': False}
    ]

def train_qlearning_prob(model, data, epochs=100, batch_size=128, max_steps=8):
    device = next(model.parameters()).device
    inputs_all, memories_all, targets_all, steps_req_all = [d.to(device) for d in data]
    num_samples = inputs_all.size(0)
    
    param_groups = get_ars2_groups(model)
    optimizer = SingleDeviceARS2Neo(param_groups, lr=5e-4, rho=0.02, k=1)
    
    for epoch in range(epochs):
        indices = torch.randperm(num_samples)
        epoch_loss = 0
        
        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            inputs = inputs_all[batch_idx]
            memories = memories_all[batch_idx]
            targets = targets_all[batch_idx]
            steps_req = steps_req_all[batch_idx]
            
            def closure():
                optimizer.zero_grad()
                m_keys_emb, m_vals_emb = model.get_memory_embeddings(memories)
                h = model.norm(model.embed(inputs))
                
                total_task_loss = 0
                q_loss = 0
                correctness_seq = []
                h_seq = [h]
                
                for t in range(max_steps):
                    h, logits = model.forward_step(h, m_keys_emb, m_vals_emb)
                    is_correct = (torch.argmax(logits, dim=-1) == targets).float()
                    correctness_seq.append(is_correct)
                    h_seq.append(h)
                    
                    mask = (t + 1 == steps_req).float()
                    if mask.sum() > 0:
                        step_loss = F.cross_entropy(logits, targets, reduction='none')
                        total_task_loss += (step_loss * mask).sum() / (mask.sum() + 1e-9)

                correctness_seq_stack = torch.stack(correctness_seq, dim=1)
                
                for t in range(max_steps):
                    q_logits = model.q_head(h_seq[t])
                    target_halt = correctness_seq_stack[:, t]
                    if t < max_steps - 1:
                        target_cont = correctness_seq_stack[:, t+1:].max(dim=1)[0]
                    else:
                        target_cont = torch.zeros_like(target_halt)
                        
                    q_loss += F.binary_cross_entropy_with_logits(q_logits[:, 0], target_halt)
                    q_loss += F.binary_cross_entropy_with_logits(q_logits[:, 1], target_cont)
                
                loss = total_task_loss + 0.5 * (q_loss / max_steps)
                return loss
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss_val = optimizer.step(closure)
            epoch_loss += loss_val.item()
            
        if epoch % 10 == 0:
            print(f"Q-Prob Epoch {epoch} | Avg Loss: {epoch_loss / (num_samples/batch_size):.4f} | Phi: {optimizer.diagnostics['phi_t']:.4f}")

def train_plsd_robust(model, data, epochs=100, batch_size=128, max_steps=8, overthink_penalty=False):
    device = next(model.parameters()).device
    inputs_all, memories_all, targets_all, steps_req_all = [d.to(device) for d in data]
    num_samples = inputs_all.size(0)
    
    param_groups = get_ars2_groups(model)
    optimizer = SingleDeviceARS2Neo(param_groups, lr=5e-4, rho=0.02, k=1)
    
    for epoch in range(epochs):
        indices = torch.randperm(num_samples)
        epoch_loss = 0
        
        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            inputs = inputs_all[batch_idx]
            memories = memories_all[batch_idx]
            targets = targets_all[batch_idx]
            steps_req = steps_req_all[batch_idx]
            
            def closure():
                optimizer.zero_grad()
                m_keys_emb, m_vals_emb = model.get_memory_embeddings(memories)
                h = model.norm(model.embed(inputs))
                
                loss_seq = []
                h_seq = []
                
                for t in range(max_steps):
                    h, logits = model.forward_step(h, m_keys_emb, m_vals_emb)
                    
                    mask = (t + 1 == steps_req).float()
                    base_loss = F.cross_entropy(logits, targets, reduction='none')
                    # 仅在 steps_req 处提供真实梯度，其余步长 detach 以防止干扰
                    loss = base_loss * mask + (1 - mask) * base_loss.detach()
                    
                    if overthink_penalty:
                        # 惩罚项权重随训练进程逐渐增加，防止初期坍缩
                        penalty_weight = min(1.0, epoch / 50.0)
                        overstep = (t + 1 - steps_req).clamp(min=0).float()
                        loss = loss + penalty_weight * overstep
                        
                    loss_seq.append(loss)
                    h_seq.append(h)
                    
                loss_seq_stack = torch.stack(loss_seq, dim=1)
                h_seq_stack = torch.stack(h_seq, dim=1)
                
                # Oracle 步长：在模型尚未学会任务时，强制 Oracle 为 steps_req
                # 只有当模型在某一步的 base_loss 足够低时，才允许 argmin 自由发挥
                oracle_steps = torch.argmin(loss_seq_stack, dim=1)
                
                batch_indices = torch.arange(max_steps, device=device).expand(inputs.size(0), max_steps)
                targets_act_halt = (batch_indices >= oracle_steps.unsqueeze(1)).float()
                targets_act_cont = (batch_indices < oracle_steps.unsqueeze(1)).float()
                
                halt_logits = model.halt_head(h_seq_stack)
                act_loss = F.binary_cross_entropy_with_logits(halt_logits[:, :, 0], targets_act_halt)
                act_loss += F.binary_cross_entropy_with_logits(halt_logits[:, :, 1], targets_act_cont)
                
                # 任务损失仅计算 steps_req 处的
                task_loss = (loss_seq_stack * (batch_indices + 1 == steps_req.unsqueeze(1)).float()).sum() / inputs.size(0)
                
                return (task_loss + 0.5 * act_loss)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss_val = optimizer.step(closure)
            epoch_loss += loss_val.item()
            
        if epoch % 10 == 0:
            print(f"PLSD-Robust Epoch {epoch} | Avg Loss: {epoch_loss / (num_samples/batch_size):.4f} | Phi: {optimizer.diagnostics['phi_t']:.4f}")

# ==========================================
# 4. 评估与对比
# ==========================================

def evaluate_robust(model, data, mode="q", max_steps=8, lambda_val=0.5):
    device = next(model.parameters()).device
    inputs, memories, targets, steps_req = [d.to(device) for d in data]
    model.eval()
    
    with torch.no_grad():
        m_keys_emb, m_vals_emb = model.get_memory_embeddings(memories)
        h = model.norm(model.embed(inputs))
        
        active_mask = torch.ones(inputs.size(0), dtype=torch.bool, device=device)
        exit_steps = torch.full((inputs.size(0),), max_steps, dtype=torch.long, device=device)
        final_preds = torch.zeros(inputs.size(0), dtype=torch.long, device=device)
        
        for t in range(max_steps):
            if not active_mask.any(): break
            
            h_next, logits = model.forward_step(h, m_keys_emb, m_vals_emb)
            preds = torch.argmax(logits, dim=-1)
            
            if mode == "q":
                q_logits = model.q_head(h)
                halt_decision = q_logits[:, 0] > q_logits[:, 1]
            else:
                halt_logits = model.halt_head(h_next)
                halt_decision = halt_logits[:, 0] > halt_logits[:, 1]
            
            just_halted = active_mask & halt_decision
            exit_steps[just_halted] = t + 1
            final_preds[just_halted] = preds[just_halted]
            active_mask[just_halted] = False
            
            h = h_next
            
        final_preds[active_mask] = torch.argmax(model.lm_head(h), dim=-1)[active_mask]
        
        acc = (final_preds == targets).sum().item() / inputs.size(0)
        avg_steps = exit_steps.float().mean().item()
        step_error = torch.abs(exit_steps - steps_req).float().mean().item()
        
    return acc, avg_steps, step_error

if __name__ == "__main__":
    torch.manual_seed(42)
    MAX_STEPS = 8
    OOD_STEPS = 12
    VOCAB_SIZE = 64
    D_MODEL = 128
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = RPCDataset(vocab_size=VOCAB_SIZE, num_samples=5000, max_steps=MAX_STEPS)
    train_data = dataset.generate_data()
    test_data = dataset.generate_data()
    ood_data = dataset.generate_data(override_max_steps=OOD_STEPS)
    
    print(f"--- Training Probabilistic Q-Learning (BCE) + ARS2-Neo [D={D_MODEL}] ---")
    model_q = RecursiveRPC(vocab_size=VOCAB_SIZE, d_model=D_MODEL).to(device)
    train_qlearning_prob(model_q, train_data, epochs=200, batch_size=256, max_steps=MAX_STEPS)
    
    print(f"\n--- Training PLSD (Overthink Penalty) + ARS2-Neo [D={D_MODEL}] ---")
    model_p = RecursiveRPC(vocab_size=VOCAB_SIZE, d_model=D_MODEL).to(device)
    train_plsd_robust(model_p, train_data, epochs=200, batch_size=256, max_steps=MAX_STEPS, overthink_penalty=True)
    
    print("\n" + "="*40)
    print("RESULTS: IID (Steps <= 8)")
    print("="*40)
    
    q_acc, q_steps, q_err = evaluate_robust(model_q, test_data, mode="q", max_steps=MAX_STEPS)
    print(f"Q-Prob      | Acc: {q_acc:.4f} | Avg Steps: {q_steps:.2f} | Step Error: {q_err:.2f}")
    
    p_acc, p_steps, p_err = evaluate_robust(model_p, test_data, mode="p", max_steps=MAX_STEPS, lambda_val=0.5)
    print(f"PLSD-Robust | Acc: {p_acc:.4f} | Avg Steps: {p_steps:.2f} | Step Error: {p_err:.2f}")
    
    print("\n" + "="*40)
    print("RESULTS: OOD (Steps up to 12)")
    print("="*40)
    
    q_acc_ood, q_steps_ood, q_err_ood = evaluate_robust(model_q, ood_data, mode="q", max_steps=OOD_STEPS)
    print(f"Q-Prob      | Acc: {q_acc_ood:.4f} | Avg Steps: {q_steps_ood:.2f} | Step Error: {q_err_ood:.2f}")
    
    p_acc_ood, p_steps_ood, p_err_ood = evaluate_robust(model_p, ood_data, mode="p", max_steps=OOD_STEPS, lambda_val=0.5)
    print(f"PLSD-Robust | Acc: {p_acc_ood:.4f} | Avg Steps: {p_steps_ood:.2f} | Step Error: {p_err_ood:.2f}")
