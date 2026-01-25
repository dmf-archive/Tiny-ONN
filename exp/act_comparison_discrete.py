import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from typing import Tuple, List

# ==========================================
# 1. 任务设计: 递归指针追逐 (RPC)
# ==========================================

class RPCDataset:
    def __init__(self, vocab_size=128, memory_size=16, max_steps=8, num_samples=1000):
        self.vocab_size = vocab_size
        self.memory_size = memory_size
        self.max_steps = max_steps
        self.num_samples = num_samples
        
    def generate_data(self):
        inputs = []
        memories = []
        targets = []
        steps_required = []
        
        for _ in range(self.num_samples):
            # 随机生成内存库 (Key-Value pairs)
            # 保证没有环，且存在终止符
            keys = torch.randperm(self.vocab_size - 2)[:self.memory_size] + 2 # 0, 1 留作特殊标记
            values = torch.zeros_like(keys)
            
            chain_len = np.random.randint(1, self.max_steps + 1)
            chain_indices = torch.randperm(self.memory_size)[:chain_len]
            
            # 构建链条: K[i] -> K[i+1]
            for i in range(chain_len - 1):
                values[chain_indices[i]] = keys[chain_indices[i+1]]
            
            # 终止符: 1 (Leaf)
            leaf_val = torch.randint(2, self.vocab_size, (1,))
            values[chain_indices[-1]] = 1 
            
            # 填充其余内存
            mask = torch.ones(self.memory_size, dtype=torch.bool)
            mask[chain_indices] = False
            remaining_vals = torch.randint(2, self.vocab_size, (mask.sum(),))
            values[mask] = remaining_vals
            
            start_key = keys[chain_indices[0]]
            
            inputs.append(start_key)
            memories.append(torch.stack([keys, values], dim=1)) # [M, 2]
            targets.append(leaf_val)
            steps_required.append(chain_len)
            
        return (torch.stack(inputs), 
                torch.stack(memories), 
                torch.stack(targets).squeeze(), 
                torch.tensor(steps_required))

# ==========================================
# 2. 模型架构: 递归离散推断机
# ==========================================

class RecursiveRPC(nn.Module):
    def __init__(self, vocab_size=128, d_model=64, memory_size=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.memory_encoder = nn.Linear(2 * d_model, d_model)
        
        # 递归块: 简单的 Cross-Attention 模拟
        self.recursive_block = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # ACT Heads
        self.q_head = nn.Linear(d_model, 2) # [Halt, Continue]
        self.halt_head = nn.Linear(d_model, 1) # PLSD Halt Probability
        
    def encode_memory(self, memory):
        # memory: [B, M, 2]
        k_emb = self.embed(memory[:, :, 0])
        v_emb = self.embed(memory[:, :, 1])
        m_emb = torch.cat([k_emb, v_emb], dim=-1) # [B, M, 2*D]
        m_context = self.memory_encoder(m_emb).mean(dim=1) # [B, D] 简化为平均池化
        return m_context

    def forward_step(self, h, m_context):
        # h: [B, D], m_context: [B, D]
        combined = torch.cat([h, m_context], dim=-1)
        h_next = h + self.recursive_block(combined) # 残差连接
        logits = self.lm_head(h_next)
        return h_next, logits

# ==========================================
# 3. 训练逻辑
# ==========================================

def train_qlearning(model, data, epochs=100, max_steps=8):
    inputs, memories, targets, steps_req = data
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        m_context = model.encode_memory(memories)
        h = model.embed(inputs)
        
        total_loss = 0
        q_loss = 0
        
        # 初始预测
        logits_prev = model.lm_head(h)
        loss_prev = F.cross_entropy(logits_prev, targets, reduction='none')
        
        for t in range(max_steps):
            q_vals = model.q_head(h) # [B, 2]
            h_next, logits = model.forward_step(h, m_context)
            loss_curr = F.cross_entropy(logits, targets, reduction='none')
            
            # 差分奖励: 损失下降量 - 步长惩罚
            # 在 CE 任务中，如果已经预测对(1)，则不再有奖励
            reward = (loss_prev - loss_curr).detach() - 0.05
            
            # Q-learning Target: R + gamma * max(Q_next)
            with torch.no_grad():
                q_next = model.q_head(h_next)
                target_q_continue = reward + 0.95 * q_next.max(dim=1)[0]
                # Halt 的 Target 是当前的立即奖励（停止后的未来奖励为0）
                target_q_halt = reward 
            
            # 我们简化训练: 强制跑满，但优化 Q 值
            q_loss += F.mse_loss(q_vals[:, 1], target_q_continue)
            q_loss += F.mse_loss(q_vals[:, 0], target_q_halt)
            
            # 主任务 Loss (全路径监督)
            total_loss += loss_curr.mean()
            
            h = h_next
            loss_prev = loss_curr
            
        optimizer.zero_grad()
        (total_loss + q_loss).backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Q-Epoch {epoch} | Task Loss: {total_loss.item()/max_steps:.4f} | Q Loss: {q_loss.item():.4f}")

def train_plsd(model, data, epochs=100, max_steps=8):
    inputs, memories, targets, steps_req = data
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        m_context = model.encode_memory(memories)
        h = model.embed(inputs)
        
        loss_seq = []
        h_seq = []
        
        # 展开全路径
        for t in range(max_steps):
            h, logits = model.forward_step(h, m_context)
            loss = F.cross_entropy(logits, targets, reduction='none')
            loss_seq.append(loss)
            h_seq.append(h)
            
        loss_seq = torch.stack(loss_seq, dim=1) # [B, T]
        h_seq = torch.stack(h_seq, dim=1) # [B, T, D]
        
        # 找到 Oracle 步长: 损失首次低于阈值或达到最小值
        # 在离散任务中，通常是损失骤降的点
        oracle_steps = torch.argmin(loss_seq, dim=1)
        
        # 构建 PLSD Target: t < t* 为 0, t >= t* 为 1
        batch_indices = torch.arange(max_steps).expand(inputs.size(0), max_steps)
        targets_act = (batch_indices >= oracle_steps.unsqueeze(1)).float()
        
        # 计算 Halt Head 损失
        halt_logits = model.halt_head(h_seq).squeeze(-1) # [B, T]
        act_loss = F.binary_cross_entropy_with_logits(halt_logits, targets_act)
        
        # 主任务损失
        task_loss = loss_seq.mean()
        
        optimizer.zero_grad()
        (task_loss + act_loss).backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"PLSD-Epoch {epoch} | Task Loss: {task_loss.item():.4f} | ACT Loss: {act_loss.item():.4f}")

# ==========================================
# 4. 评估与对比
# ==========================================

def evaluate(model, data, mode="q", max_steps=8, lambda_val=0.5):
    inputs, memories, targets, steps_req = data
    model.eval()
    
    correct = 0
    total_steps = 0
    
    with torch.no_grad():
        m_context = model.encode_memory(memories)
        h = model.embed(inputs)
        
        active_mask = torch.ones(inputs.size(0), dtype=torch.bool)
        exit_steps = torch.full((inputs.size(0),), max_steps, dtype=torch.long)
        final_preds = torch.zeros(inputs.size(0), dtype=torch.long)
        
        for t in range(max_steps):
            if not active_mask.any(): break
            
            h_next, logits = model.forward_step(h, m_context)
            preds = torch.argmax(logits, dim=-1)
            
            if mode == "q":
                q_vals = model.q_head(h)
                # Halt if Q(Halt) > Q(Continue)
                halt_decision = q_vals[:, 0] > q_vals[:, 1]
            else:
                halt_prob = torch.sigmoid(model.halt_head(h_next)).squeeze(-1)
                halt_decision = halt_prob > lambda_val
            
            # 记录刚停止的样本
            just_halted = active_mask & halt_decision
            exit_steps[just_halted] = t + 1
            final_preds[just_halted] = preds[just_halted]
            active_mask[just_halted] = False
            
            h = h_next
            
        # 对于跑满还没停的
        final_preds[active_mask] = torch.argmax(model.lm_head(h), dim=-1)[active_mask]
        
        correct = (final_preds == targets).sum().item()
        avg_steps = exit_steps.float().mean().item()
        
        # 计算 Oracle 对齐度 (MAE)
        oracle_steps = steps_req # 在 RPC 任务中，steps_req 就是理论最优
        step_error = torch.abs(exit_steps - oracle_steps).float().mean().item()
        
    return correct / inputs.size(0), avg_steps, step_error

if __name__ == "__main__":
    torch.manual_seed(42)
    dataset = RPCDataset(num_samples=2000)
    train_data = dataset.generate_data()
    test_data = dataset.generate_data()
    
    print("--- Training QLearning-ACT (Discrete CE) ---")
    model_q = RecursiveRPC()
    train_qlearning(model_q, train_data)
    
    print("\n--- Training PLSD-ACT (Discrete CE) ---")
    model_p = RecursiveRPC()
    train_plsd(model_p, train_data)
    
    print("\n" + "="*30)
    print("RESULTS COMPARISON (Discrete RPC Task)")
    print("="*30)
    
    q_acc, q_steps, q_err = evaluate(model_q, test_data, mode="q")
    print(f"QLearning-ACT | Acc: {q_acc:.4f} | Avg Steps: {q_steps:.2f} | Step Error: {q_err:.2f}")
    
    for l in [0.3, 0.5, 0.7]:
        p_acc, p_steps, p_err = evaluate(model_p, test_data, mode="p", lambda_val=l)
        print(f"Better-PLSD (λ={l}) | Acc: {p_acc:.4f} | Avg Steps: {p_steps:.2f} | Step Error: {p_err:.2f}")
