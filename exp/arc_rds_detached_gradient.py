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
# 1. 梯度切断版 RDS (Gradient-Detached Recursive)
# -----------------------------------------------------------------------------
class DetachedRecursiveDynSIHA(RecursiveDynSIHAForCausalLM):
    """
    重写 forward 逻辑，在递归步之间切断梯度。
    这确保了每一层接收到的梯度仅来自该层对 Loss 的直接贡献，
    消除了 BPTT 带来的“高维捷径”累积。
    """
    def forward(self, input_ids, labels=None, return_dict=True, **kwargs):
        inputs_embeds = self.embedding(input_ids)
        hidden_states = inputs_embeds
        
        all_routing_weights = []
        all_losses = []
        logits = None
        
        for i in range(self.config.max_steps):
            # --- 关键手术：切断前一步的梯度 ---
            step_input = hidden_states.detach() if i > 0 else hidden_states
            
            layer_outputs = self.block(
                step_input,
                position_ids=None,
                past_key_value=None,
                use_cache=False
            )
            
            hidden_states = layer_outputs[0]
            # 关键修复：从 DynSIHABlock 的返回中提取 mlp 权重
            # DynSIHABlock 返回 (output, routing_info, pkv)
            # routing_info 包含 mlp_weights
            routing_info = layer_outputs[1]
            all_routing_weights.append({
                "mlp": routing_info["mlp_weights"]
            })
            
            # 每一层都产生 logits 以便计算损失
            logits = self.lm_head(self.ln_f(hidden_states))
            
            if labels is not None:
                loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1), ignore_index=-100)
                all_losses.append(loss)
        
        total_loss = sum(all_losses) / len(all_losses) if all_losses else None
        
        if not return_dict:
            return (logits, hidden_states, all_routing_weights)
            
        from src.models.dynsiha.recursive.modeling_recursive_dynsiha import RecursiveDynSIHAOutput
        return RecursiveDynSIHAOutput(
            loss=total_loss,
            logits=logits,
            routing_weights=all_routing_weights
        )

# -----------------------------------------------------------------------------
# 2. 指标计算
# -----------------------------------------------------------------------------
class MetricsTracker:
    def __init__(self):
        self.history = [] 

    def update(self, task_ids, routing_weights):
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

# -----------------------------------------------------------------------------
# 3. 实验运行
# -----------------------------------------------------------------------------
def run_ablation_detached_gradient(epochs=30):
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
        hidden_size=128, num_heads=4, num_experts=8, top_k=8,
        max_steps=6, vocab_size=tokenizer.vocab_size, max_position_embeddings=512
    )
    
    # 使用切断梯度的模型
    model = DetachedRecursiveDynSIHA(config).to(device)
    collator = ArcCollator(tokenizer, max_len=512)
    loader = DataLoader(dataset, batch_size=8, collate_fn=collator, shuffle=True)
    optimizer = SingleDeviceARS2Neo(model.parameters(), lr=1e-3, rho=0.05)
    tracker = MetricsTracker()

    print(f"Starting Ablation: Gradient-Detached RDS (MaxSteps=6)")
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            input_ids, labels, task_ids = batch["input_ids"].to(device), batch["labels"].to(device), batch["task_id"]
            optimizer.zero_grad()
            def closure():
                output = model(input_ids=input_ids, labels=labels, return_dict=True)
                return output.loss
            loss = optimizer.step(closure)
            total_loss += loss.item()
            with torch.no_grad():
                output = model(input_ids=input_ids, return_dict=True)
                tracker.update(task_ids, output.routing_weights)
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            rmi = tracker.compute_rmi()
            last_w = output.routing_weights[-1]["mlp"]
            sparsity = (last_w < 0.05).float().mean().item()
            print(f"Detached | Ep {epoch} | Loss: {total_loss/len(loader):.4f} | RMI: {rmi:.4f} | Sparsity: {sparsity:.4f}")

    print("\n" + "="*60)
    print("GRADIENT-DETACHED RDS FINAL REPORT")
    print("="*60)
    print(f"Loss: {total_loss/len(loader):.4f}")
    print(f"RMI: {rmi:.4f}")
    print(f"Sparsity: {sparsity:.4f}")
    print("="*60)

if __name__ == "__main__":
    run_ablation_detached_gradient()
