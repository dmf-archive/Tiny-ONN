import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict

"""
Tiny-ONN Routing Expressiveness & Evolution Tutorial
====================================================

本脚本旨在演示 DynSIHA 演进过程中不同路由架构的表达力差异。
核心结论：
1. 基于原型的路由 (CPR/CAPR) 在处理非线性逻辑（如 XOR）时存在本质局限。
2. MLP 路由凭借通用函数近似能力，是目前处理逻辑密集型任务（如 ARC）的最佳选择。
3. 路由的本质是在推断空间中寻找最优的激活路径以逼近测地线。

Mathematical Foundations:
- Linear: y = xW + b
- Cosine (CPR): y = (x · p) / (‖x‖‖p‖)
- MLP: y = W₂σ(W₁x + b₁) + b₂
"""

# --- 1. Router Definitions ---

class LinearRouter(nn.Module):
    """
    最基础的线性路由。
    优点：计算极快，参数最少。
    缺点：无法处理任何非线性分类任务。
    """
    def __init__(self, d, n):
        super().__init__()
        self.layer = nn.Linear(d, n)
    def forward(self, x):
        return self.layer(x)

class CPRRouter(nn.Module):
    """
    CPR (Cosine Prototype Routing): 基于余弦相似度的原型匹配。
    优点：具有极强的几何直观性，权重即原型。
    缺点：受限于线性流形划分，无法解决 XOR 问题。
    """
    def __init__(self, d, n):
        super().__init__()
        self.proto = nn.Parameter(torch.randn(n, d) * 0.02)
    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=-1)
        p_norm = F.normalize(self.proto, p=2, dim=-1)
        return torch.matmul(x_norm, p_norm.t())

class CAPRRouter(nn.Module):
    """
    CAPR (Cross-Attention Prototype Routing): 缩放点积原型路由。
    优点：引入了缩放因子，比 CPR 具有更尖锐的决策边界。
    缺点：本质仍是原型匹配，表达力上限受限。
    """
    def __init__(self, d, n):
        super().__init__()
        self.proto_k = nn.Parameter(torch.randn(n, d) * 0.02)
        self.gate = nn.Parameter(torch.zeros(n))
    def forward(self, x):
        logits = torch.einsum("bd,nd->bn", x, self.proto_k) / (x.shape[-1]**0.5)
        return F.relu(logits - self.gate)

class MLPRouter(nn.Module):
    """
    MLP Router (2-layer): 通用函数近似器。
    优点：能够完美解决 XOR 和复杂流形划分问题。
    缺点：缺乏显式的原型解释性，容易过拟合（需配合 ARS2-Neo 的平坦度约束）。
    """
    def __init__(self, d, n, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h),
            nn.SiLU(),
            nn.Linear(h, n)
        )
    def forward(self, x):
        return self.net(x)

class GatedMLPRouter(nn.Module):
    """
    Gated MLP Router: 利用门控机制控制信息流。
    优点：引入了乘法交互，增强了特征选择能力。
    """
    def __init__(self, d, n, h=64):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d, h), nn.Sigmoid())
        self.up = nn.Sequential(nn.Linear(d, h), nn.SiLU())
        self.down = nn.Linear(h, n)
    def forward(self, x):
        return self.down(self.up(x) * self.gate(x))

class TrLocalRouter(nn.Module):
    """
    Tr-Local: 特征级注意力路由。
    优点：通过动态权重投影实现极高的表达力，曾是早期实验的冠军。
    缺点：计算开销略高于 MLP。
    """
    def __init__(self, d, n, h=64):
        super().__init__()
        self.qkv = nn.Linear(d, d * 3)
        self.ffn = nn.Sequential(
            nn.Linear(d, h),
            nn.SiLU(),
            nn.Linear(h, d)
        )
        self.head = nn.Linear(d, n)
    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        attn = torch.softmax((q * k) / (x.shape[-1]**0.5), dim=-1)
        x = x + (attn * v)
        x = x + self.ffn(x)
        return self.head(x)

# --- 2. Task Generators ---

def generate_benchmark_data(b, d, n):
    torch.manual_seed(42)
    x = torch.randn(b, d)
    
    # Task 1: Linear Rule
    w_true = torch.randn(d, n)
    y_linear = (x @ w_true).argmax(-1)
    
    # Task 2: XOR Rule (Tests non-linear logic)
    y_xor = (torch.sum(x[:, :4] > 0, dim=-1) % n).long()
    
    # Task 3: Proto Rule (Tests clustering)
    centroids = torch.randn(n, d)
    y_proto = torch.cdist(x, centroids).argmin(-1)
    
    # Task 4: V5 Complex Rule (XOR + Conditional + Manifold)
    y_complex = torch.zeros(b, dtype=torch.long)
    for i in range(b):
        if (x[i, 0] > 0) ^ (x[i, 1] > 0):
            y_complex[i] = 0
        elif (x[i, 2] > 0 and x[i, 3] > 0.5) or (x[i, 2] <= 0 and x[i, 4] > 0.5):
            y_complex[i] = 1
        elif (x[i, 5]**2 + x[i, 6]**2) < 1.0:
            y_complex[i] = 2
        else:
            y_complex[i] = 3
            
    return x, {
        "Linear": y_linear,
        "XOR": y_xor,
        "Proto": y_proto,
        "Complex": y_complex % n
    }

# --- 3. Evaluation Engine ---

def run_eval(name, model, x, y, epochs=1000):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    start_cpu = time.process_time()
    
    for e in range(epochs):
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        
        # 引入熵正则化以保证路由决策的确定性
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
        
        (loss + 0.05 * entropy).backward()
        optimizer.step()
        
        if (e + 1) % 100 == 0:
            acc = (logits.argmax(-1) == y).float().mean().item()
            if acc > 0.99: break
            
    duration = (time.process_time() - start_cpu) * 1000
    
    # 归因分析 (Interpretability Analysis)
    model.eval()
    x_test = x[:512].detach().clone()
    x_test.requires_grad = True
    logits_test = model(x_test)
    loss_test = F.cross_entropy(logits_test, y[:512])
    loss_test.backward()
    
    # 计算显著性 (Saliency)
    saliency = x_test.grad.abs().mean(dim=0)
    # 在 Complex 任务中，前 7 维是语义相关的
    snr = saliency[:7].mean() / (saliency[7:].mean() + 1e-9)
    
    final_acc = (logits_test.argmax(-1) == y[:512]).float().mean().item()
    return final_acc, duration, snr.item()

def benchmark():
    B, D, N = 2048, 16, 4
    x, tasks = generate_benchmark_data(B, D, N)
    
    configs = [
        ("Linear", lambda: LinearRouter(D, N)),
        ("CPR", lambda: CPRRouter(D, N)),
        ("CAPR", lambda: CAPRRouter(D, N)),
        ("MLP", lambda: MLPRouter(D, N)),
        ("Gated-MLP", lambda: GatedMLPRouter(D, N)),
        ("Tr-Local", lambda: TrLocalRouter(D, N))
    ]
    
    print(f"{'Router':<10} | {'Task':<10} | {'Acc':<6} | {'SNR':<6} | {'Time (ms)':<10}")
    print("-" * 55)
    
    for r_name, r_factory in configs:
        for t_name, y in tasks.items():
            acc, dur, snr = run_eval(r_name, r_factory(), x, y)
            print(f"{r_name:<10} | {t_name:<10} | {acc:.2f} | {snr:.2f} | {dur:.2f}")

if __name__ == "__main__":
    benchmark()
