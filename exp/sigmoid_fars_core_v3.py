import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import track
from collections import Counter

# -----------------------------------------------------------------------------
# 0. 环境配置与依赖注入
# -----------------------------------------------------------------------------
# 注入 ARS2-Neo 路径
sys.path.append(os.path.join(os.getcwd(), "ref", "ARS"))
try:
    from optimizer.ars2_neo import SingleDeviceARS2Neo
except ImportError:
    print("Error: Could not import SingleDeviceARS2Neo. Please ensure ref/ARS exists.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# 1. 核心组件：Sigmoid Router & Experts
# -----------------------------------------------------------------------------

class SigmoidRouter(nn.Module):
    def __init__(self, input_dim, num_experts, temperature=0.1):
        super().__init__()
        # 使用更深的网络以增强路由的非线性决策能力
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.SiLU(),
            nn.Linear(input_dim * 2, num_experts)
        )
        self.temperature = temperature

    def forward(self, x):
        # Sigmoid 允许独立概率，不进行归一化
        logits = self.net(x) / self.temperature
        return torch.sigmoid(logits)

class ToyExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------------
# 2. 生产级 FARS 引擎 (基于 ARS2-Neo 真实状态)
# -----------------------------------------------------------------------------

class RealFARSEngine:
    def __init__(self, optimizer, experts, lambda_fars=0.05):
        self.optimizer = optimizer
        self.experts = experts
        self.lambda_fars = lambda_fars

    def get_expert_costs(self, device):
        """
        从优化器状态中提取真实的 Fisher 信息 (v_t)
        Cost_i = Mean(Sqrt(v_t)) over all params in Expert_i
        """
        costs = []
        for expert in self.experts:
            expert_cost_accum = 0.0
            param_count = 0
            
            for p in expert.parameters():
                if p in self.optimizer.state:
                    state = self.optimizer.state[p]
                    if 'exp_avg_sq' in state:
                        # v_t 是二阶矩估计
                        v_t = state['exp_avg_sq']
                        # 使用 sqrt(v_t) 近似 Fisher 信息对角线
                        # 取 mean 以消除参数量级影响
                        expert_cost_accum += v_t.sqrt().mean().item()
                        param_count += 1
            
            if param_count > 0:
                # 平均每个参数组的代价
                costs.append(expert_cost_accum / param_count)
            else:
                # 优化器尚未初始化状态（第一步）
                costs.append(0.0)
        
        # 归一化 Costs 以稳定训练 (可选，但推荐)
        costs_tensor = torch.tensor(costs, device=device, dtype=torch.float32)
        if costs_tensor.sum() > 0:
            costs_tensor = costs_tensor / (costs_tensor.mean() + 1e-8)
            
        return costs_tensor

    def compute_shaping_loss(self, beliefs):
        """
        计算 FARS 塑造损失
        L_fars = Mean(Belief * Cost)
        """
        costs = self.get_expert_costs(beliefs.device)
        # beliefs: [Batch, Num_Experts]
        # costs: [Num_Experts]
        # 我们希望最小化高 Cost 专家的 Belief
        shaping_loss = (beliefs * costs.unsqueeze(0)).sum(dim=1).mean()
        return shaping_loss * self.lambda_fars, costs

# -----------------------------------------------------------------------------
# 3. 实验环境：ARC 任务模拟器 (增强版)
# -----------------------------------------------------------------------------

class ARCTaskSimulator:
    def __init__(self, dim=32):
        self.dim = dim
        # 定义具有不同几何/语义特性的任务
        self.tasks = {
            "Identity": lambda x: x,
            "Invert": lambda x: -x,
            "Shift": lambda x: torch.roll(x, shifts=1, dims=-1),
            "Scale": lambda x: x * 2.0,
            "Permute": lambda x: x[:, torch.randperm(dim)], # 随机置换，模拟复杂映射
            "ZeroOut": lambda x: x * (torch.rand_like(x) > 0.5).float() # 模拟部分信息丢失
        }
        self.task_names = list(self.tasks.keys())

    def sample(self, batch_size=32):
        task_idx = np.random.randint(len(self.task_names))
        task_name = self.task_names[task_idx]
        
        x = torch.randn(batch_size, self.dim)
        y = self.tasks[task_name](x)
        
        return x, y, task_idx, task_name

# -----------------------------------------------------------------------------
# 4. 指标计算：RMI (Routing Mutual Information)
# -----------------------------------------------------------------------------

class RMICalculator:
    def __init__(self, num_experts):
        self.num_experts = num_experts
        self.history = [] # List of (task_idx, active_mask_int)

    def update(self, task_idxs, beliefs, threshold=0.5):
        # 将 beliefs 二值化并转换为整数表示状态
        active_masks = (beliefs > threshold).int()
        # 转换为整数 hash
        # [B, E] -> [B]
        powers = 2 ** torch.arange(self.num_experts, device=beliefs.device)
        state_ints = (active_masks * powers).sum(dim=1).cpu().numpy()
        
        for t, s in zip(task_idxs, state_ints):
            self.history.append((t, s))

    def compute(self):
        if not self.history:
            return 0.0
            
        total = len(self.history)
        task_counts = Counter()
        state_counts = Counter()
        joint_counts = Counter()
        
        for t, s in self.history:
            task_counts[t] += 1
            state_counts[s] += 1
            joint_counts[(t, s)] += 1
            
        # H(R)
        h_r = 0.0
        for s, count in state_counts.items():
            p = count / total
            h_r -= p * np.log2(p + 1e-10)
            
        # H(R|T) = Sum P(t) * H(R|t)
        h_r_given_t = 0.0
        for t, t_count in task_counts.items():
            p_t = t_count / total
            # H(R|t)
            sub_h = 0.0
            # 找到所有该 task 下的 state
            # 优化：预先分组会更快，但这里数据量不大直接遍历
            # 为了效率，我们在 joint_counts 里找
            current_task_states = {k[1]: v for k, v in joint_counts.items() if k[0] == t}
            for s, joint_c in current_task_states.items():
                p_s_given_t = joint_c / t_count
                sub_h -= p_s_given_t * np.log2(p_s_given_t + 1e-10)
            
            h_r_given_t += p_t * sub_h
            
        # I(R; T) = H(R) - H(R|T)
        return h_r - h_r_given_t

# -----------------------------------------------------------------------------
# 5. 主实验流程
# -----------------------------------------------------------------------------

def run_experiment():
    console = Console()
    dim = 32
    num_experts = 8
    batch_size = 128
    epochs = 2000 # 增加轮数以确保收敛
    
    simulator = ARCTaskSimulator(dim)
    router = SigmoidRouter(dim, num_experts, temperature=0.1)
    experts = nn.ModuleList([ToyExpert(dim) for _ in range(num_experts)])
    
    # 使用生产级优化器 ARS2-Neo
    # 注意：我们将 router 和 experts 放在不同的参数组，以便未来可以应用不同的优化策略
    optimizer = SingleDeviceARS2Neo([
        {'params': router.parameters(), 'lr': 1e-3, 'rho': 0.05}, # Router 需要更稳
        {'params': experts.parameters(), 'lr': 1e-3, 'rho': 0.1}
    ])
    
    fars = RealFARSEngine(optimizer, experts, lambda_fars=0.1)
    rmi_calc = RMICalculator(num_experts)
    criterion = nn.MSELoss()
    
    console.print("[bold cyan]Starting Sigmoid-FARS Production Experiment (ARS2-Neo Powered)...[/bold cyan]")
    
    # 记录指标
    metrics = {
        "loss": [],
        "sparsity": [],
        "rmi": [],
        "itjd": []
    }
    
    # 用于 ITJD 计算
    task_routing_accum = {name: torch.zeros(num_experts) for name in simulator.task_names}
    task_counts = {name: 0 for name in simulator.task_names}

    for epoch in track(range(epochs), description="Training"):
        x, y_true, task_idx, task_name = simulator.sample(batch_size)
        
        optimizer.zero_grad()
        
        # 定义闭包供 ARS2-Neo 使用
        def closure():
            beliefs = router(x)
            expert_outputs = torch.stack([e(x) for e in experts], dim=1)
            # 加权求和 (Sigmoid 允许非归一化叠加)
            y_pred = (beliefs.unsqueeze(-1) * expert_outputs).sum(dim=1)
            
            main_loss = criterion(y_pred, y_true)
            shaping_loss, _ = fars.compute_shaping_loss(beliefs)
            
            return main_loss + shaping_loss

        # 执行优化步
        loss = optimizer.step(closure)
        
        # --- 统计与记录 (在 step 之后，使用更新后的状态) ---
        with torch.no_grad():
            beliefs = router(x)
            
            # 更新 RMI
            rmi_calc.update([task_idx]*batch_size, beliefs)
            
            # 更新 ITJD 累积
            avg_belief = beliefs.mean(dim=0).cpu()
            task_routing_accum[task_name] += avg_belief
            task_counts[task_name] += 1
            
            # 记录稀疏度
            sparsity = (beliefs < 0.05).float().mean().item()
            metrics["sparsity"].append(sparsity)
            metrics["loss"].append(loss.item())

    # -------------------------------------------------------------------------
    # 6. 结果分析
    # -------------------------------------------------------------------------
    
    # 计算最终 RMI
    final_rmi = rmi_calc.compute()
    
    # 计算 ITJD
    def jaccard_dist(a, b):
        intersection = torch.min(a, b).sum()
        union = torch.max(a, b).sum()
        return 1.0 - (intersection / (union + 1e-8)).item()

    avg_routing = {}
    for name in simulator.task_names:
        if task_counts[name] > 0:
            avg_routing[name] = task_routing_accum[name] / task_counts[name]
        else:
            avg_routing[name] = torch.zeros(num_experts)

    table = Table(title="Sigmoid-FARS Production Analysis")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Final RMI (Routing Mutual Info)", f"{final_rmi:.4f}")
    table.add_row("Avg Sparsity (<0.05)", f"{np.mean(metrics['sparsity'][-100:]):.4f}")
    table.add_row("Final Loss", f"{np.mean(metrics['loss'][-100:]):.6f}")
    
    console.print(table)
    
    # ITJD Matrix
    itjd_table = Table(title="Inter-Task Jaccard Distance (ITJD)")
    itjd_table.add_column("Task Pair", style="green")
    itjd_table.add_column("Distance", style="yellow")
    
    pairs = []
    task_names = list(avg_routing.keys())
    for i in range(len(task_names)):
        for j in range(i + 1, len(task_names)):
            t1, t2 = task_names[i], task_names[j]
            dist = jaccard_dist(avg_routing[t1], avg_routing[t2])
            pairs.append((dist, f"{t1} vs {t2}"))
    
    # 展示 Top 5 隔离度最高的任务对
    pairs.sort(key=lambda x: x[0], reverse=True)
    for dist, name in pairs[:5]:
        itjd_table.add_row(name, f"{dist:.4f}")
        
    console.print(itjd_table)
    
    # 打印最终的 Cost 分布
    final_costs = fars.get_expert_costs(torch.device('cpu'))
    console.print(f"\n[bold]Final Expert Fisher Costs (Normalized):[/bold]\n{final_costs.numpy()}")
    console.print(f"[bold]Global Expert Activation Mean:[/bold]\n{torch.stack(list(avg_routing.values())).mean(dim=0).numpy()}")

if __name__ == "__main__":
    run_experiment()
