import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from rich.console import Console
from rich.table import Table
from rich.progress import track
import sys
import os
import numpy as np

# -----------------------------------------------------------------------------
# 0. 依赖注入
# -----------------------------------------------------------------------------
sys.path.append(os.getcwd())
from src.models.dynsiha.flat.configuration_flat_dynsiha import FlatDynSIHAConfig
from src.models.dynsiha.flat.modeling_flat_dynsiha import FlatDynSIHAForCausalLM

sys.path.append(os.path.join(os.getcwd(), "ref", "ARS"))
from optimizer.ars2_neo import SingleDeviceARS2Neo

# -----------------------------------------------------------------------------
# 1. 真正的 FARS 塑造器：从 ARS2-Neo 提取 Fisher 信息
# -----------------------------------------------------------------------------
class RealFisherShaper:
    def __init__(self, optimizer, lambda_fars=0.01):
        self.optimizer = optimizer
        self.lambda_fars = lambda_fars
        self.console = Console()

    def get_expert_costs(self, model):
        """
        从优化器状态中提取专家的 Fisher 信息近似 (sqrt(v_t))
        """
        costs = {}
        # 遍历模型中所有的专家权重
        for name, module in model.named_modules():
            if "experts" in name and hasattr(module, "weight"):
                # 假设专家权重在 VectorizedExpertMLP 中是合并存储的
                # 我们需要根据专家索引拆分 v_t
                if module.weight in self.optimizer.state:
                    state = self.optimizer.state[module.weight]
                    if "exp_avg_sq" in state:
                        v_t = state["exp_avg_sq"]
                        # 计算每个专家的平均 Fisher 强度
                        # 假设权重形状为 [num_experts, out_features, in_features]
                        expert_v = v_t.mean(dim=(1, 2)) 
                        costs[name] = torch.sqrt(expert_v + 1e-8)
        return costs

    def compute_fars_loss(self, model, model_output):
        if not hasattr(model_output, "routing_weights"):
            return torch.tensor(0.0, device=next(model.parameters()).device)

        expert_costs = self.get_expert_costs(model)
        total_fars_loss = 0.0
        
        # 简化的 FARS：Belief * Real_Fisher_Cost
        for layer_idx, weights_dict in enumerate(model_output.routing_weights):
            w = weights_dict["mlp"] # 聚焦 MLP 专家路由
            # 寻找对应的专家层成本
            # 注意：这里的映射逻辑需要与模型架构对齐
            cost_key = f"model.layers.{layer_idx}.mlp.experts"
            if cost_key in expert_costs:
                cost = expert_costs[cost_key].detach() # Cost 作为环境场，不传梯度
                # 归一化成本场以保持量级稳定
                cost = cost / (cost.mean() + 1e-8)
                
                # 𝒢 = Σ (Belief * Cost)
                fars_term = (w * cost).sum(dim=-1).mean()
                
                # 附加极化约束，防止坍塌
                polarization = 1.0 - (w**2).sum(dim=-1).mean()
                
                total_fars_loss += fars_term + 0.5 * polarization

        return total_fars_loss * self.lambda_fars

# -----------------------------------------------------------------------------
# 2. 实验运行
# -----------------------------------------------------------------------------
def run_mnist_fars():
    console = Console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 配置 FDS 架构
    config = FlatDynSIHAConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_heads=4,
        num_experts=10, # 对应 10 个数字
        top_k=10,
        vocab_size=256, # 像素值
        max_position_embeddings=784 # 28x28
    )
    model = FlatDynSIHAForCausalLM(config).to(device)

    # 准备 MNIST 数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).long().view(-1)) # 展平并转为 token
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    # 缩小规模以快速验证
    train_dataset = Subset(train_dataset, range(2000))
    loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    optimizer = SingleDeviceARS2Neo(model.parameters(), lr=1e-3, rho=0.05)
    shaper = RealFisherShaper(optimizer, lambda_fars=0.05)

    console.print("[bold blue]Starting MNIST-FDS Real FARS Experiment...[/bold blue]")

    for epoch in range(5):
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for x, y in track(loader, description=f"Epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # 模拟分类任务：取序列最后一个 token 的输出进行分类
            def closure():
                output = model(input_ids=x, return_dict=True)
                logits = output.logits[:, -1, :10] # 映射到 10 分类
                ce_loss = F.cross_entropy(logits, y)
                
                fars_loss = shaper.compute_fars_loss(model, output)
                return ce_loss + fars_loss
            
            loss = optimizer.step(closure)
            total_loss += loss.item()
            
            # 统计准确率
            with torch.no_grad():
                output = model(input_ids=x, return_dict=True)
                pred = output.logits[:, -1, :10].argmax(dim=-1)
                correct += (pred == y).sum().item()
                total_samples += y.size(0)

        avg_acc = correct / total_samples
        console.print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f} | Acc: {avg_acc:.4f}")

if __name__ == "__main__":
    run_mnist_fars()
