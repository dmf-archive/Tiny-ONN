import torch
import torch.nn as nn
import bayesian_torch.layers as bl
from bayesian_torch.models.dnn_to_bnn import get_kl_loss
from rich.console import Console
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONFIG = {"LR": 1e-3, "KL_WEIGHT": 0.01, "GAMMA": 1.0, "ALPHA": 4}
console = Console()

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blinear = bl.LinearReparameterization(1, 1, posterior_rho_init=-3.0)

    def forward(self, x):
        return self.blinear(x)[0]

def get_total_loss(model, x, y):
    pred = model(x)
    task_loss = nn.functional.mse_loss(pred, y)
    kl_loss = get_kl_loss(model)
    return task_loss + CONFIG["KL_WEIGHT"] * kl_loss

def analyze_gradients(loss, model, title):
    model.zero_grad()
    loss.backward(retain_graph=True)
    console.print(f"\n--- [bold cyan]{title}[/bold cyan] ---")
    for name, param in model.named_parameters():
        console.print(f"{name}: grad = {param.grad.item():.6f}")

def main():
    torch.manual_seed(0)
    model = SimpleModel()
    x = torch.tensor([[1.0]])
    y = torch.tensor([[10.0]])
    
    std_loss = get_total_loss(model, x, y)
    analyze_gradients(std_loss, model, "Standard Loss Gradients")

    losses = [get_total_loss(model, x, y) for _ in range(CONFIG["ALPHA"])]
    surprises = [torch.cat([g.view(-1) for g in torch.autograd.grad(l, model.parameters(), retain_graph=True)]).norm() for l in losses]
    weights = nn.functional.softmax(-CONFIG["GAMMA"] * torch.stack(surprises), dim=0)
    eavi_loss = (torch.stack(losses) * weights).sum()
    analyze_gradients(eavi_loss, model, "EAVI Loss Gradients")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    mu_vals = torch.linspace(model.blinear.mu_weight.item() - 2, model.blinear.mu_weight.item() + 2, 100)
    rho_vals = torch.linspace(model.blinear.rho_weight.item() - 1, model.blinear.rho_weight.item() + 1, 100)
    
    base_state_dict = model.state_dict()
    
    def get_loss_for_val(param_name, val):
        temp_model = SimpleModel()
        temp_state_dict = base_state_dict.copy()
        temp_state_dict[param_name] = torch.tensor([[val]])
        temp_model.load_state_dict(temp_state_dict)
        return get_total_loss(temp_model, x, y).item()

    std_loss_mu = [get_loss_for_val('blinear.mu_weight', v) for v in mu_vals]
    ax1.plot(mu_vals.numpy(), std_loss_mu)
    ax1.set_title('Loss vs. Mu')
    ax1.set_xlabel('Mu Value')
    ax1.set_ylabel('Standard Loss')
    
    std_loss_rho = [get_loss_for_val('blinear.rho_weight', v) for v in rho_vals]
    ax2.plot(rho_vals.numpy(), std_loss_rho)
    ax2.set_title('Loss vs. Rho')
    ax2.set_xlabel('Rho Value')
    
    plt.tight_layout()
    plt.savefig("exp/loss_landscape.png")
    console.print("\n[green]Loss landscape saved to exp/loss_landscape.png[/green]")

if __name__ == "__main__":
    main()