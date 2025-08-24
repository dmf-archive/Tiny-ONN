import torch
import torch.nn as nn
import torch.nn.functional as F
import bayesian_torch.layers as bl
from bayesian_torch.models.dnn_to_bnn import get_kl_loss
from rich.console import Console
from rich.table import Table

# --- Configuration ---
CONFIG = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "DTYPE": torch.bfloat16,
    "BATCH_SIZE": 4,
    "INPUT_DIM": 16,
    "OUTPUT_DIM": 10,
    "HIDDEN_DIM": 32,
    "PI_ALPHA": 8,
    "PI_GAMMA": 0.5,
    "LR": 1e-3,
    "KL_WEIGHT": 0.1,  # Significantly increased weight
    "POSTERIOR_RHO_INIT": -3.0,
}

console = Console()

# --- Simplified Model ---
class SimpleBNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.blinear1 = bl.LinearReparameterization(
            CONFIG["INPUT_DIM"], CONFIG["HIDDEN_DIM"], posterior_rho_init=CONFIG["POSTERIOR_RHO_INIT"]
        )
        self.blinear2 = bl.LinearReparameterization(
            CONFIG["HIDDEN_DIM"], CONFIG["OUTPUT_DIM"], posterior_rho_init=CONFIG["POSTERIOR_RHO_INIT"]
        )

    def forward(self, x):
        x, _ = self.blinear1(x)
        x = F.relu(x)
        x, _ = self.blinear2(x)
        return x

def log_results(model, initial_params, final_params):
    table = Table(title="Gradient and Parameter Update Analysis")
    table.add_column("Parameter", style="cyan")
    table.add_column("Grad Norm", style="magenta")
    table.add_column("Initial Value (mean)", style="green")
    table.add_column("Final Value (mean)", style="yellow")
    table.add_column("Updated?", style="bold")

    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norm = p.grad.norm().item()
            initial_val = initial_params[name].mean().item()
            final_val = final_params[name].mean().item()
            updated = "✅" if abs(initial_val - final_val) > 1e-6 else "❌"
            table.add_row(name, f"{grad_norm:.6f}", f"{initial_val:.6f}", f"{final_val:.6f}", updated)
        else:
            table.add_row(name, "[red]None[/red]", "-", "-", "❌")
    
    console.print(table)


def simplified_eavi_grad_calc(model, optimizer, input_data, labels):
    batch_size = input_data.shape[0]
    grad_norms = torch.zeros(batch_size, CONFIG["PI_ALPHA"], device=CONFIG["DEVICE"])
    
    all_total_losses_per_sample = []
    for k in range(CONFIG["PI_ALPHA"]):
        logits = model(input_data)
        task_loss_per_sample = F.cross_entropy(logits, labels, reduction='none')
        # We now use KL_WEIGHT instead of dividing by a large number
        kl_loss = get_kl_loss(model)
        total_loss_per_sample = task_loss_per_sample + CONFIG["KL_WEIGHT"] * kl_loss
        all_total_losses_per_sample.append(total_loss_per_sample)

        for i in range(batch_size):
            grads = torch.autograd.grad(total_loss_per_sample[i], model.parameters(), retain_graph=True, allow_unused=True)
            flat_grads = torch.cat([g.view(-1) for g in grads if g is not None])
            grad_norms[i, k] = torch.linalg.norm(flat_grads)

    weights = F.softmax(-CONFIG["PI_GAMMA"] * grad_norms, dim=1)
    total_losses_tensor = torch.stack(all_total_losses_per_sample, dim=1)
    final_weighted_loss = (weights * total_losses_tensor).sum() / batch_size

    optimizer.zero_grad(set_to_none=True)
    final_weighted_loss.backward()
    optimizer.step()


def run_poc(poc_name: str, grad_calculation_method: callable):
    console.print(f"\n--- Running PoC: [bold cyan]{poc_name}[/bold cyan] (KL_WEIGHT={CONFIG['KL_WEIGHT']}) ---")
    
    torch.manual_seed(42)
    model = SimpleBNN().to(CONFIG["DEVICE"], dtype=CONFIG["DTYPE"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["LR"])
    
    input_data = torch.randn(CONFIG["BATCH_SIZE"], CONFIG["INPUT_DIM"], device=CONFIG["DEVICE"], dtype=CONFIG["DTYPE"])
    labels = torch.randint(0, CONFIG["OUTPUT_DIM"], (CONFIG["BATCH_SIZE"],), device=CONFIG["DEVICE"])

    initial_params = {name: p.clone().detach() for name, p in model.named_parameters()}
    
    grad_calculation_method(model, optimizer, input_data, labels)

    final_params = {name: p.clone().detach() for name, p in model.named_parameters()}
    log_results(model, initial_params, final_params)


if __name__ == "__main__":
    run_poc("Simplified EAVI (Loss Weighting)", simplified_eavi_grad_calc)
