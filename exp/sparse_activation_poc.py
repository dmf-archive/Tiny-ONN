import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from rich.console import Console
from rich.live import Live
from rich.table import Table

# --- Configuration ---
CONFIG = {
    "BATCH_SIZE": 32,
    "SEQ_LEN": 16,
    "D_MODEL": 128,
    "VOCAB_SIZE": 16,
    "NUM_LAYERS": 4,
    "LR": 1e-3,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "DTYPE": torch.float32, # float32 for stable gradients
    "TRAINING_STEPS": 2000,
    "LOG_INTERVAL": 50,
    "SPARSITY_LAMBDA": 0.001, # Weight for the sparsity loss
}

# --- Core Mechanism: Gated Linear Unit with STE ---

class GateSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class GatedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Each layer has its own learnable threshold
        self.threshold = nn.Parameter(torch.tensor(0.0)) 
        self.gate_fn = GateSTE.apply

    def forward(self, x):
        # Pre-activation
        z = self.linear(x)
        
        # Gate is computed from the pre-activation magnitude
        # We use absolute value to make the gate invariant to the sign
        gate_input = z.abs() - self.threshold
        gate = self.gate_fn(gate_input)
        
        # Sparsity is the ratio of activated neurons
        sparsity = gate.mean()
        
        # Apply gate
        output = F.gelu(z) * gate
        
        return output, sparsity

# --- Model ---
class SparseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(CONFIG["VOCAB_SIZE"], CONFIG["D_MODEL"])
        
        layers = []
        for _ in range(CONFIG["NUM_LAYERS"]):
            layers.append(GatedLinear(CONFIG["D_MODEL"], CONFIG["D_MODEL"]))
        self.layers = nn.ModuleList(layers)
        
        self.lm_head = nn.Linear(CONFIG["D_MODEL"], CONFIG["VOCAB_SIZE"])

    def forward(self, x):
        x = self.embedding(x)
        total_sparsity = 0
        for layer in self.layers:
            x, sparsity = layer(x)
            total_sparsity += sparsity
        
        avg_sparsity = total_sparsity / CONFIG["NUM_LAYERS"]
        logits = self.lm_head(x)
        
        return logits, avg_sparsity

# --- Training Loop ---
def run_experiment(console):
    model = SparseMLP().to(CONFIG["DEVICE"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LR"])
    
    table = Table(title="Sparse Activation PoC")
    table.add_column("Step")
    table.add_column("Total Loss", style="magenta")
    table.add_column("CE Loss", style="cyan")
    table.add_column("Sparsity Loss", style="yellow")
    table.add_column("Avg Sparsity", style="green")

    with Live(table, console=console, screen=True, vertical_overflow="visible") as live:
        for step in range(CONFIG["TRAINING_STEPS"] + 1):
            # Simple auto-regressive task: predict the next token in a sequence
            x = torch.randint(0, CONFIG["VOCAB_SIZE"], (CONFIG["BATCH_SIZE"], CONFIG["SEQ_LEN"]), device=CONFIG["DEVICE"])
            labels = torch.roll(x, -1, dims=1)
            
            optimizer.zero_grad()
            
            logits, avg_sparsity = model(x)
            
            ce_loss = F.cross_entropy(logits.view(-1, CONFIG["VOCAB_SIZE"]), labels.view(-1))
            sparsity_loss = CONFIG["SPARSITY_LAMBDA"] * avg_sparsity
            total_loss = ce_loss + sparsity_loss
            
            total_loss.backward()
            optimizer.step()
            
            if step % CONFIG["LOG_INTERVAL"] == 0:
                live.console.clear()
                table = Table(title=f"Sparse Activation PoC (Step {step})")
                table.add_column("Step")
                table.add_column("Total Loss", style="magenta")
                table.add_column("CE Loss", style="cyan")
                table.add_column("Sparsity Loss", style="yellow")
                table.add_column("Avg Sparsity", style="green")
                table.add_row(
                    str(step),
                    f"{total_loss.item():.4f}",
                    f"{ce_loss.item():.4f}",
                    f"{sparsity_loss.item():.4f}",
                    f"{avg_sparsity.item():.4f}"
                )
                live.update(table)

if __name__ == "__main__":
    console = Console()
    run_experiment(console)