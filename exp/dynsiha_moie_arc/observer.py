import torch
import torch.nn.functional as F
from rich.columns import Columns
from rich.console import Console
from rich.padding import Padding
from rich.text import Text

from .config import TrainConfig


class Observer:
    ARC_COLORS: list[str] = [
        "black", "blue", "red", "green", "yellow",
        "grey", "magenta", "orange", "cyan", "brown"
    ]

    def __init__(self, console: Console, config: TrainConfig):
        self.console = console
        self.config = config

    def log_step(self, epoch: int, step: int, metrics: dict[str, float], elapsed_time: float):
        main_loss = metrics.get('main_loss', 0.0)
        gate_loss = metrics.get('gate_loss', 0.0)
        kl_loss = metrics.get('kl_loss', 0.0)
        token_acc = metrics.get('token_acc', 0.0)
        activation_rate = metrics.get('activation_rate', 0.0)
        avg_tau = metrics.get('avg_tau', 0.0)
        prior_std = metrics.get('prior_std', 0.0)
        avg_sigma = metrics.get('avg_sigma', 0.0)
        avg_gate = metrics.get('avg_gate', 0.0)
        
        steps_per_sec = 1 / elapsed_time if elapsed_time > 0 else float('inf')

        log_str = (
            f"E:{epoch:2d} S:{step:5d} | Loss(m/g/k): {main_loss:.3f}/{gate_loss:.3f}/{kl_loss:.3f} | "
            f"Acc: {token_acc:.3f} | Avg σ/g: {avg_sigma:.4f}/{avg_gate:.4f} | τ/p_std: {avg_tau:.3f}/{prior_std:.3f} | "
            f"Act%: {activation_rate*100:.2f} | step/s: {steps_per_sec:.2f}"
        )
        self.console.print(log_str)

    def _create_grid_text(self, grid: torch.Tensor, title: str) -> Text:
        text = Text(f"{title}\n", style="bold")
        h, w = grid.shape
        for r in range(h):
            for p in range(w):
                pixel = grid[r, p].item()
                color = self.ARC_COLORS[pixel] if 0 <= pixel < len(self.ARC_COLORS) else "white"
                text.append("■ ", style=color)
            text.append("\n")
        return text

    def visualize_evaluation_sample(self, input_grid: torch.Tensor, target_grid: torch.Tensor, pred_grid: torch.Tensor, step: int):
        self.console.print()
        self.console.print(f"--- Evaluation Sample @ Step {step} ---", style="bold yellow")

        if pred_grid is None:
            pred_grid = torch.zeros((1, 1), dtype=torch.long)
        
        input_text = self._create_grid_text(input_grid, "Input")
        target_text = self._create_grid_text(target_grid, "Target")
        pred_text = self._create_grid_text(pred_grid, "Prediction")

        self.console.print(Columns([input_text, target_text, pred_text], expand=True, equal=True))
        self.console.print()

    def log_eval_summary(self, metrics: dict[str, float], step: int):
        eval_loss = metrics.get('eval_loss', 0)
        eval_acc = metrics.get('eval_acc', 0)
        total_count = int(metrics.get('total_count', 0))
        summary_text = f"Avg Loss: {eval_loss:.4f} | Token Accuracy: {eval_acc:.4f} on {total_count} samples."
        self.console.print(Padding(f"[bold yellow]===== EVALUATION COMPLETE (Step: {step}) =====\n"
                                   f"{summary_text}\n"
                                   f"=============================================[/bold yellow]", (1, 2)))
