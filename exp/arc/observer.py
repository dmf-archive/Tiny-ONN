import torch
from rich.console import Console
from rich.padding import Padding
from rich.table import Table
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
        sml_loss = metrics.get('sml_loss', 0.0)
        div_loss = metrics.get('div_loss', 0.0)
        kl_loss = metrics.get('kl_loss', 0.0)
        token_acc = metrics.get('token_acc', 0.0)
        activation_rate = metrics.get('activation_rate', 0.0)
        avg_tau = metrics.get('avg_tau', 0.0)
        avg_gate = metrics.get('avg_gate', 0.0)
        pi_score = metrics.get('pi_score', 0.0)

        steps_per_sec = 1 / elapsed_time if elapsed_time > 0 else float('inf')

        log_str = (
            f"E:{epoch:2d} S:{step:5d} | Loss(m/s/d/k): {main_loss:.3f}/{sml_loss:.3f}/{div_loss:.3f}/{kl_loss:.3f} | "
            f"Acc: {token_acc:.3f} | PI: {pi_score:.3f} | Avg g: {avg_gate:.4f} | τ: {avg_tau:.3f} | "
            f"Act%: {activation_rate*100:.2f} | step/s: {steps_per_sec:.2f}"
        )
        self.console.print(log_str)

    def _create_grid_text(self, grid: torch.Tensor) -> Text:
        text = Text()
        h, w = grid.shape
        for r in range(h):
            for p in range(w):
                pixel = grid[r, p].item()
                color = self.ARC_COLORS[pixel] if 0 <= pixel < len(self.ARC_COLORS) else "white"
                text.append("■ ", style=color)
            if r < h - 1:
                text.append("\n")
        return text

    def visualize_evaluation_sample(self, input_grid: torch.Tensor, target_grid: torch.Tensor, pred_grid: torch.Tensor, step: int):
        self.console.print()
        self.console.print(f"--- Evaluation Sample @ Step {step} ---", style="bold yellow")

        if pred_grid is None:
            pred_grid = torch.zeros((1, 1), dtype=torch.long)

        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column("Input", justify="center")
        table.add_column("Target", justify="center")
        table.add_column("Prediction", justify="center")

        input_text = self._create_grid_text(input_grid)
        target_text = self._create_grid_text(target_grid)
        pred_text = self._create_grid_text(pred_grid)

        table.add_row(input_text, target_text, pred_text)
        self.console.print(table)
        self.console.print()

    def log_eval_summary(self, metrics: dict[str, float], step: int):
        eval_loss = metrics.get('eval_loss', 0.0)
        eval_grid_acc = metrics.get('eval_grid_acc', 0.0)
        total_count = int(metrics.get('total_count', 0))
        summary_text = f"Avg Loss: {eval_loss:.4f} | Grid Accuracy: {eval_grid_acc:.4f} on {total_count} samples."
        self.console.print(Padding(f"[bold yellow]===== EVALUATION COMPLETE (Step: {step}) =====\n"
                                   f"{summary_text}\n"
                                   f"=============================================[/bold yellow]", (1, 2)))
