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

    def log_step(self, epoch: int, step: int, metrics: dict[str, torch.Tensor], elapsed_time: float, eval_grid_acc: float = 0.0):
        main_loss = metrics.get('main_loss', torch.tensor(0.0)).item()
        token_acc = metrics.get('token_acc', torch.tensor(0.0)).item()
        grid_acc = metrics.get('grid_acc', torch.tensor(0.0)).item()
        sdl_loss = metrics.get('sdl_loss', torch.tensor(0.0)).item()
        moe_avg_k = metrics.get('moe_avg_k', 0.0)

        time_per_step = elapsed_time * self.config.gradient_accumulation_steps
        time_str = f"{time_per_step:.2f}s/it" if time_per_step >= 1 else f"{1/time_per_step:.2f}it/s"

        log_str = (
            f"E:{epoch:03d} S:{step:06d} | "
            f"Loss(Main/SDL):{main_loss:.3f}/{sdl_loss:.3f} | "
            f"Train Acc(Tok/Grid):{token_acc:.3f}/{grid_acc:.3f} | "
            f"Eval Grid Acc:{eval_grid_acc:.3f} | "
            f"MoE Avg K:{moe_avg_k:.2f} | "
            f"{time_str}"
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
        grid_acc = metrics.get('grid_acc', 0)
        total_count = int(metrics.get('total_count', 0))
        correct_count = int(grid_acc * total_count)
        summary_text = f"Grid Accuracy: {grid_acc:.4f} ({correct_count}/{total_count} correct)"
        self.console.print(Padding(f"[bold yellow]===== EVALUATION COMPLETE (Step: {step}) =====\n"
                                   f"{summary_text}\n"
                                   f"=============================================[/bold yellow]", (1, 2)))
