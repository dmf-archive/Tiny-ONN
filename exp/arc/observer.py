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
        steps_per_sec = 1 / elapsed_time if elapsed_time > 0 else float('inf')

        table = Table(title=f"Epoch {epoch} | Step {step}", show_header=True, header_style="bold magenta", expand=True)
        
        # Row 1: Headers
        table.add_column("Loss (Main/IBS)", justify="center")
        table.add_column("Accuracy", justify="center")
        table.add_column("PI Score", justify="center")
        table.add_column("Proto (Norm/Avg)", justify="center")
        table.add_column("τ (Tau)", justify="center")
        table.add_column("Act % (L0/Avg/LN)", justify="center")
        table.add_column("Seq (Len/Entr)", justify="center")
        table.add_column("Speed", justify="center")

        # Row 2: Values
        act_l0 = metrics.get('activation_rate_l0', 0.0) * 100
        act_avg = metrics.get('activation_rate_avg', 0.0) * 100
        act_ln = metrics.get('activation_rate_ln', 0.0) * 100
        table.add_row(
            f"{metrics.get('main_loss', 0.0):.3f}/{metrics.get('ibs_loss', 0.0):.3f}",
            f"{metrics.get('token_acc', 0.0):.3f}",
            f"{metrics.get('pi_score', 0.0):.3f}",
            f"{metrics.get('proto_norm', 0.0):.3f} / {metrics.get('avg_proto', 0.0):.4f}",
            f"{metrics.get('avg_tau', 0.0):.3f}",
            f"{act_l0:.1f}/{act_avg:.1f}/{act_ln:.1f}",
            f"{int(metrics.get('seq_len', 0))} / {metrics.get('seq_entropy', 0.0):.2f}",
            f"{steps_per_sec:.2f} st/s"
        )
        
        self.console.print(table)

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
